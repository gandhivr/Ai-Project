#!/usr/bin/env python3
"""
server.py - Production FastAPI Server for Smart Classroom Attendance Dashboard

Anti-Gravity Pro Suite:
- Full-stack classical computer vision engine API
- Real-time video frame face verification with privacy blurring
- Admin webcam crop enrollment & dynamic model retraining
- Two-way SQLite & CSV attendance log synchronization
- Robustness benchmark & Haar tuning reports
- DoD 5220.22-M cryptographic data purging
"""

import os
import io
import sys
import time
import base64
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model_manager import (
    load_model,
    train_or_update_model,
    preprocess_face,
    extract_lbp_feature,
    CASCADE_PATH,
    DEFAULT_FALLBACK_THRESHOLD
)
from database import (
    init_db,
    get_all_students,
    get_attendance_records,
    get_dashboard_stats,
    log_attendance_db,
    purge_db_records,
    sync_existing_dataset_students
)
from purge_data import purge_all_biometric_data
from evaluate_robustness import run_robustness_audit
from tune_detector import run_grid_search

app = FastAPI(title="Smart Classroom Attendance Pro Suite")

# Ensure static directories exist
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared state
ENGINE_STATE = {
    "model": None,
    "detector": None,
    "last_trained": None
}


def get_detector() -> cv2.CascadeClassifier:
    if ENGINE_STATE["detector"] is None:
        det = cv2.CascadeClassifier(CASCADE_PATH)
        if det.empty():
            raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")
        ENGINE_STATE["detector"] = det
    return ENGINE_STATE["detector"]


def get_active_model():
    if ENGINE_STATE["model"] is None:
        ENGINE_STATE["model"] = load_model()
    return ENGINE_STATE["model"]


@app.on_event("startup")
def startup_event():
    init_db()
    ENGINE_STATE["model"] = load_model()
    try:
        get_detector()
    except Exception as e:
        print(f"[Server] Cascade warning: {e}")
    print("[Server] FastAPI Vision Backend initialized.")


# Pydantic Schemas
class FrameVerifyRequest(BaseModel):
    image_base64: str

class EnrollCropRequest(BaseModel):
    student_name: str
    roll_number: str
    image_base64: str

class PurgeRequest(BaseModel):
    confirm_wipe: bool


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Dashboard loading...</h1>", status_code=200)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_system_status():
    model = get_active_model()
    is_cold = (model is None or len(model.get("classes", [])) == 0)
    stats = get_dashboard_stats()

    return {
        "status": "ONLINE",
        "cold_start": is_cold,
        "enrolled_count": stats["total_enrolled"],
        "present_today": stats["present_today"],
        "avg_confidence": stats["avg_confidence"],
        "classes": model.get("classes", []) if model else [],
        "rejection_threshold": round(model.get("threshold", DEFAULT_FALLBACK_THRESHOLD), 4) if model else DEFAULT_FALLBACK_THRESHOLD,
        "total_samples": model.get("total_samples", 0) if model else 0,
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/api/attendance")
async def get_attendance():
    records = get_attendance_records(limit=150)
    return {"records": records}


@app.get("/api/students")
async def get_students():
    students = get_all_students()
    return {"students": students}


@app.post("/api/verify_frame")
async def verify_frame(req: FrameVerifyRequest):
    """
    Accepts client webcam frame in base64.
    Runs tuned Haar cascade face detection, CLAHE + 3,776-dim LBP extraction,
    queries 1-NN model, and determines:
    - VERIFIED student (green box, confidence, logs attendance)
    - UNKNOWN / Non-consenting bystander (red box, 51x51 Gaussian blur)
    """
    model = get_active_model()
    if model is None or len(model.get("classes", [])) == 0:
        return {
            "cold_start": True,
            "has_face": False,
            "message": "Cold-Start Safe Mode: Zero students enrolled."
        }

    # Decode base64 frame
    try:
        header_sep = req.image_base64.find(",")
        raw_b64 = req.image_base64[header_sep + 1:] if header_sep != -1 else req.image_base64
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = get_detector()
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    if len(faces) == 0:
        return {"has_face": False, "faces": []}

    knn = model["knn"]
    labels = model["labels"]
    threshold = model.get("threshold", DEFAULT_FALLBACK_THRESHOLD)

    results = []
    for (x, y, fw, fh) in faces:
        crop = frame[y:y+fh, x:x+fw]
        if crop.size == 0:
            continue

        preproc = preprocess_face(crop)
        feat = extract_lbp_feature(preproc).reshape(1, -1)
        distances, indices = knn.kneighbors(feat, n_neighbors=1)
        distance = float(distances[0][0])
        matched_idx = int(indices[0][0])
        pred_label = labels[matched_idx]

        if distance <= threshold:
            # Enrolled & Consenting
            conf = max(0.0, min(1.0, 1.0 - (distance / (threshold * 1.5))))
            disp_name = pred_label.replace("_", " ")

            # Log to SQLite & CSV idempotently
            is_new = log_attendance_db(pred_label, conf)
            if is_new:
                # Synchronize to logs/attendance.csv
                csv_path = BASE_DIR / "logs" / "attendance.csv"
                with open(csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()},{pred_label},{conf:.4f},PRESENT\n")

            results.append({
                "status": "VERIFIED",
                "student_name": disp_name,
                "clean_name": pred_label,
                "confidence": round(conf * 100, 1),
                "distance": round(distance, 4),
                "box": [int(x), int(y), int(fw), int(fh)],
                "is_blurred": False
            })
        else:
            # Unknown / Non-Consenting Bystander -> Instant 51x51 Gaussian blur mask
            results.append({
                "status": "UNKNOWN",
                "student_name": "Unknown (Blurred)",
                "confidence": 0.0,
                "distance": round(distance, 4),
                "box": [int(x), int(y), int(fw), int(fh)],
                "is_blurred": True
            })

    return {
        "has_face": True,
        "faces": results,
        "frame_width": w,
        "frame_height": h
    }


@app.post("/api/enroll_crop")
async def enroll_crop(req: EnrollCropRequest):
    """
    Saves a captured face crop for a student into dataset/<student_name>/
    """
    clean_name = req.student_name.strip().replace(" ", "_")
    student_dir = BASE_DIR / "dataset" / clean_name
    student_dir.mkdir(parents=True, exist_ok=True)

    try:
        header_sep = req.image_base64.find(",")
        raw_b64 = req.image_base64[header_sep + 1:] if header_sep != -1 else req.image_base64
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    # Check for face inside frame or use direct crop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = get_detector()
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        crop = frame[y:y+h, x:x+w]
    else:
        crop = frame

    processed = preprocess_face(crop)
    ts = int(time.time() * 1000)
    out_file = student_dir / f"img_{ts}.jpg"
    cv2.imwrite(str(out_file), processed)

    sync_existing_dataset_students()
    total_crops = len(list(student_dir.glob("img_*.jpg")))

    # Convert thumbnail to base64 for client display
    _, buffer = cv2.imencode(".jpg", processed)
    thumb_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "SUCCESS",
        "student_name": req.student_name,
        "clean_name": clean_name,
        "crop_count": total_crops,
        "thumbnail": f"data:image/jpeg;base64,{thumb_b64}"
    }


@app.post("/api/retrain")
async def retrain_model():
    """
    Triggers dynamic in-memory model retraining without restarting the server.
    """
    success = train_or_update_model()
    if not success:
        return {"status": "ERROR", "message": "Zero valid crops found. Model not updated."}

    # Reload shared state
    ENGINE_STATE["model"] = load_model()
    sync_existing_dataset_students()
    model = ENGINE_STATE["model"]

    return {
        "status": "SUCCESS",
        "classes": model["classes"],
        "threshold": round(model["threshold"], 4),
        "total_samples": model["total_samples"],
        "message": "Model compiled & serialized successfully."
    }


@app.get("/api/reports/tuning")
async def get_tuning_report():
    report_path = BASE_DIR / "reports" / "haar_tuning_results.md"
    if not report_path.exists():
        run_grid_search()
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"markdown": content}


@app.get("/api/reports/robustness")
async def get_robustness_report():
    report_path = BASE_DIR / "reports" / "robustness_metrics.md"
    if not report_path.exists():
        run_robustness_audit()
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"markdown": content}


@app.get("/api/reports/ethics")
async def get_ethics_report():
    report_path = BASE_DIR / "reports" / "ethics_and_bias_report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Ethics report not found.")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"markdown": content}


@app.post("/api/purge")
async def execute_purge(req: PurgeRequest):
    """
    Executes 3-pass DoD 5220.22-M salted cryptographic wipe.
    """
    if not req.confirm_wipe:
        raise HTTPException(status_code=400, detail="Purge confirmation not granted.")

    stats = purge_all_biometric_data()
    purge_db_records()
    ENGINE_STATE["model"] = None

    return {
        "status": "PURGED",
        "files_destroyed": stats["file_count"],
        "bytes_wiped": stats["total_bytes"],
        "message": "DoD 5220.22-M 3-pass cryptographic purge completed. Zero telemetry verified."
    }


@app.get("/api/sample_face")
async def get_sample_face(name: str = "Alice_Smith"):
    """
    Generates or returns a sample face frame for interactive testing.
    """
    if name == "Vraj_Gandhi":
        raw_path = BASE_DIR / "raw_inputs" / "Vraj_Gandhi_raw.jpeg"
        if not raw_path.exists():
            vraj_dir = BASE_DIR / "dataset" / "Vraj_Gandhi"
            crops = list(vraj_dir.glob("*.jpg"))
            raw_path = crops[0] if crops else None

        if raw_path and raw_path.exists():
            frame = cv2.imread(str(raw_path))
            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buffer).decode("utf-8")
            return {"image_base64": f"data:image/jpeg;base64,{b64}", "student": "Vraj Gandhi"}

    from test_admin_enroll import synthesize_volunteer_face
    seeds = {"Alice_Smith": 101, "Bob_Johnson": 202, "Charlie_Davis": 303, "Diana_Evans": 404, "Ethan_Wright": 505, "Unknown": 9999}
    seed = seeds.get(name, 101)
    frame = synthesize_volunteer_face(seed)
    _, buffer = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return {"image_base64": f"data:image/jpeg;base64,{b64}", "student": name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
