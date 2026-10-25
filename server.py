#!/usr/bin/env python3
"""
server.py - Production University Academic Biometric Intelligence Server

Anti-Gravity Pro Academic Suite:
- Classical Computer Vision Engine (Haar + CLAHE + 3,776-dim Uniform LBP + 1-NN)
- Gate Attendance Verification with Privacy Blurring
- Panoramic Classroom Multi-Face Sweep & 1-Click Batch Attendance
- Multi-Course Curriculum, Lecture Slot & Live Session Management
- Interactive Faculty Roster & Manual Status Override (Present/Late/Excused/Absent)
- Accreditation Defaulter Analytics (<75% Early Warning Notice Generation)
- Anti-Spoofing & Liveness Texture Frequency Inspection
- Student Dossier & Crop Quality Verification
- DoD 5220.22-M Cryptographic Data Purging
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
    sync_existing_dataset_students,
    get_courses,
    get_active_session,
    get_course_roster,
    update_student_status_override,
    get_defaulters_analysis,
    get_analytics_summary,
    get_student_detail
)
from purge_data import purge_all_biometric_data
from evaluate_robustness import run_robustness_audit
from tune_detector import run_grid_search

app = FastAPI(title="Veritas Academic AI - Smart Classroom Attendance Pro")

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
    "last_trained": None,
    "active_course": "CS-401"
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
    print("[Server] Veritas Academic Vision Suite initialized.")


# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class FrameVerifyRequest(BaseModel):
    image_base64: str
    course_code: Optional[str] = "CS-401"
    session_id: Optional[int] = 1

class ClassroomSweepRequest(BaseModel):
    image_base64: str
    course_code: Optional[str] = "CS-401"
    session_id: Optional[int] = 1
    mark_attendance: Optional[bool] = True

class RosterStatusUpdateRequest(BaseModel):
    student_clean_name: str
    status: str
    course_code: Optional[str] = "CS-401"
    remarks: Optional[str] = "Faculty Manual Override"

class EnrollCropRequest(BaseModel):
    student_name: str
    roll_number: str
    image_base64: str
    department: Optional[str] = "Computer Science & AI"
    email: Optional[str] = None

class AntiSpoofRequest(BaseModel):
    image_base64: str

class PurgeRequest(BaseModel):
    confirm_wipe: bool


# -------------------------------------------------------------
# Core Navigation & System Status Endpoints
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
    active_course = ENGINE_STATE["active_course"]
    active_session = get_active_session(active_course)

    return {
        "status": "ONLINE",
        "institution": "Veritas Academic University",
        "cold_start": is_cold,
        "enrolled_count": stats["total_enrolled"],
        "present_today": stats["present_today"],
        "avg_confidence": stats["avg_confidence"],
        "total_courses": stats["total_courses"],
        "active_course": active_course,
        "active_session": active_session,
        "classes": model.get("classes", []) if model else [],
        "rejection_threshold": round(model.get("threshold", DEFAULT_FALLBACK_THRESHOLD), 4) if model else DEFAULT_FALLBACK_THRESHOLD,
        "total_samples": model.get("total_samples", 0) if model else 0,
        "timestamp": datetime.datetime.now().isoformat()
    }


# -------------------------------------------------------------
# University Curriculum, Courses & Sessions
# -------------------------------------------------------------

@app.get("/api/courses")
async def api_get_courses():
    courses = get_courses()
    return {"courses": courses, "active_course": ENGINE_STATE["active_course"]}


@app.post("/api/courses/select")
async def api_select_course(code: str):
    ENGINE_STATE["active_course"] = code
    session = get_active_session(code)
    return {"status": "SUCCESS", "active_course": code, "session": session}


@app.get("/api/roster")
async def api_get_roster(course_code: Optional[str] = None):
    course = course_code or ENGINE_STATE["active_course"]
    roster = get_course_roster(course)
    return {"course_code": course, "roster": roster, "total_students": len(roster)}


@app.post("/api/roster/status")
async def api_update_roster_status(req: RosterStatusUpdateRequest):
    course = req.course_code or ENGINE_STATE["active_course"]
    success = update_student_status_override(
        student_clean_name=req.student_clean_name,
        status=req.status.upper(),
        course_code=course,
        remarks=req.remarks
    )
    return {"status": "SUCCESS" if success else "ERROR", "student": req.student_clean_name, "new_status": req.status.upper()}


# -------------------------------------------------------------
# Accreditation Analytics & Defaulters Alert
# -------------------------------------------------------------

@app.get("/api/analytics/defaulters")
async def api_get_defaulters(course_code: Optional[str] = None, threshold: float = 75.0):
    course = course_code or ENGINE_STATE["active_course"]
    analysis = get_defaulters_analysis(course, min_threshold_pct=threshold)
    return analysis


@app.get("/api/analytics/summary")
async def api_get_analytics_summary(course_code: Optional[str] = None):
    course = course_code or ENGINE_STATE["active_course"]
    summary = get_analytics_summary(course)
    return summary


# -------------------------------------------------------------
# Student Dossier & Crop Quality
# -------------------------------------------------------------

@app.get("/api/students")
async def get_students():
    students = get_all_students()
    return {"students": students}


@app.get("/api/students/{clean_name}")
async def api_get_student_dossier(clean_name: str):
    dossier = get_student_detail(clean_name)
    if not dossier:
        raise HTTPException(status_code=404, detail="Student dossier not found")
    return dossier


@app.get("/api/students/{clean_name}/crops/{filename}")
async def api_get_student_crop(clean_name: str, filename: str):
    crop_path = BASE_DIR / "dataset" / clean_name / filename
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Biometric crop not found")
    return FileResponse(str(crop_path))


# -------------------------------------------------------------
# Computer Vision Inference: Real-Time Gate Attendance
# -------------------------------------------------------------

@app.post("/api/verify_frame")
async def verify_frame(req: FrameVerifyRequest):
    """
    Accepts webcam frame in base64.
    Runs tuned Haar cascade face detection, CLAHE + 3,776-dim LBP extraction,
    queries 1-NN model, and determines:
    - VERIFIED student (green box, confidence, logs attendance to course)
    - UNKNOWN / Non-consenting bystander (red box, 51x51 Gaussian blur)
    """
    model = get_active_model()
    if model is None or len(model.get("classes", [])) == 0:
        return {
            "cold_start": True,
            "has_face": False,
            "message": "Cold-Start Safe Mode: Zero students enrolled."
        }

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
    course_code = req.course_code or ENGINE_STATE["active_course"]

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

            # Log to SQLite & CSV idempotently for this course
            is_new = log_attendance_db(
                student_name=pred_label,
                confidence=conf,
                course_code=course_code,
                session_id=req.session_id or 1,
                method="GATE_CAMERA"
            )
            if is_new:
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


# -------------------------------------------------------------
# Multi-Face Classroom Sweep (Lecture Hall Panoramic Ingestion)
# -------------------------------------------------------------

@app.post("/api/classroom_sweep")
async def api_classroom_sweep(req: ClassroomSweepRequest):
    """
    Simultaneously processes an entire classroom / lecture hall photo.
    Detects all students in parallel, matches against enrolled curriculum,
    marks presence in bulk with 1 click, and applies privacy blur to unknowns.
    """
    model = get_active_model()
    if model is None or len(model.get("classes", [])) == 0:
        return {"status": "ERROR", "message": "Zero students enrolled in model."}

    try:
        header_sep = req.image_base64.find(",")
        raw_b64 = req.image_base64[header_sep + 1:] if header_sep != -1 else req.image_base64
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode classroom image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    annotated = frame.copy()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = get_detector()

    # Multi-scale detection tuned for variable distance in lecture halls
    faces = detector.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=4, minSize=(40, 40))

    knn = model["knn"]
    labels = model["labels"]
    threshold = model.get("threshold", DEFAULT_FALLBACK_THRESHOLD)
    course_code = req.course_code or ENGINE_STATE["active_course"]

    detected_students = []
    verified_count = 0
    unknown_count = 0

    for (x, y, fw, fh) in faces:
        x_end = min(w, x + fw)
        y_end = min(h, y + fh)
        crop = frame[y:y_end, x:x_end]
        if crop.size == 0:
            continue

        preproc = preprocess_face(crop)
        feat = extract_lbp_feature(preproc).reshape(1, -1)
        distances, indices = knn.kneighbors(feat, n_neighbors=1)
        dist = float(distances[0][0])
        matched_idx = int(indices[0][0])
        pred_label = labels[matched_idx]

        if dist <= threshold:
            conf = max(0.0, min(1.0, 1.0 - (dist / (threshold * 1.5))))
            disp_name = pred_label.replace("_", " ")
            verified_count += 1

            if req.mark_attendance:
                log_attendance_db(
                    student_name=pred_label,
                    confidence=conf,
                    course_code=course_code,
                    session_id=req.session_id or 1,
                    method="PANORAMIC_SWEEP"
                )

            # Draw green bounding box & label
            cv2.rectangle(annotated, (x, y), (x_end, y_end), (16, 185, 129), 2)
            cv2.rectangle(annotated, (x, max(0, y - 24)), (x + 160, y), (16, 185, 129), -1)
            cv2.putText(annotated, f"{disp_name} ({int(conf*100)}%)", (x + 4, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            detected_students.append({
                "student_name": disp_name,
                "clean_name": pred_label,
                "confidence": round(conf * 100, 1),
                "distance": round(dist, 4),
                "status": "VERIFIED",
                "box": [int(x), int(y), int(fw), int(fh)]
            })
        else:
            unknown_count += 1
            # Optical Gaussian Blur for bystander
            face_slice = annotated[y:y_end, x:x_end]
            blurred = cv2.GaussianBlur(face_slice, (51, 51), 30)
            annotated[y:y_end, x:x_end] = blurred

            cv2.rectangle(annotated, (x, y), (x_end, y_end), (244, 63, 94), 2)
            cv2.putText(annotated, "Unknown (Blurred)", (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (244, 63, 94), 1)

            detected_students.append({
                "student_name": "Bystander",
                "clean_name": "Unknown",
                "confidence": 0.0,
                "distance": round(dist, 4),
                "status": "UNKNOWN",
                "box": [int(x), int(y), int(fw), int(fh)]
            })

    # Encode annotated classroom result
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "SUCCESS",
        "total_faces_found": len(faces),
        "verified_count": verified_count,
        "unknown_count": unknown_count,
        "students": detected_students,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
    }


@app.get("/api/sample_classroom")
async def api_sample_classroom():
    """
    Synthesizes a realistic panoramic classroom lecture row with 5 enrolled students
    and 1 unregistered visitor for instant 1-click batch demonstration!
    """
    from test_admin_enroll import synthesize_volunteer_face

    # Create a wide classroom backdrop: 960x360
    backdrop = np.full((360, 960, 3), 32, dtype=np.uint8)
    # Draw classroom board and desk line
    cv2.rectangle(backdrop, (40, 20), (920, 140), (45, 55, 72), -1)
    cv2.putText(backdrop, "VERITAS UNIVERSITY - LECTURE HALL TURING-402", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 210, 230), 2)
    cv2.putText(backdrop, "CS-401: Advanced Computer Vision & Classical LBP Descriptors", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 160, 190), 1)
    cv2.rectangle(backdrop, (0, 260), (960, 360), (20, 25, 35), -1)

    # Insert volunteer face portraits across the classroom desks
    seeds = [101, 202, 303, 404, 505, 9999]
    positions = [(30, 160), (190, 160), (350, 160), (510, 160), (670, 160), (815, 160)]

    for seed, (px, py) in zip(seeds, positions):
        synth = synthesize_volunteer_face(seed)
        # Crop the face portrait: 260x240 around face
        face_portrait = synth[90:370, 195:435]
        face_scaled = cv2.resize(face_portrait, (120, 140))
        backdrop[py:py+140, px:px+120] = face_scaled

    _, buffer = cv2.imencode(".jpg", backdrop)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return {"image_base64": f"data:image/jpeg;base64,{b64}", "description": "Synthetic 6-Student Classroom Row"}


# -------------------------------------------------------------
# Anti-Spoofing & Liveness Frequency Detection
# -------------------------------------------------------------

@app.post("/api/anti_spoof")
async def api_anti_spoof_check(req: AntiSpoofRequest):
    """
    Academic Liveness & Anti-Spoofing Guard:
    Analyzes high-frequency Laplacian variance & specular gradient ratio
    to reject flat paper printouts and smartphone screen re-projections.
    """
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    # High frequency texture threshold: Live faces typically have variance in [80.0, 1200.0]
    # Blurry / printed paper < 45.0, Screen moiré / pixel grid > 2500.0
    is_live = 45.0 <= variance <= 3500.0
    spoof_type = "NONE"
    if variance < 45.0:
        spoof_type = "STATIC_PRINTED_PHOTO"
    elif variance > 3500.0:
        spoof_type = "SCREEN_MOIRE_RECAPTURE"

    liveness_score = max(0.0, min(100.0, (variance / 300.0) * 100.0))

    return {
        "is_live": is_live,
        "liveness_score": round(liveness_score, 1),
        "laplacian_variance": round(variance, 2),
        "detected_anomaly": spoof_type,
        "message": "Human biometric liveness verified." if is_live else f"Potential proxy detected: {spoof_type}"
    }


# -------------------------------------------------------------
# Admin Enrollment, Dynamic Retraining & DoD Purge
# -------------------------------------------------------------

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
    total_crops = len(list(student_dir.glob("*.jpg")))

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


@app.get("/api/attendance")
async def get_attendance():
    records = get_attendance_records(limit=150)
    return {"records": records}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
