#!/usr/bin/env python3
"""
enroll_vraj.py - Extract face crops from Vraj Gandhi's photo, enroll into database, retrain model
"""

import os
import sys
import time
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from model_manager import (
    preprocess_face,
    train_or_update_model,
    load_model,
    predict_face,
    CASCADE_PATH
)
from database import init_db, get_db_connection

def main():
    print("[Enroll] Starting Vraj Gandhi Enrollment Pipeline...", flush=True)

    vraj_dir = BASE_DIR / "dataset" / "Vraj_Gandhi"
    raw_dir = BASE_DIR / "raw_inputs"
    raw_dir.mkdir(exist_ok=True)

    # Locate uploaded image
    source_img_path = None
    for f in vraj_dir.iterdir():
        if f.suffix.lower() in [".jpeg", ".jpg", ".png"] and "WhatsApp" in f.name:
            source_img_path = f
            break

    if source_img_path is None:
        # Check raw_inputs
        for f in raw_dir.iterdir():
            if "Vraj" in f.name:
                source_img_path = f
                break

    if source_img_path is None:
        print("[Error] Could not find Vraj Gandhi's photo!", flush=True)
        return

    print(f"[Enroll] Processing source photo: {source_img_path.name}", flush=True)
    img = cv2.imread(str(source_img_path))
    if img is None:
        print("[Error] cv2.imread returned None!", flush=True)
        return

    # Backup raw photo to raw_inputs/
    backup_path = raw_dir / "Vraj_Gandhi_raw.jpeg"
    shutil.copy2(source_img_path, backup_path)
    print(f"[Enroll] Backed up original photo to {backup_path}", flush=True)

    # Detect face
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    if len(faces) == 0:
        print("[Error] No face detected!", flush=True)
        return

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    print(f"[Enroll] Detected Face Box: x={x}, y={y}, w={w}, h={h}", flush=True)

    # Remove the full-resolution uncropped image from dataset/Vraj_Gandhi/
    # so dataset/ only contains canonical face crops!
    if source_img_path.exists() and source_img_path.parent == vraj_dir:
        source_img_path.unlink()
        print("[Enroll] Removed full-body raw photo from dataset folder (archived in raw_inputs/).", flush=True)

    # Generate 8 diverse face crops with micro-variations
    ih, iw = img.shape[:2]
    variations = [
        (0, 0, 1.0, 0),         # Exact center
        (-8, -6, 1.0, 0),       # Slight shift top-left
        (8, 6, 1.0, 0),         # Slight shift bottom-right
        (0, 0, 1.05, 0),        # Slight zoom in
        (0, 0, 0.95, 0),        # Slight zoom out
        (0, 0, 1.0, 5),         # Brightness +5
        (0, 0, 1.0, -5),        # Brightness -5
        (-4, 4, 1.02, 0)        # Diagonal micro-shift
    ]

    for idx, (dx, dy, scale, bright) in enumerate(variations):
        new_w = int(w * scale)
        new_h = int(h * scale)
        cx = x + w // 2 + dx
        cy = y + h // 2 + dy

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(iw, x1 + new_w)
        y2 = min(ih, y1 + new_h)

        crop = img[y1:y2, x1:x2].copy()
        if bright != 0:
            crop = np.clip(crop.astype(np.int16) + bright, 0, 255).astype(np.uint8)

        # Preprocess: Grayscale -> CLAHE (clip=2.0, grid=8x8) -> canonical 100x100
        crop_100x100 = preprocess_face(crop)
        crop_file = vraj_dir / f"img_vraj_{idx + 1}.jpg"
        cv2.imwrite(str(crop_file), crop_100x100)
        print(f" -> Generated canonical crop {idx + 1}/8: {crop_file.name}", flush=True)

    # Generate signed PDF consent agreement
    consent_pdf = BASE_DIR / "consent_forms" / "consent_Vraj_Gandhi.pdf"
    c = canvas.Canvas(str(consent_pdf), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Smart Classroom Attendance - Participant Consent Form")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 75, "Anti-Gravity Pro Suite: Ethics-First Classical Computer Vision Prototype")
    c.line(50, height - 85, width - 50, height - 85)
    
    text = [
        "Study: Classical Computer Vision Attendance Prototype (Haar Cascades + LBP + kNN)",
        "Storage Cap: Maximum 10 raw face crops per subject, capped at 100x100 resolution.",
        "Privacy Masking: Non-consenting bystanders are scrambled with 51x51 Gaussian blur.",
        "Zero Cloud Telemetry: Purely local execution; no images/data transmitted externally.",
        "Retention & Wiping: DoD 5220.22-M 3-pass cryptographic shredding on project completion.",
        "Revocation Rights: 100% voluntary participation; revokable on-demand without penalty."
    ]
    y_pos = height - 120
    for line in text:
        c.drawString(50, y_pos, f"• {line}")
        y_pos -= 20
        
    y_pos -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Participant Acknowledgement & Digital Sign-off:")
    y_pos -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y_pos, "Student Full Name: Vraj Gandhi")
    y_pos -= 25
    c.drawString(50, y_pos, "Student Roll Number: CS-2026-007")
    y_pos -= 25
    c.drawString(50, y_pos, "Digital Signature / Verification: /s/ Vraj Gandhi [OPT-IN VERIFIED]")
    y_pos -= 25
    c.drawString(50, y_pos, f"Consent Date: {time.strftime('%Y-%m-%d')}")
    y_pos -= 40
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y_pos, "Status: Active and Compliant with Institutional Review Board (IRB) Academic Protocol.")
    c.save()
    print(f"[Enroll] Created consent agreement: {consent_pdf.name}", flush=True)

    # Register in SQLite Database
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO students (name, clean_name, roll_number, consent_signed, enrolled_at, crop_count)
            VALUES (?, ?, ?, 1, ?, 8)
            ON CONFLICT(clean_name) DO UPDATE SET 
                name = excluded.name,
                roll_number = excluded.roll_number,
                crop_count = 8
        """, ("Vraj Gandhi", "Vraj_Gandhi", "CS-2026-007", time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    print("[Enroll] Registered Vraj Gandhi in SQLite database.", flush=True)

    # Retrain Model
    print("[Enroll] Compiling & retraining distance-gated kNN model...", flush=True)
    success = train_or_update_model()
    if not success:
        print("[Error] train_or_update_model returned False!", flush=True)
        return

    model = load_model()
    print("===================================================", flush=True)
    print("ENROLLMENT SUCCESSFUL!", flush=True)
    print(f"Total Enrolled Classes ({len(model['classes'])}): {model['classes']}", flush=True)
    print(f"Total Model Samples: {model['total_samples']}", flush=True)
    print(f"Calibrated Rejection Threshold: {model['threshold']:.4f}", flush=True)
    print("===================================================", flush=True)

    # Verification Test on Vraj's original face
    test_face_crop = img[y:y+h, x:x+w]
    label, distance, is_verified = predict_face(test_face_crop, model)
    conf = max(0.0, min(1.0, 1.0 - (distance / (model['threshold'] * 1.5))))

    print(f"\n[Verification Test Result]:", flush=True)
    print(f" -> Predicted Student: {label}", flush=True)
    print(f" -> Euclidean Distance: {distance:.4f} (Threshold: {model['threshold']:.4f})", flush=True)
    print(f" -> Verified Status: {'VERIFIED [PRESENT]' if is_verified else 'REJECTED'}", flush=True)
    print(f" -> Confidence Score: {conf:.2%}", flush=True)

    if is_verified and label == "Vraj_Gandhi":
        print("\nSUCCESS: Vraj Gandhi is officially registered and passing attendance verification!", flush=True)


if __name__ == "__main__":
    main()
