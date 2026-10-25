#!/usr/bin/env python3
"""
src/admin_enroll.py - Dynamic Administrator Enrollment Tool

Interactive webcam tool for enrolling volunteer students:
- Prompts for Student Name and Roll Number
- Real-time Haar face tracking (scaleFactor=1.15, minNeighbors=5)
- UI Overlay with Green tracking box, metadata, and "Captured: X/10" counter
- 'c' to capture face crop, preprocess via model_manager, and save to dataset/
- 'q' to finalize enrollment (enforcing minimum 5 crops)
- Automatically triggers background model retraining upon completion
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Import preprocessing and model training from model_manager
from model_manager import preprocess_face, train_or_update_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Tuned Haar Cascade parameters
SCALE_FACTOR = 1.15
MIN_NEIGHBORS = 5
MIN_CAPTURES = 5
MAX_CAPTURES = 10


def get_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from: {CASCADE_PATH}")
    return detector


def draw_enrollment_ui(
    frame: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]],
    student_name: str,
    roll_number: str,
    capture_count: int,
    status_msg: str = "",
    status_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Renders the HUD overlay:
    - Green bounding box with student tag around detected face
    - Header banner with student name & roll number
    - Captured counter "Captured: X/10 (Press 'c' to capture, 'q' to finish)"
    - Status message notification
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw detected face box
    if face_box is not None:
        x, y, fw, fh = face_box
        # Green tracking box
        cv2.rectangle(display, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        # Bounding box tag
        tag = f"{student_name} ({roll_number})"
        cv2.putText(display, tag, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Top banner background (semi-transparent dark bar)
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    # Top Banner Text
    cv2.putText(
        display,
        f"ADMIN ENROLLMENT: {student_name} | ROLL: {roll_number}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    counter_text = f"Captured: {capture_count}/{MAX_CAPTURES}  (Press 'c' to capture, 'q' to finish)"
    counter_color = (0, 255, 0) if capture_count >= MIN_CAPTURES else (0, 200, 255)
    cv2.putText(display, counter_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, counter_color, 2)

    # Bottom status message if active
    if status_msg:
        cv2.putText(display, status_msg, (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2)

    return display


def run_enrollment(
    student_name: str,
    roll_number: str,
    camera_source: int = 0,
    video_source: Optional[str] = None,
    mock_images: Optional[list] = None,
    auto_capture: bool = False,
    headless: bool = False
) -> int:
    """
    Executes live enrollment session.
    """
    clean_name = student_name.strip().replace(" ", "_")
    student_dir = DATASET_DIR / clean_name
    student_dir.mkdir(parents=True, exist_ok=True)

    detector = get_face_detector()
    capture_count = len(list(student_dir.glob("img_*.jpg")))

    print(f"\n[AdminEnroll] Initiating enrollment for: {student_name} ({roll_number})")
    print(f"[AdminEnroll] Existing images for student: {capture_count}")
    print(f"[AdminEnroll] Minimum required: {MIN_CAPTURES}, Maximum allowed: {MAX_CAPTURES}")

    # Handle synthetic/mock test injection for headless or automated testing
    if mock_images is not None:
        print("[AdminEnroll] Ingesting mock image stream for automated testing...")
        for img in mock_images:
            if capture_count >= MAX_CAPTURES:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            faces = detector.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                crop = img[y:y+h, x:x+w]
                processed = preprocess_face(crop)
                ts = int(time.time() * 1000)
                out_file = student_dir / f"img_{ts}_{capture_count + 1}.jpg"
                cv2.imwrite(str(out_file), processed)
                capture_count += 1
                time.sleep(0.01)

        print(f"[AdminEnroll] Mock capture finished: {capture_count}/{MAX_CAPTURES} images.")
        if capture_count >= MIN_CAPTURES:
            print("[AdminEnroll] Triggering immediate background model retraining...")
            success = train_or_update_model()
            print(f"[AdminEnroll] Retraining status: {'SUCCESS' if success else 'FAILED'}")
        return capture_count

    # Video or webcam source
    src = video_source if video_source else camera_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[AdminEnroll] Error: Unable to open camera source: {src}", file=sys.stderr)
        return 0

    status_message = ""
    status_expiry = 0.0
    status_color = (0, 255, 0)
    window_name = "Smart Classroom Attendance - Admin Enrollment"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[AdminEnroll] Video stream ended or unavailable.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS)

            best_face = None
            if len(faces) > 0:
                # Select the largest detected face
                best_face = max(faces, key=lambda b: b[2] * b[3])

            current_time = time.time()
            if current_time > status_expiry:
                status_message = ""

            display_frame = draw_enrollment_ui(
                frame,
                best_face,
                student_name,
                roll_number,
                capture_count,
                status_message,
                status_color
            )

            if not headless:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            else:
                key = 0xFF
                if auto_capture and best_face is not None and capture_count < MAX_CAPTURES:
                    key = ord('c')
                elif auto_capture and capture_count >= MIN_CAPTURES:
                    key = ord('q')

            # Handle capture 'c'
            if key == ord('c') or key == ord('C'):
                if best_face is None:
                    status_message = "WARNING: No face detected. Position yourself in frame."
                    status_color = (0, 0, 255)
                    status_expiry = current_time + 1.5
                elif capture_count >= MAX_CAPTURES:
                    status_message = f"CAP REACHED: Maximum {MAX_CAPTURES} images collected."
                    status_color = (0, 200, 255)
                    status_expiry = current_time + 1.5
                else:
                    x, y, fw, fh = best_face
                    crop = frame[y:y+fh, x:x+fw]
                    try:
                        preprocessed = preprocess_face(crop)
                        ts = int(time.time() * 1000)
                        crop_filename = student_dir / f"img_{ts}.jpg"
                        cv2.imwrite(str(crop_filename), preprocessed)
                        capture_count += 1
                        status_message = f"SUCCESS: Captured crop {capture_count}/{MAX_CAPTURES}!"
                        status_color = (0, 255, 0)
                        status_expiry = current_time + 1.2
                        print(f"[AdminEnroll] Saved crop {capture_count}: {crop_filename.name}")
                    except Exception as err:
                        status_message = f"ERROR: Preprocessing failed: {err}"
                        status_color = (0, 0, 255)
                        status_expiry = current_time + 2.0

            # Handle quit / finish 'q'
            elif key == ord('q') or key == ord('Q'):
                if capture_count < MIN_CAPTURES:
                    status_message = f"CANNOT FINISH: Need at least {MIN_CAPTURES} images! (Current: {capture_count})"
                    status_color = (0, 0, 255)
                    status_expiry = current_time + 2.5
                    print(f"[AdminEnroll] Rejecting exit: minimum {MIN_CAPTURES} required.")
                else:
                    print(f"\n[AdminEnroll] Enrollment finished with {capture_count} images.")
                    print("[AdminEnroll] Triggering immediate background model retraining...")
                    train_success = train_or_update_model()
                    if train_success:
                        print("[AdminEnroll] System retrained and ready for live door inference!")
                    else:
                        print("[AdminEnroll] Model retraining returned False. Check dataset integrity.")
                    break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    return capture_count


def main():
    parser = argparse.ArgumentParser(description="Admin Enrollment Tool for Smart Classroom Attendance.")
    parser.add_argument("--name", type=str, help="Volunteer student full name.")
    parser.add_argument("--roll", type=str, help="Volunteer student roll/matriculation number.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--video", type=str, help="Optional video file path instead of live webcam.")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows.")
    parser.add_argument("--auto-capture", action="store_true", help="Automatically capture detected faces.")

    args = parser.parse_args()

    name = args.name
    roll = args.roll

    if not name:
        name = input("Enter Student Full Name: ").strip()
    if not roll:
        roll = input("Enter Student Roll Number: ").strip()

    if not name or not roll:
        print("Error: Student Name and Roll Number are required.", file=sys.stderr)
        sys.exit(1)

    run_enrollment(
        student_name=name,
        roll_number=roll,
        camera_source=args.camera,
        video_source=args.video,
        auto_capture=args.auto_capture,
        headless=args.headless
    )


if __name__ == "__main__":
    main()
