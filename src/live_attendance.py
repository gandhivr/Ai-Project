#!/usr/bin/env python3
"""
src/live_attendance.py - Live Exam Door Inference & Privacy Blurring Engine

Real-time classroom / exam room door verification:
1. Cold-Start Safe Mode:
   - Banner warning: "STATUS: Zero Students Enrolled. Please run admin_enroll.py"
   - Survives zero-enrollment state without crashing.
2. Inference Pipeline:
   - Tuned Haar face detection (scaleFactor=1.15, minNeighbors=5).
   - Preprocessing + 3,776-dim Uniform LBP feature extraction.
   - Distance-gated 1-NN query.
3. Ethics & Privacy Masking (M7):
   - If distance > threshold: Instant 51x51 Gaussian blur (sigma=30).
   - Red box labeled "Unknown (Blurred)". Zero logging.
4. Idempotent Attendance Logger (M5):
   - If distance <= threshold: Green box with student name and confidence.
   - In-memory session set 'marked_today' ensures zero duplicate writes.
   - Appends atomically to logs/attendance.csv.
"""

import os
import sys
import time
import argparse
import datetime
import threading
from pathlib import Path
from typing import Optional, Set, Tuple, Dict, Any, List

import cv2
import numpy as np

# Import preprocessing, feature extraction, and model loader
from model_manager import (
    load_model,
    preprocess_face,
    extract_lbp_feature,
    CASCADE_PATH,
    DEFAULT_FALLBACK_THRESHOLD
)

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
ATTENDANCE_LOG_PATH = LOGS_DIR / "attendance.csv"

# Tuned Haar Cascade parameters
SCALE_FACTOR = 1.15
MIN_NEIGHBORS = 5

# Gaussian Blur parameters for unknown/non-consenting faces
BLUR_KSIZE = (51, 51)
BLUR_SIGMA = 30


class AttendanceEngine:
    def __init__(self, model_data: Optional[Dict[str, Any]] = None, force_cold_start: bool = False):
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {CASCADE_PATH}")

        if force_cold_start:
            self.model_data = None
        else:
            self.model_data = model_data if model_data is not None else load_model()

        self.marked_today: Set[str] = set()
        self.lock = threading.Lock()
        
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_log_file()

    def _init_log_file(self):
        """Ensures attendance.csv exists with correct CSV header."""
        with self.lock:
            if not ATTENDANCE_LOG_PATH.exists() or ATTENDANCE_LOG_PATH.stat().st_size == 0:
                with open(ATTENDANCE_LOG_PATH, "w", encoding="utf-8") as f:
                    f.write("timestamp,student_name,confidence_score,status\n")
                    f.flush()
                    os.fsync(f.fileno())

    def reload_model(self):
        """Refreshes the in-memory model from disk."""
        with self.lock:
            self.model_data = load_model()

    def is_cold_start(self) -> bool:
        if self.model_data is None:
            return True
        classes = self.model_data.get("classes", [])
        return len(classes) == 0

    def log_attendance_idempotent(self, student_name: str, confidence: float) -> bool:
        """
        Thread-safe idempotent attendance recording:
        Only logs the student once per session.
        Returns True if a new entry was written, False if already marked.
        """
        with self.lock:
            if student_name in self.marked_today:
                return False

            self.marked_today.add(student_name)
            iso_timestamp = datetime.datetime.now().isoformat()
            log_line = f"{iso_timestamp},{student_name},{confidence:.4f},PRESENT\n"

            with open(ATTENDANCE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(log_line)
                f.flush()
                os.fsync(f.fileno())

            print(f"[AttendanceLogger] Verified & Logged: {student_name} (Confidence: {confidence:.2%})")
            return True

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs face detection, inference, privacy blurring, and logging on a single frame.
        """
        display = frame.copy()
        h, w = display.shape[:2]
        detections = []

        # 1. Cold-start check
        if self.is_cold_start():
            # Draw persistent cold-start warning banner
            banner_overlay = display.copy()
            cv2.rectangle(banner_overlay, (0, 0), (w, 55), (0, 0, 180), -1)
            cv2.addWeighted(banner_overlay, 0.85, display, 0.15, 0, display)
            warning_text = "STATUS: Zero Students Enrolled. Please run admin_enroll.py"
            cv2.putText(display, warning_text, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            return display, detections

        # 2. Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS
        )

        knn = self.model_data["knn"]
        labels = self.model_data["labels"]
        threshold = self.model_data.get("threshold", DEFAULT_FALLBACK_THRESHOLD)

        for (x, y, fw, fh) in faces:
            # Boundary checks
            x_end = min(w, x + fw)
            y_end = min(h, y + fh)
            crop = frame[y:y_end, x:x_end]

            if crop.size == 0:
                continue

            # Feature extraction
            preprocessed = preprocess_face(crop)
            feat = extract_lbp_feature(preprocessed).reshape(1, -1)

            # Query kNN
            distances, indices = knn.kneighbors(feat, n_neighbors=1)
            distance = float(distances[0][0])
            matched_idx = int(indices[0][0])
            predicted_label = labels[matched_idx]

            # 3. Privacy Masking vs. Enrolled Verification
            if distance > threshold:
                # UNKNOWN / NON-CONSENTING BYSTANDER
                # Instant optical scrambling with 51x51 Gaussian Blur
                face_region = display[y:y_end, x:x_end]
                blurred = cv2.GaussianBlur(face_region, BLUR_KSIZE, BLUR_SIGMA)
                display[y:y_end, x:x_end] = blurred

                # Red bounding box & label
                cv2.rectangle(display, (x, y), (x_end, y_end), (0, 0, 255), 2)
                tag = "Unknown (Blurred)"
                cv2.putText(display, tag, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                detections.append({
                    "status": "UNKNOWN",
                    "label": "Unknown",
                    "distance": distance,
                    "box": (x, y, fw, fh)
                })
                # STRICT PRIVACY RULE: Zero logging or metadata storage for unknown faces

            else:
                # ENROLLED & CONSENTING STUDENT
                confidence = max(0.0, min(1.0, 1.0 - (distance / (threshold * 1.5))))
                disp_name = predicted_label.replace("_", " ")

                # Green bounding box & label
                cv2.rectangle(display, (x, y), (x_end, y_end), (0, 255, 0), 2)
                tag = f"{disp_name} ({confidence:.2f})"
                cv2.putText(display, tag, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                # Record attendance idempotently
                self.log_attendance_idempotent(predicted_label, confidence)

                detections.append({
                    "status": "VERIFIED",
                    "label": predicted_label,
                    "confidence": confidence,
                    "distance": distance,
                    "box": (x, y, fw, fh)
                })

        # Top Session Status Bar
        status_bar = display.copy()
        cv2.rectangle(status_bar, (0, 0), (w, 35), (30, 30, 30), -1)
        cv2.addWeighted(status_bar, 0.70, display, 0.30, 0, display)
        session_text = f"DOOR ATTENDANCE ACTIVE | Logged Today: {len(self.marked_today)} | Enrolled: {len(self.model_data.get('classes', []))}"
        cv2.putText(display, session_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        return display, detections


def run_live_attendance(
    camera_source: int = 0,
    video_source: Optional[str] = None,
    mock_frames: Optional[List[np.ndarray]] = None,
    headless: bool = False,
    max_frames: Optional[int] = None
) -> int:
    """
    Runs door attendance loop.
    Supports live webcam, video feed, or mock image arrays for testing.
    """
    engine = AttendanceEngine()

    if mock_frames is not None:
        processed_count = 0
        for frame in mock_frames:
            _, _ = engine.process_frame(frame)
            processed_count += 1
            if max_frames and processed_count >= max_frames:
                break
        return processed_count

    src = video_source if video_source else camera_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[LiveAttendance] Error: Unable to open camera source {src}", file=sys.stderr)
        return 0

    window_name = "Classroom Door Attendance - Live Inference & Privacy Shield"
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            display_frame, _ = engine.process_frame(frame)
            frame_count += 1

            if not headless:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("[LiveAttendance] 'q' pressed. Terminating session.")
                    break
            else:
                if max_frames and frame_count >= max_frames:
                    break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[LiveAttendance] Session concluded. Total students marked today: {len(engine.marked_today)}")
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Live Exam Door Attendance & Privacy Blurring System.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--video", type=str, help="Optional input video file path.")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows.")
    parser.add_argument("--frames", type=int, help="Maximum frames to process before exiting.")

    args = parser.parse_args()
    run_live_attendance(
        camera_source=args.camera,
        video_source=args.video,
        headless=args.headless,
        max_frames=args.frames
    )


if __name__ == "__main__":
    main()
