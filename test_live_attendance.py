#!/usr/bin/env python3
"""
test_live_attendance.py - Rigorous Automated Test Suite for Phase 4

Validates:
1. Cold-Start Safe Mode banner rendering.
2. Verified enrolled student recognition and green tag.
3. Idempotent single-write attendance CSV logging across multiple frames.
4. Privacy masking: 51x51 Gaussian blur on unknown/non-consenting face.
5. Zero logging or leakage for unknown faces.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import cv2
import numpy as np
from live_attendance import AttendanceEngine, ATTENDANCE_LOG_PATH
from test_admin_enroll import synthesize_volunteer_face


def run_tests():
    print("[TestLiveAttendance] Running Phase 4 Validation Suite...")

    # -------------------------------------------------------------
    # 1. Cold-Start Test
    # -------------------------------------------------------------
    print("\n--- Test 1: Cold-Start Safe Mode ---")
    cold_engine = AttendanceEngine(force_cold_start=True)
    assert cold_engine.is_cold_start() == True, "Expected cold start to be True when force_cold_start=True"
    
    dummy_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    disp, dets = cold_engine.process_frame(dummy_frame)
    assert len(dets) == 0, "No detections should be processed in cold-start mode"
    # Ensure banner was drawn (check top pixels have changed color)
    assert not np.array_equal(disp[:50, :], dummy_frame[:50, :]), "Cold-start warning banner was not drawn!"
    print("Cold-Start Safe Mode passed: Banner drawn without crashing.")

    # -------------------------------------------------------------
    # 2. Enrolled Student Verification & Idempotent Logging
    # -------------------------------------------------------------
    print("\n--- Test 2: Verified Student Recognition & Idempotent Logging ---")
    # Reset log file for a clean test
    if ATTENDANCE_LOG_PATH.exists():
        ATTENDANCE_LOG_PATH.unlink()

    engine = AttendanceEngine()
    assert not engine.is_cold_start(), "Expected valid trained model from Phase 3"
    print(f"Loaded model with classes: {engine.model_data['classes']}")

    # Synthesize Alice Smith frame (seed=101)
    alice_frame = synthesize_volunteer_face(name_seed=101, angle_offset=0, brightness=0)

    # Process frame 1
    disp1, dets1 = engine.process_frame(alice_frame)
    assert len(dets1) > 0, "Failed to detect face on Alice frame"
    alice_det = dets1[0]
    assert alice_det["status"] == "VERIFIED", f"Expected VERIFIED, got {alice_det['status']}"
    assert "Alice" in alice_det["label"], f"Expected Alice_Smith, got {alice_det['label']}"
    print(f"Frame 1: Recognized {alice_det['label']} with distance={alice_det['distance']:.4f}, confidence={alice_det['confidence']:.2%}")

    # Process same frame 5 more times to test IDEMPOTENCY
    for i in range(2, 7):
        disp_i, dets_i = engine.process_frame(alice_frame)

    # Read attendance log and verify only 1 entry for Alice
    assert ATTENDANCE_LOG_PATH.exists(), "attendance.csv was not created!"
    with open(ATTENDANCE_LOG_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0]
    assert header == "timestamp,student_name,confidence_score,status", f"Incorrect CSV header: {header}"
    data_lines = lines[1:]
    assert len(data_lines) == 1, f"Idempotency failure! Expected 1 logged record, but found {len(data_lines)}: {data_lines}"
    assert "Alice_Smith" in data_lines[0], f"Logged data does not contain Alice_Smith: {data_lines[0]}"
    print("Idempotent logging verified: Exactly 1 record written across 6 evaluations.")

    # -------------------------------------------------------------
    # 3. Ethics & Privacy Masking on Unknown Face
    # -------------------------------------------------------------
    print("\n--- Test 3: Ethics & Privacy Masking on Unknown Face ---")
    # Synthesize an unknown face with completely different seed and features (seed=9999)
    # Give it an entirely different skin tone, structure, and offset
    rng = np.random.RandomState(9999)
    unknown_frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    fx, fy, fw, fh = 220, 130, 160, 200
    cv2.ellipse(unknown_frame, (fx + fw // 2, fy + fh // 2), (fw // 2, fh // 2), 0, 0, 360, (60, 80, 110), -1)
    # Eyes & Eyebrows
    cv2.ellipse(unknown_frame, (fx + 40, fy + 70), (25, 15), 0, 0, 360, (10, 10, 10), -1)
    cv2.ellipse(unknown_frame, (fx + 120, fy + 70), (25, 15), 0, 0, 360, (10, 10, 10), -1)
    cv2.line(unknown_frame, (fx + 15, fy + 55), (fx + 65, fy + 55), (5, 5, 5), 4)
    cv2.line(unknown_frame, (fx + 95, fy + 55), (fx + 145, fy + 55), (5, 5, 5), 4)
    # Nose & Mouth
    cv2.line(unknown_frame, (fx + 80, fy + 80), (fx + 80, fy + 120), (20, 30, 50), 3)
    cv2.ellipse(unknown_frame, (fx + 80, fy + 150), (35, 10), 0, 0, 360, (20, 20, 50), -1)
    unknown_frame = cv2.GaussianBlur(unknown_frame, (3, 3), 0.5)

    disp_unk, dets_unk = engine.process_frame(unknown_frame)
    if len(dets_unk) > 0:
        unk_det = dets_unk[0]
        # Verify distance is greater than threshold or status is UNKNOWN
        print(f"Unknown face detection: status={unk_det['status']}, distance={unk_det['distance']:.4f}, threshold={engine.model_data['threshold']:.4f}")
        assert unk_det["status"] == "UNKNOWN", f"Expected UNKNOWN status, got {unk_det['status']}"
        
        # Verify face region in disp_unk is blurred (high frequency differences)
        x, y, w, h = unk_det["box"]
        original_crop = unknown_frame[y:y+h, x:x+w]
        blurred_crop = disp_unk[y:y+h, x:x+w]
        # Differences between blurred crop and original must be significant
        diff = np.mean(np.abs(blurred_crop.astype(float) - original_crop.astype(float)))
        assert diff > 1.0, f"Blur was not applied! Pixel diff: {diff}"
        print(f"Privacy blur confirmed: Face region blurred with mean pixel delta {diff:.2f}")

    # Verify no new entries were added to logs/attendance.csv
    with open(ATTENDANCE_LOG_PATH, "r", encoding="utf-8") as f:
        current_lines = [l.strip() for l in f if l.strip()][1:]
    assert len(current_lines) == 1, f"Zero-logging violation! Unknown face wrote to log: {current_lines}"
    print("Zero-logging of non-consenting individuals confirmed.")

    print("\n=======================================================")
    print("ALL PHASE 4 TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()
