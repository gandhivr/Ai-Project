#!/usr/bin/env python3
"""
src/database.py - SQLite Database Management Layer for Smart Classroom Attendance

Provides persistence for:
- Enrolled volunteer students
- Real-time attendance verification logs
- System audit lifecycle events (retraining, wiping, tuning)
- Two-way synchronization with logs/attendance.csv
"""

import os
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "attendance.db"
CSV_LOG_PATH = BASE_DIR / "logs" / "attendance.csv"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables and indexes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                clean_name TEXT UNIQUE NOT NULL,
                roll_number TEXT NOT NULL,
                consent_signed BOOLEAN NOT NULL DEFAULT 1,
                enrolled_at TEXT NOT NULL,
                crop_count INTEGER DEFAULT 0
            )
        """)

        # Attendance Records Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                roll_number TEXT,
                timestamp TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                status TEXT NOT NULL,
                session_date TEXT NOT NULL
            )
        """)

        # System Audit Log Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        conn.commit()

    # Seed existing students from dataset/ directory if available
    sync_existing_dataset_students()
    # Sync with attendance.csv
    sync_from_csv()


def sync_existing_dataset_students():
    """Scans dataset/ directory and registers existing students into SQLite."""
    dataset_dir = BASE_DIR / "dataset"
    if not dataset_dir.exists():
        return

    default_rolls = {
        "Alice_Smith": "CS-2026-001",
        "Bob_Johnson": "CS-2026-002",
        "Charlie_Davis": "CS-2026-003",
        "Diana_Evans": "CS-2026-004",
        "Ethan_Wright": "CS-2026-005",
    }

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for d in dataset_dir.iterdir():
            if d.is_dir():
                clean_name = d.name
                disp_name = clean_name.replace("_", " ")
                roll = default_rolls.get(clean_name, f"ROLL-{clean_name[:4].upper()}")
                crops = len(list(d.glob("img_*.jpg")))

                cursor.execute("""
                    INSERT INTO students (name, clean_name, roll_number, consent_signed, enrolled_at, crop_count)
                    VALUES (?, ?, ?, 1, ?, ?)
                    ON CONFLICT(clean_name) DO UPDATE SET crop_count = excluded.crop_count
                """, (disp_name, clean_name, roll, datetime.datetime.now().isoformat(), crops))
        conn.commit()


def sync_from_csv():
    """Synchronizes records from logs/attendance.csv into SQLite."""
    if not CSV_LOG_PATH.exists():
        return

    with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) <= 1:
        return

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 4:
                ts, name, conf, status = parts[0], parts[1], float(parts[2]), parts[3]
                date_part = ts.split("T")[0] if "T" in ts else ts.split(" ")[0]

                # Check if already exists
                cursor.execute("""
                    SELECT id FROM attendance_logs WHERE student_name = ? AND timestamp = ?
                """, (name, ts))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO attendance_logs (student_name, timestamp, confidence_score, status, session_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, ts, conf, status, date_part))
        conn.commit()


def log_attendance_db(student_name: str, confidence: float, roll: Optional[str] = None) -> bool:
    """Logs an attendance record if not already recorded today."""
    today = datetime.date.today().isoformat()
    now_iso = datetime.datetime.now().isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM attendance_logs 
            WHERE student_name = ? AND session_date = ?
        """, (student_name, today))

        if cursor.fetchone():
            return False  # Already marked today

        cursor.execute("""
            INSERT INTO attendance_logs (student_name, roll_number, timestamp, confidence_score, status, session_date)
            VALUES (?, ?, ?, ?, 'PRESENT', ?)
        """, (student_name, roll, now_iso, confidence, today))
        conn.commit()

    return True


def get_all_students() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_attendance_records(limit: int = 100) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM attendance_logs 
            ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_dashboard_stats() -> Dict[str, Any]:
    today = datetime.date.today().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT student_name) FROM attendance_logs WHERE session_date = ?", (today,))
        present_today = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence_score) FROM attendance_logs")
        avg_conf_row = cursor.fetchone()[0]
        avg_conf = float(avg_conf_row) if avg_conf_row is not None else 0.0

    return {
        "total_enrolled": total_students,
        "present_today": present_today,
        "avg_confidence": round(avg_conf * 100, 1),
        "today_date": today
    }


def purge_db_records():
    """Wipes all student records and attendance logs upon DoD purge execution."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students")
        cursor.execute("DELETE FROM attendance_logs")
        cursor.execute("DELETE FROM audit_events")
        conn.commit()


if __name__ == "__main__":
    init_db()
    print("[Database] Initialized and synchronized successfully.")
    print("[Database] Current Stats:", get_dashboard_stats())
