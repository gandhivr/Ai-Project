#!/usr/bin/env python3
"""
src/database.py - University Academic Intelligence & SQLite Database Management Layer

Provides enterprise persistence for:
- Enrolled students, university departments & academic majors
- Multi-course curriculum & faculty lecture session schedules
- Real-time and batch attendance verification logs (Face Scan, Panoramic Sweep, Manual Override)
- Accreditation analytics: Defaulter alert lists (<75%), attendance heatmaps, student dossiers
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
    """Initializes tables, indexes, and academic seeds."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                clean_name TEXT UNIQUE NOT NULL,
                roll_number TEXT NOT NULL,
                department TEXT NOT NULL DEFAULT 'Computer Science & AI',
                semester TEXT NOT NULL DEFAULT 'Semester VII',
                email TEXT,
                consent_signed BOOLEAN NOT NULL DEFAULT 1,
                enrolled_at TEXT NOT NULL,
                crop_count INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 98.4
            )
        """)

        # Courses / Subjects Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                instructor TEXT NOT NULL,
                room TEXT NOT NULL,
                semester TEXT NOT NULL,
                schedule TEXT NOT NULL,
                credits INTEGER DEFAULT 4,
                total_classes INTEGER DEFAULT 20
            )
        """)

        # Student Course Enrollments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS course_enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_clean_name TEXT NOT NULL,
                course_code TEXT NOT NULL,
                UNIQUE(student_clean_name, course_code)
            )
        """)

        # Lecture Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lecture_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                session_title TEXT NOT NULL,
                session_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
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
                status TEXT NOT NULL DEFAULT 'PRESENT',
                session_date TEXT NOT NULL,
                course_code TEXT DEFAULT 'CS-401',
                session_id INTEGER DEFAULT 1,
                method TEXT DEFAULT 'FACE_SCAN',
                remarks TEXT DEFAULT ''
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

        # Safe Column Migrations for existing databases
        def _ensure_col(table: str, col_name: str, col_type: str):
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = [r["name"] for r in cursor.fetchall()]
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        _ensure_col("students", "department", "TEXT DEFAULT 'Computer Science & AI'")
        _ensure_col("students", "semester", "TEXT DEFAULT 'Semester VII'")
        _ensure_col("students", "email", "TEXT")
        _ensure_col("students", "quality_score", "REAL DEFAULT 98.4")

        _ensure_col("attendance_logs", "course_code", "TEXT DEFAULT 'CS-401'")
        _ensure_col("attendance_logs", "session_id", "INTEGER DEFAULT 1")
        _ensure_col("attendance_logs", "method", "TEXT DEFAULT 'FACE_SCAN'")
        _ensure_col("attendance_logs", "remarks", "TEXT DEFAULT ''")

        conn.commit()

    # Seed Courses & Students
    seed_university_curriculum()
    sync_existing_dataset_students()
    sync_from_csv()
    seed_realistic_lecture_history()


def seed_university_curriculum():
    """Populates standard university curriculum modules."""
    courses = [
        ("CS-401", "Advanced Computer Vision & Biometrics", "Prof. Vraj Gandhi", "Auditorium Turing-402", "Semester VII", "Mon/Wed 10:00 - 11:30 AM", 4, 24),
        ("CS-302", "Distributed Cloud Systems & Microservices", "Dr. Alice Smith", "Hall Hopper-301", "Semester VI", "Tue/Thu 09:00 - 10:30 AM", 4, 22),
        ("CS-205", "Data Structures & Algorithmic Complexity", "Prof. Diana Evans", "Lab Knuth-104", "Semester IV", "Mon/Fri 14:00 - 15:30 PM", 4, 26),
        ("AI-503", "Deep Neural Representations & Ethics", "Dr. Charlie Davis", "Auditorium Lovelace-501", "Semester VIII", "Wed/Fri 11:30 - 13:00 PM", 3, 20),
    ]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for code, name, inst, room, sem, sched, cr, tot in courses:
            cursor.execute("""
                INSERT OR IGNORE INTO courses (code, name, instructor, room, semester, schedule, credits, total_classes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (code, name, inst, room, sem, sched, cr, tot))

        # Ensure an active session exists for CS-401 today
        today = datetime.date.today().isoformat()
        cursor.execute("SELECT id FROM lecture_sessions WHERE course_code = 'CS-401' AND session_date = ?", (today,))
        existing = cursor.fetchone()
        if not existing:
            cursor.execute("""
                INSERT INTO lecture_sessions (course_code, session_title, session_date, start_time, end_time, is_active, created_at)
                VALUES ('CS-401', 'Lecture #14: Uniform LBP & Real-Time Face Verification', ?, '10:00 AM', '11:30 AM', 1, ?)
            """, (today, datetime.datetime.now().isoformat()))
        conn.commit()


def sync_existing_dataset_students():
    """Scans dataset/ directory and registers existing students with full academic profile."""
    dataset_dir = BASE_DIR / "dataset"
    if not dataset_dir.exists():
        return

    student_profiles = {
        "Vraj_Gandhi": {
            "roll": "CS-2026-001",
            "name": "Vraj Gandhi",
            "dept": "Artificial Intelligence & Robotics",
            "sem": "Semester VII",
            "email": "vraj.gandhi@university.edu"
        },
        "Alice_Smith": {
            "roll": "CS-2026-002",
            "name": "Alice Smith",
            "dept": "Computer Science & Engineering",
            "sem": "Semester VII",
            "email": "alice.smith@university.edu"
        },
        "Bob_Johnson": {
            "roll": "CS-2026-003",
            "name": "Bob Johnson",
            "dept": "Computer Science & Engineering",
            "sem": "Semester VII",
            "email": "bob.johnson@university.edu"
        },
        "Charlie_Davis": {
            "roll": "CS-2026-004",
            "name": "Charlie Davis",
            "dept": "Information Systems & Cybernetics",
            "sem": "Semester VII",
            "email": "charlie.davis@university.edu"
        },
        "Diana_Evans": {
            "roll": "CS-2026-005",
            "name": "Diana Evans",
            "dept": "Data Science & Quantum Computing",
            "sem": "Semester VII",
            "email": "diana.evans@university.edu"
        },
        "Ethan_Wright": {
            "roll": "CS-2026-006",
            "name": "Ethan Wright",
            "dept": "Software Engineering",
            "sem": "Semester VII",
            "email": "ethan.wright@university.edu"
        },
    }

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for d in dataset_dir.iterdir():
            if d.is_dir():
                clean_name = d.name
                crops = len(list(d.glob("*.jpg")))
                prof = student_profiles.get(clean_name, {
                    "roll": f"CS-2026-{clean_name[:3].upper()}",
                    "name": clean_name.replace("_", " "),
                    "dept": "Computer Science & AI",
                    "sem": "Semester VII",
                    "email": f"{clean_name.lower()}@university.edu"
                })

                cursor.execute("""
                    INSERT INTO students (name, clean_name, roll_number, department, semester, email, consent_signed, enrolled_at, crop_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(clean_name) DO UPDATE SET 
                        crop_count = excluded.crop_count,
                        department = excluded.department,
                        roll_number = excluded.roll_number,
                        email = excluded.email
                """, (prof["name"], clean_name, prof["roll"], prof["dept"], prof["sem"], prof["email"], datetime.datetime.now().isoformat(), crops))

                # Auto-enroll in CS-401 and CS-302
                for code in ["CS-401", "CS-302", "CS-205"]:
                    cursor.execute("""
                        INSERT OR IGNORE INTO course_enrollments (student_clean_name, course_code)
                        VALUES (?, ?)
                    """, (clean_name, code))
        conn.commit()


def seed_realistic_lecture_history():
    """
    Seeds past lecture attendance sessions so analytics, heatmaps, and defaulter
    ratios reflect a true live university semester!
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance_logs")
        count = cursor.fetchone()[0]

        # Only seed if fewer than 15 historical records exist
        if count < 15:
            # Seed 5 historical lecture sessions over past 2 weeks
            past_sessions = [
                ("CS-401", "Lecture #09: Grayscale Histogram Equalization", "2026-10-12", "10:00 AM", "11:30 AM"),
                ("CS-401", "Lecture #10: Integral Images & Haar Cascades", "2026-10-14", "10:00 AM", "11:30 AM"),
                ("CS-401", "Lecture #11: AdaBoost Feature Selection", "2026-10-19", "10:00 AM", "11:30 AM"),
                ("CS-401", "Lecture #12: Spatial Block Partitioning in LBP", "2026-10-21", "10:00 AM", "11:30 AM"),
                ("CS-401", "Lecture #13: 1-NN Metric Rejection Boundaries", "2026-10-23", "10:00 AM", "11:30 AM"),
            ]

            session_map = {}
            for code, title, s_date, start, end in past_sessions:
                cursor.execute("""
                    INSERT OR IGNORE INTO lecture_sessions (course_code, session_title, session_date, start_time, end_time, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                """, (code, title, s_date, start, end, f"{s_date}T09:55:00"))
                session_map[title] = cursor.lastrowid

            # Attendance matrix: Vraj (100%), Alice (100%), Bob (80%), Charlie (60% - Defaulter!), Diana (100%), Ethan (40% - Defaulter!)
            matrix = {
                "Vraj_Gandhi": [True, True, True, True, True],
                "Alice_Smith": [True, True, True, True, True],
                "Bob_Johnson": [True, True, False, True, True],
                "Diana_Evans": [True, True, True, True, True],
                "Charlie_Davis": [True, False, True, False, True],  # 60%
                "Ethan_Wright": [False, True, False, False, True],  # 40%
            }

            for clean_name, presences in matrix.items():
                for idx, is_pres in enumerate(presences):
                    code, title, s_date, _, _ = past_sessions[idx]
                    status = "PRESENT" if is_pres else "ABSENT"
                    conf = 0.92 if is_pres else 0.0
                    ts = f"{s_date}T10:0{idx+1}:15"
                    cursor.execute("""
                        INSERT INTO attendance_logs (student_name, roll_number, timestamp, confidence_score, status, session_date, course_code, session_id, method, remarks)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'HISTORICAL', ?)
                    """, (clean_name, f"ROLL-{clean_name[:4].upper()}", ts, conf, status, s_date, code, idx + 1, "Verified" if is_pres else "Unexcused Absence"))

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

                cursor.execute("""
                    SELECT id FROM attendance_logs WHERE student_name = ? AND timestamp = ?
                """, (name, ts))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO attendance_logs (student_name, timestamp, confidence_score, status, session_date, course_code, method)
                        VALUES (?, ?, ?, ?, ?, 'CS-401', 'CSV_SYNC')
                    """, (name, ts, conf, status, date_part))
        conn.commit()


def log_attendance_db(
    student_name: str, 
    confidence: float, 
    roll: Optional[str] = None,
    course_code: str = "CS-401",
    session_id: Optional[int] = None,
    method: str = "FACE_SCAN",
    remarks: str = ""
) -> bool:
    """Logs an attendance record idempotently per course session."""
    today = datetime.date.today().isoformat()
    now_iso = datetime.datetime.now().isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # If roll is not passed, fetch from students table
        if not roll:
            cursor.execute("SELECT roll_number FROM students WHERE clean_name = ?", (student_name,))
            r = cursor.fetchone()
            if r:
                roll = r[0]

        # Check idempotency: 1 entry per student per course per date
        cursor.execute("""
            SELECT id FROM attendance_logs 
            WHERE student_name = ? AND session_date = ? AND course_code = ?
        """, (student_name, today, course_code))

        if cursor.fetchone():
            return False  # Already marked today for this course

        cursor.execute("""
            INSERT INTO attendance_logs (student_name, roll_number, timestamp, confidence_score, status, session_date, course_code, session_id, method, remarks)
            VALUES (?, ?, ?, ?, 'PRESENT', ?, ?, ?, ?, ?)
        """, (student_name, roll, now_iso, confidence, today, course_code, session_id or 1, method, remarks))
        conn.commit()

    return True


def get_all_students() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_courses() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM courses ORDER BY code ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_active_session(course_code: str = "CS-401") -> Optional[Dict[str, Any]]:
    today = datetime.date.today().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM lecture_sessions 
            WHERE course_code = ? AND session_date = ?
            ORDER BY id DESC LIMIT 1
        """, (course_code, today))
        row = cursor.fetchone()
        if row:
            return dict(row)
        
        # Fallback to latest session for course
        cursor.execute("""
            SELECT * FROM lecture_sessions 
            WHERE course_code = ?
            ORDER BY id DESC LIMIT 1
        """, (course_code,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_course_roster(course_code: str = "CS-401") -> List[Dict[str, Any]]:
    """Returns all enrolled students for a course along with today's status."""
    today = datetime.date.today().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.id, s.name, s.clean_name, s.roll_number, s.department, s.semester, s.email, s.crop_count, s.quality_score,
                COALESCE(a.status, 'NOT_MARKED') as today_status,
                COALESCE(a.confidence_score, 0.0) as today_confidence,
                a.timestamp as marked_at,
                a.method as verify_method,
                a.remarks as remarks
            FROM students s
            JOIN course_enrollments ce ON s.clean_name = ce.student_clean_name
            LEFT JOIN attendance_logs a ON (s.clean_name = a.student_name AND a.course_code = ? AND a.session_date = ?)
            WHERE ce.course_code = ?
            ORDER BY s.name ASC
        """, (course_code, today, course_code))
        return [dict(row) for row in cursor.fetchall()]


def update_student_status_override(
    student_clean_name: str,
    status: str,
    course_code: str = "CS-401",
    remarks: str = "Faculty Manual Override"
) -> bool:
    """Allows professor or TA to manually adjust a student's attendance status."""
    today = datetime.date.today().isoformat()
    now_iso = datetime.datetime.now().isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Fetch roll number
        cursor.execute("SELECT roll_number FROM students WHERE clean_name = ?", (student_clean_name,))
        r = cursor.fetchone()
        roll = r[0] if r else "N/A"

        cursor.execute("""
            SELECT id FROM attendance_logs 
            WHERE student_name = ? AND session_date = ? AND course_code = ?
        """, (student_clean_name, today, course_code))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("""
                UPDATE attendance_logs 
                SET status = ?, remarks = ?, method = 'MANUAL_OVERRIDE', timestamp = ?
                WHERE id = ?
            """, (status, remarks, now_iso, existing["id"]))
        else:
            cursor.execute("""
                INSERT INTO attendance_logs (student_name, roll_number, timestamp, confidence_score, status, session_date, course_code, method, remarks)
                VALUES (?, ?, ?, 1.0, ?, ?, ?, 'MANUAL_OVERRIDE', ?)
            """, (student_clean_name, roll, now_iso, status, today, course_code, remarks))
        conn.commit()
    return True


def get_defaulters_analysis(course_code: str = "CS-401", min_threshold_pct: float = 75.0) -> Dict[str, Any]:
    """
    Computes rigorous university-grade defaulter analytics.
    Flags students below min_threshold_pct and calculates:
    - Current attendance percentage
    - Total lectures attended vs total held
    - Consecutive classes needed to clear the defaulter list
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Count total distinct lecture dates held for this course
        cursor.execute("""
            SELECT COUNT(DISTINCT session_date) FROM attendance_logs WHERE course_code = ?
        """, (course_code,))
        total_sessions = cursor.fetchone()[0]
        if total_sessions == 0:
            total_sessions = 1

        # Fetch per-student attendance count
        cursor.execute("""
            SELECT 
                s.name, s.clean_name, s.roll_number, s.department, s.email,
                SUM(CASE WHEN a.status IN ('PRESENT', 'EXCUSED') THEN 1 ELSE 0 END) as attended_count,
                COUNT(a.id) as logged_count
            FROM students s
            JOIN course_enrollments ce ON s.clean_name = ce.student_clean_name
            LEFT JOIN attendance_logs a ON (s.clean_name = a.student_name AND a.course_code = ?)
            WHERE ce.course_code = ?
            GROUP BY s.clean_name
            ORDER BY attended_count ASC
        """, (course_code, course_code))

        students_summary = []
        defaulters = []

        for row in cursor.fetchall():
            d = dict(row)
            attended = d["attended_count"] or 0
            pct = min(100.0, round((attended / total_sessions) * 100, 1))
            d["total_sessions"] = total_sessions
            d["attended"] = attended
            d["attendance_pct"] = pct

            # Calculate classes required to hit min_threshold_pct
            # (attended + x) / (total_sessions + x) >= threshold / 100
            # attended + x >= 0.75 * total_sessions + 0.75 * x
            # 0.25 * x >= 0.75 * total_sessions - attended
            needed = 0
            if pct < min_threshold_pct:
                t = min_threshold_pct / 100.0
                if t < 1.0:
                    numerator = (t * total_sessions) - attended
                    denominator = 1.0 - t
                    needed = max(0, int(round(numerator / denominator + 0.49)))
                d["classes_needed"] = needed
                d["is_defaulter"] = True
                defaulters.append(d)
            else:
                d["classes_needed"] = 0
                d["is_defaulter"] = False

            students_summary.append(d)

    return {
        "course_code": course_code,
        "total_sessions_held": total_sessions,
        "min_threshold_pct": min_threshold_pct,
        "defaulters_count": len(defaulters),
        "defaulters": defaulters,
        "all_students": students_summary
    }


def get_analytics_summary(course_code: str = "CS-401") -> Dict[str, Any]:
    """Generates time-series and day-by-day attendance trends for charts."""
    today = datetime.date.today().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Trend by date
        cursor.execute("""
            SELECT session_date, 
                   SUM(CASE WHEN status IN ('PRESENT', 'EXCUSED') THEN 1 ELSE 0 END) as present_count,
                   COUNT(*) as total_records
            FROM attendance_logs
            WHERE course_code = ?
            GROUP BY session_date
            ORDER BY session_date ASC
            LIMIT 10
        """, (course_code,))
        trend = [dict(row) for row in cursor.fetchall()]

        # Method breakdown
        cursor.execute("""
            SELECT method, COUNT(*) as count
            FROM attendance_logs
            WHERE course_code = ?
            GROUP BY method
        """, (course_code,))
        methods = [dict(row) for row in cursor.fetchall()]

    return {
        "course_code": course_code,
        "trend": trend,
        "methods": methods
    }


def get_student_detail(student_clean_name: str) -> Optional[Dict[str, Any]]:
    """Returns detailed student dossier with enrollment, crops, and log history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE clean_name = ?", (student_clean_name,))
        student_row = cursor.fetchone()
        if not student_row:
            return None

        student = dict(student_row)

        cursor.execute("""
            SELECT * FROM attendance_logs 
            WHERE student_name = ? 
            ORDER BY id DESC LIMIT 50
        """, (student_clean_name,))
        student["attendance_history"] = [dict(row) for row in cursor.fetchall()]

    # Check dataset crops
    crops_dir = BASE_DIR / "dataset" / student_clean_name
    crops = []
    if crops_dir.exists():
        for img_p in crops_dir.glob("*.jpg"):
            crops.append(img_p.name)
    student["crops_available"] = crops

    return student


def get_attendance_records(limit: int = 150) -> List[Dict[str, Any]]:
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

        cursor.execute("SELECT COUNT(DISTINCT student_name) FROM attendance_logs WHERE session_date = ? AND status = 'PRESENT'", (today,))
        present_today = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence_score) FROM attendance_logs WHERE confidence_score > 0")
        avg_conf_row = cursor.fetchone()[0]
        avg_conf = float(avg_conf_row) if avg_conf_row is not None else 0.0

        cursor.execute("SELECT COUNT(*) FROM courses")
        total_courses = cursor.fetchone()[0]

    return {
        "total_enrolled": total_students,
        "present_today": present_today,
        "avg_confidence": round(avg_conf * 100, 1),
        "total_courses": total_courses,
        "today_date": today
    }


def purge_db_records():
    """Wipes all student records and attendance logs upon DoD purge execution."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students")
        cursor.execute("DELETE FROM attendance_logs")
        cursor.execute("DELETE FROM course_enrollments")
        cursor.execute("DELETE FROM lecture_sessions")
        cursor.execute("DELETE FROM audit_events")
        conn.commit()


if __name__ == "__main__":
    init_db()
    print("[Database] University schema initialized successfully.")
    print("[Database] Stats:", get_dashboard_stats())
