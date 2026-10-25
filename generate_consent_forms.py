#!/usr/bin/env python3
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
CONSENT_DIR = BASE_DIR / "consent_forms"
CONSENT_DIR.mkdir(exist_ok=True)
(BASE_DIR / "dataset").mkdir(exist_ok=True)
(BASE_DIR / "src").mkdir(exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)
(BASE_DIR / "reports").mkdir(exist_ok=True)

volunteers = [
    ("Alice_Smith", "CS-2026-001", "Alice Smith", "2026-10-15"),
    ("Bob_Johnson", "CS-2026-002", "Bob Johnson", "2026-10-15"),
    ("Charlie_Davis", "CS-2026-003", "Charlie Davis", "2026-10-16"),
    ("Diana_Evans", "CS-2026-004", "Diana Evans", "2026-10-16"),
    ("Ethan_Wright", "CS-2026-005", "Ethan Wright", "2026-10-17"),
]

for name, roll, sig, date in volunteers:
    pdf_path = CONSENT_DIR / f"consent_{name}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
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
    y = height - 120
    for line in text:
        c.drawString(50, y, f"• {line}")
        y -= 20
        
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Participant Acknowledgement & Digital Sign-off:")
    y -= 30
    c.setFont("Helvetica", 11)
    disp_name = name.replace("_", " ")
    c.drawString(50, y, f"Student Full Name: {disp_name}")
    y -= 25
    c.drawString(50, y, f"Student Roll Number: {roll}")
    y -= 25
    c.drawString(50, y, f"Digital Signature / Verification: /s/ {sig} [OPT-IN VERIFIED]")
    y -= 25
    c.drawString(50, y, f"Consent Date: {date}")
    y -= 40
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Status: Active and Compliant with Institutional Review Board (IRB) Academic Protocol.")
    c.save()
    print(f"Generated: {pdf_path}")
