# Participant Informed Consent Agreement: Smart Classroom Attendance Prototype

**Project Title:** Classical Computer Vision Attendance Prototype  
**Evaluation Scope:** Academic Research & Evaluation Only  
**Protocol Version:** 1.0 (Ethics-First Compliance)

---

## 1. Purpose of the Study
This academic prototype evaluates the performance, accuracy, and operational fairness of a classical computer vision attendance verification system using Haar Cascade face detection, Uniform Local Binary Patterns (LBP) feature histograms, and distance-gated k-Nearest Neighbors (kNN) classification.

---

## 2. Data Collection and Resolution Cap
- **Capped Resolution:** All biometric face crops are preprocessed, downsampled, and stored strictly at a maximum resolution of **100×100 pixels**. No high-resolution imagery or original camera streams are preserved.
- **Sample Limit:** A maximum of **10 face image crops** will be captured per registered volunteer.
- **Storage Directory:** Stored strictly in local sandboxed storage under `dataset/<student_name>/`.
- **Zero Network Telemetry:** No biometric features, images, identifiers, or logs are transmitted over the internet or cloud infrastructure. All computation runs purely locally.

---

## 3. Privacy, Masking & Bystander Protection
- **Live Privacy Masking:** Any face detected in the live attendance camera feed that is unregistered, unverified, or exceeds the strict acceptance distance threshold is classified as *Non-Consenting / Unknown*.
- **Instant Optical Scrambling:** Non-consenting faces are immediately masked with a real-time 51×51 Gaussian blur ($\sigma = 30$) directly in the video display pipeline.
- **Zero Logging of Non-Consenting Individuals:** No images, embeddings, timestamps, or metadata belonging to unrecognized or non-consenting individuals are ever written to disk or logged.

---

## 4. Retention Period and Cryptographic Wiping
- **Zero-Retention Policy:** Collected face crops and trained model artifacts (`model.pkl`) are retained strictly for the duration of the course evaluation.
- **DoD 5220.22-M Automated Wiping:** Upon course completion, participants or administrators can trigger the automated cryptographic wipe utility (`python purge_data.py --wipe`). This executes a 3-pass DoD 5220.22-M compliant overwrite (Pass 1: zeros `0x00`, Pass 2: ones `0xFF`, Pass 3: cryptographically random salted bytes, followed by file truncation and unlinking) on all image files and serialized model parameters.

---

## 5. Voluntary Participation & Revocation Rights
Participation is 100% voluntary. You retain the absolute right to:
1. Withdraw consent at any time without penalty or academic disadvantage.
2. Request immediate deletion of your enrolled face crops and exclusion from the serialized model.
3. Inspect any attendance logs recorded under your student name.

---

## Participant Statement of Consent
By signing below or opting in digitally, I acknowledge that:
1. I have read and understood the data collection limits (max 10 crops, 100×100 px).
2. I understand that my data is processed solely on local hardware without cloud telemetry.
3. I may revoke my consent and request cryptographic deletion at any time.

**Volunteer Student Full Name:** _____________________________________________  
**Student Roll / ID Number:** _____________________________________________  
**Signature / Digital Token:** _____________________________________________  
**Date:** ________________________  
