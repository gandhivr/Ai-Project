/**
 * Anti-Gravity Pro Suite: Smart Classroom Attendance Client Application
 * Features: Live Video Canvas, Gaussian Privacy Blur, Dynamic Enrollment, Database Query, DoD Purge
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    activeTab: "tab-live",
    systemStatus: null,
    webcamStream: null,
    enrollWebcamStream: null,
    isVerifying: false,
    enrollCrops: [],
    recentDetections: new Set(),
  };

  // DOM Elements
  const navTabs = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const liveClock = document.getElementById("live-clock");
  const statusPill = document.getElementById("system-status-pill");
  const statusText = document.getElementById("system-status-text");

  // Video / Canvas Elements (Live Tab)
  const videoLive = document.getElementById("webcam-live");
  const canvasOverlay = document.getElementById("canvas-overlay");
  const ctxOverlay = canvasOverlay.getContext("2d");
  const simulatedImage = document.getElementById("simulated-image");
  const btnToggleCamera = document.getElementById("btn-toggle-camera");
  const btnSimulateVerified = document.getElementById("btn-simulate-verified");
  const btnSimulateUnknown = document.getElementById("btn-simulate-unknown");
  const tickerList = document.getElementById("ticker-list");
  const tickerCount = document.getElementById("ticker-count");

  // Enrollment Elements
  const videoEnroll = document.getElementById("webcam-enroll");
  const canvasEnroll = document.getElementById("canvas-enroll");
  const enrollNameInput = document.getElementById("enroll-name");
  const enrollRollInput = document.getElementById("enroll-roll");
  const enrollConsentCheckbox = document.getElementById("enroll-consent");
  const btnCaptureCrop = document.getElementById("btn-capture-crop");
  const btnFinalizeTrain = document.getElementById("btn-finalize-train");
  const enrollCounterBadge = document.getElementById("enroll-counter-badge");
  const cropsGallery = document.getElementById("crops-gallery");

  // Records Table Elements
  const tableBody = document.getElementById("attendance-table-body");
  const searchInput = document.getElementById("search-attendance");
  const btnRefreshRecords = document.getElementById("btn-refresh-records");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const btnExportJson = document.getElementById("btn-export-json");

  // Metric Cards
  const metricEnrolled = document.getElementById("metric-enrolled");
  const metricPresent = document.getElementById("metric-present");
  const metricConfidence = document.getElementById("metric-confidence");

  // Purge Modal Elements
  const purgeModal = document.getElementById("purge-modal");
  const btnOpenPurgeModal = document.getElementById("btn-open-purge-modal");
  const btnCancelPurge = document.getElementById("btn-cancel-purge");
  const btnConfirmPurge = document.getElementById("btn-confirm-purge");

  // -------------------------------------------------------------
  // 1. Clock & Periodic Status Polling
  // -------------------------------------------------------------
  function updateClock() {
    const now = new Date();
    liveClock.textContent = now.toTimeString().split(" ")[0];
  }
  setInterval(updateClock, 1000);
  updateClock();

  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      state.systemStatus = data;

      if (data.cold_start) {
        statusPill.className = "status-indicator cold";
        statusText.textContent = "COLD-START: 0 ENROLLED";
      } else {
        statusPill.className = "status-indicator";
        statusText.textContent = `SYSTEM ACTIVE: ${data.enrolled_count} ENROLLED (${data.present_today} PRESENT)`;
      }

      if (metricEnrolled) metricEnrolled.textContent = data.enrolled_count;
      if (metricPresent) metricPresent.textContent = data.present_today;
      if (metricConfidence) metricConfidence.textContent = `${data.avg_confidence}%`;
    } catch (err) {
      console.error("Status polling failed:", err);
    }
  }

  fetchSystemStatus();
  setInterval(fetchSystemStatus, 6000);

  // -------------------------------------------------------------
  // 2. Navigation Tabs
  // -------------------------------------------------------------
  navTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navTabs.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const activePane = document.getElementById(targetTab);
      if (activePane) activePane.classList.add("active");
      state.activeTab = targetTab;

      if (targetTab === "tab-records") {
        loadAttendanceTable();
      } else if (targetTab === "tab-enroll") {
        startEnrollmentCamera();
      }
    });
  });

  // -------------------------------------------------------------
  // 3. Live Door Attendance & Real-Time Verification Canvas
  // -------------------------------------------------------------
  async function startLiveCamera() {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        videoLive.srcObject = stream;
        state.webcamStream = stream;
        videoLive.style.display = "block";
        simulatedImage.style.display = "none";
        showToast("Webcam connected. Real-time door monitoring active.");
        startVerificationLoop();
      }
    } catch (err) {
      console.warn("Webcam not available:", err);
      showToast("No physical webcam found. Using Simulated Campus Feed.");
      injectSimulatedFrame("Alice_Smith");
    }
  }

  function startVerificationLoop() {
    if (state.isVerifying) return;
    state.isVerifying = true;

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = 640;
    captureCanvas.height = 480;
    const captureCtx = captureCanvas.getContext("2d");

    setInterval(async () => {
      if (state.activeTab !== "tab-live" || !state.webcamStream) return;

      captureCtx.drawImage(videoLive, 0, 0, 640, 480);
      const b64 = captureCanvas.toDataURL("image/jpeg", 0.7);

      try {
        const res = await fetch("/api/verify_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: b64 }),
        });
        const data = await res.json();
        renderDetections(data, videoLive);
      } catch (e) {
        console.error("Frame verification failed:", e);
      }
    }, 280);
  }

  btnToggleCamera.addEventListener("click", () => {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
      videoLive.srcObject = null;
      videoLive.style.display = "none";
      showToast("Camera stopped.");
    } else {
      startLiveCamera();
    }
  });

  // Render bounding boxes & 51x51 Gaussian Privacy Blur
  function renderDetections(data, sourceElement) {
    ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    if (data.cold_start) {
      ctxOverlay.fillStyle = "rgba(220, 38, 38, 0.85)";
      ctxOverlay.fillRect(0, 0, 640, 40);
      ctxOverlay.fillStyle = "#ffffff";
      ctxOverlay.font = "bold 14px Inter";
      ctxOverlay.fillText("STATUS: Zero Students Enrolled. Please run admin enrollment.", 15, 25);
      return;
    }

    if (!data.faces || data.faces.length === 0) return;

    data.faces.forEach((face) => {
      const [x, y, fw, fh] = face.box;

      if (face.is_blurred) {
        // UNKNOWN / NON-CONSENTING BYSTANDER
        // 1. Live Optical Scrambling directly on canvas (51x51 Gaussian blur equivalent)
        ctxOverlay.save();
        ctxOverlay.beginPath();
        ctxOverlay.rect(x, y, fw, fh);
        ctxOverlay.clip();
        ctxOverlay.filter = "blur(22px)";
        ctxOverlay.drawImage(sourceElement, 0, 0, 640, 480);
        ctxOverlay.restore();

        // 2. Draw Crimson Bounding Box & Label
        ctxOverlay.strokeStyle = "#f43f5e";
        ctxOverlay.lineWidth = 2.5;
        ctxOverlay.strokeRect(x, y, fw, fh);

        ctxOverlay.fillStyle = "rgba(244, 63, 94, 0.9)";
        ctxOverlay.fillRect(x, Math.max(0, y - 24), 160, 24);
        ctxOverlay.fillStyle = "#ffffff";
        ctxOverlay.font = "bold 12px Inter";
        ctxOverlay.fillText("Unknown (Blurred)", x + 6, Math.max(16, y - 7));
      } else {
        // ENROLLED & VERIFIED STUDENT
        // 1. Draw Emerald Bounding Box & Tag
        ctxOverlay.strokeStyle = "#10b981";
        ctxOverlay.lineWidth = 2.5;
        ctxOverlay.strokeRect(x, y, fw, fh);

        const tag = `${face.student_name} (${face.confidence}%)`;
        ctxOverlay.fillStyle = "rgba(16, 185, 129, 0.9)";
        ctxOverlay.fillRect(x, Math.max(0, y - 24), tag.length * 8.5 + 10, 24);
        ctxOverlay.fillStyle = "#ffffff";
        ctxOverlay.font = "bold 12px Inter";
        ctxOverlay.fillText(tag, x + 6, Math.max(16, y - 7));

        // 2. Add to Live Event Ticker
        appendTickerRecord(face.student_name, face.confidence);
      }
    });
  }

  // Simulated Frame Injection for testing without physical camera
  async function injectSimulatedFrame(name) {
    try {
      const res = await fetch(`/api/sample_face?name=${name}`);
      const data = await res.json();
      simulatedImage.src = data.image_base64;
      simulatedImage.style.display = "block";
      videoLive.style.display = "none";

      simulatedImage.onload = async () => {
        const verifyRes = await fetch("/api/verify_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: data.image_base64 }),
        });
        const vData = await verifyRes.json();
        renderDetections(vData, simulatedImage);

        if (name === "Unknown") {
          showToast("Injected Unknown Face: 51×51 Gaussian blur mask active!");
        } else {
          showToast(`Injected Enrolled Face: ${name.replace("_", " ")} verified!`);
        }
      };
    } catch (e) {
      console.error("Simulation failed:", e);
    }
  }

  const btnSimulateVraj = document.getElementById("btn-simulate-vraj");
  if (btnSimulateVraj) {
    btnSimulateVraj.addEventListener("click", () => injectSimulatedFrame("Vraj_Gandhi"));
  }
  btnSimulateVerified.addEventListener("click", () => injectSimulatedFrame("Alice_Smith"));
  btnSimulateUnknown.addEventListener("click", () => injectSimulatedFrame("Unknown"));

  // Append new verified student card to ticker
  function appendTickerRecord(name, confidence) {
    if (state.recentDetections.has(name)) return;
    state.recentDetections.add(name);

    const timeStr = new Date().toTimeString().split(" ")[0];
    const initials = name
      .split(" ")
      .map((n) => n[0])
      .join("");

    const card = document.createElement("div");
    card.className = "ticker-card";
    card.innerHTML = `
      <div class="ticker-avatar">${initials}</div>
      <div class="ticker-info">
        <div class="ticker-name">${name}</div>
        <div class="ticker-meta">${timeStr} • Verified</div>
      </div>
      <span class="ticker-badge">${confidence}%</span>
    `;

    tickerList.prepend(card);
    tickerCount.textContent = `${state.recentDetections.size} Marked`;
  }

  // -------------------------------------------------------------
  // 4. Admin Enrollment Studio
  // -------------------------------------------------------------
  async function startEnrollmentCamera() {
    if (state.enrollWebcamStream) return;
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        videoEnroll.srcObject = stream;
        state.enrollWebcamStream = stream;
      }
    } catch (err) {
      console.warn("Enrollment webcam unavailable:", err);
    }
  }

  btnCaptureCrop.addEventListener("click", async () => {
    const name = enrollNameInput.value.trim();
    const roll = enrollRollInput.value.trim();

    if (!name || !roll) {
      showToast("Please enter Student Name and Roll Number first.", "warn");
      return;
    }

    if (!enrollConsentCheckbox.checked) {
      showToast("Consent agreement must be acknowledged.", "warn");
      return;
    }

    // Capture frame from enrollment video
    const capCanvas = document.createElement("canvas");
    capCanvas.width = 640;
    capCanvas.height = 480;
    const capCtx = capCanvas.getContext("2d");

    if (state.enrollWebcamStream) {
      capCtx.drawImage(videoEnroll, 0, 0, 640, 480);
    } else {
      // Use sample face if no physical camera
      const sample = await fetch("/api/sample_face?name=Diana_Evans");
      const sData = await sample.json();
      const img = new Image();
      img.src = sData.image_base64;
      await new Promise((r) => (img.onload = r));
      capCtx.drawImage(img, 0, 0, 640, 480);
    }

    const b64 = capCanvas.toDataURL("image/jpeg", 0.85);

    try {
      btnCaptureCrop.disabled = true;
      const res = await fetch("/api/enroll_crop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_name: name,
          roll_number: roll,
          image_base64: b64,
        }),
      });
      const data = await res.json();
      btnCaptureCrop.disabled = false;

      if (data.status === "SUCCESS") {
        state.enrollCrops.push(data.thumbnail);
        renderGallery();
        enrollCounterBadge.textContent = `${data.crop_count} / 10 Crops`;
        showToast(`Captured crop ${data.crop_count}/10 for ${name}!`);

        if (data.crop_count >= 5) {
          btnFinalizeTrain.disabled = false;
        }
      }
    } catch (e) {
      btnCaptureCrop.disabled = false;
      showToast("Crop capture failed.", "error");
    }
  });

  function renderGallery() {
    cropsGallery.innerHTML = "";
    state.enrollCrops.forEach((thumb) => {
      const img = document.createElement("img");
      img.className = "crop-thumbnail";
      img.src = thumb;
      cropsGallery.appendChild(img);
    });
  }

  btnFinalizeTrain.addEventListener("click", async () => {
    btnFinalizeTrain.disabled = true;
    btnFinalizeTrain.innerHTML = `<span>⏳</span> Retraining Model...`;

    try {
      const res = await fetch("/api/retrain", { method: "POST" });
      const data = await res.json();
      btnFinalizeTrain.disabled = false;
      btnFinalizeTrain.innerHTML = `<span>⚡</span> Finalize & Retrain Model (Press 'Q')`;

      if (data.status === "SUCCESS") {
        showToast(`🎉 Retrained successfully! Enrolled: ${data.classes.length} students.`);
        fetchSystemStatus();
      } else {
        showToast(data.message, "error");
      }
    } catch (e) {
      btnFinalizeTrain.disabled = false;
      showToast("Retraining failed.", "error");
    }
  });

  // Keyboard Shortcuts: 'c' to capture, 'q' to finalize
  window.addEventListener("keydown", (e) => {
    if (state.activeTab === "tab-enroll") {
      if ((e.key === "c" || e.key === "C") && !btnCaptureCrop.disabled) {
        btnCaptureCrop.click();
      } else if ((e.key === "q" || e.key === "Q") && !btnFinalizeTrain.disabled) {
        btnFinalizeTrain.click();
      }
    }
  });

  // -------------------------------------------------------------
  // 5. Attendance Database Table & Export
  // -------------------------------------------------------------
  async function loadAttendanceTable() {
    try {
      const res = await fetch("/api/attendance");
      const data = await res.json();
      const records = data.records || [];
      renderTableRows(records);
    } catch (e) {
      console.error("Failed to load attendance records:", e);
    }
  }

  function renderTableRows(records) {
    tableBody.innerHTML = "";
    if (records.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-dim); padding: 2rem;">No attendance records found today.</td></tr>`;
      return;
    }

    records.forEach((r) => {
      const tr = document.createElement("tr");
      const confPct = Math.round(r.confidence_score * 100);
      const dispRoll = r.roll_number || "REG-VERIFIED";

      tr.innerHTML = `
        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-muted);">${r.timestamp}</td>
        <td style="font-weight: 600;">${r.student_name.replace("_", " ")}</td>
        <td style="font-family: 'JetBrains Mono', monospace; color: var(--cyan);">${dispRoll}</td>
        <td>
          <div class="conf-bar-wrapper">
            <span>${confPct}%</span>
            <div class="conf-bar">
              <div class="conf-bar-fill" style="width: ${confPct}%;"></div>
            </div>
          </div>
        </td>
        <td>
          <span class="ticker-badge">${r.status}</span>
        </td>
      `;
      tableBody.appendChild(tr);
    });
  }

  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase();
    const rows = tableBody.querySelectorAll("tr");
    rows.forEach((row) => {
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(query) ? "" : "none";
    });
  });

  btnRefreshRecords.addEventListener("click", loadAttendanceTable);

  // CSV Export
  btnExportCsv.addEventListener("click", async () => {
    const res = await fetch("/api/attendance");
    const data = await res.json();
    const rows = [["Timestamp", "Student Name", "Roll Number", "Confidence", "Status"]];
    (data.records || []).forEach((r) => {
      rows.push([r.timestamp, r.student_name, r.roll_number || "", r.confidence_score, r.status]);
    });
    const csvContent = "data:text/csv;charset=utf-8," + rows.map((e) => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `attendance_${new Date().toISOString().split("T")[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Attendance CSV exported successfully!");
  });

  // JSON Export
  btnExportJson.addEventListener("click", async () => {
    const res = await fetch("/api/attendance");
    const data = await res.json();
    const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data.records, null, 2));
    const link = document.createElement("a");
    link.setAttribute("href", jsonStr);
    link.setAttribute("download", `attendance_${new Date().toISOString().split("T")[0]}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Attendance JSON exported successfully!");
  });

  // -------------------------------------------------------------
  // 6. DoD 5220.22-M Cryptographic Data Purge Modal
  // -------------------------------------------------------------
  btnOpenPurgeModal.addEventListener("click", () => purgeModal.classList.add("active"));
  btnCancelPurge.addEventListener("click", () => purgeModal.classList.remove("active"));

  btnConfirmPurge.addEventListener("click", async () => {
    btnConfirmPurge.disabled = true;
    btnConfirmPurge.textContent = "Executing 3-Pass Overwrite...";

    try {
      const res = await fetch("/api/purge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_wipe: true }),
      });
      const data = await res.json();
      purgeModal.classList.remove("active");
      btnConfirmPurge.disabled = false;
      btnConfirmPurge.textContent = "Execute DoD Wipe";

      showToast(`🔥 PURGE COMPLETE: ${data.files_destroyed} files destroyed (${data.bytes_wiped} bytes shredded).`);
      fetchSystemStatus();
      loadAttendanceTable();
    } catch (e) {
      btnConfirmPurge.disabled = false;
      btnConfirmPurge.textContent = "Execute DoD Wipe";
      showToast("Purge execution failed.", "error");
    }
  });

  // Toast Notification Helper
  function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<span>${type === "error" ? "❌" : type === "warn" ? "⚠️" : "✨"}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // Initial trigger
  startLiveCamera();
});
