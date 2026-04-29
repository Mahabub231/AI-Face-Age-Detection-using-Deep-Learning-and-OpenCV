/* =============================================================
   AI FaceAge — main.js
   ============================================================= */

// ── Theme Toggle ──────────────────────────────────────────────
const root       = document.documentElement;
const savedTheme = localStorage.getItem("theme") || "dark";
root.setAttribute("data-theme", savedTheme);

const themeBtn = document.getElementById("themeToggle");
if (themeBtn) {
  themeBtn.textContent = savedTheme === "dark" ? "🌙" : "☀️";
  themeBtn.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    themeBtn.textContent = next === "dark" ? "🌙" : "☀️";
  });
}

// ── User Avatar Dropdown ──────────────────────────────────────
const userMenu      = document.getElementById("userMenu");
const userAvatarBtn = document.getElementById("userAvatarBtn");
const userDropdown  = document.getElementById("userDropdown");

if (userAvatarBtn && userDropdown) {
  // Toggle on avatar click
  userAvatarBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = userDropdown.classList.contains("show");
    closeDropdown();
    if (!isOpen) openDropdown();
  });

  // Close on outside click
  document.addEventListener("click", (e) => {
    if (userMenu && !userMenu.contains(e.target)) {
      closeDropdown();
    }
  });

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDropdown();
  });

  function openDropdown() {
    userDropdown.classList.add("show");
    userAvatarBtn.classList.add("open");
    userAvatarBtn.setAttribute("aria-expanded", "true");
  }
  function closeDropdown() {
    userDropdown.classList.remove("show");
    userAvatarBtn.classList.remove("open");
    userAvatarBtn.setAttribute("aria-expanded", "false");
  }
}

// ── Auto-dismiss flash messages ───────────────────────────────
setTimeout(() => {
  document.querySelectorAll(".flash").forEach(el => {
    el.style.transition = "opacity .5s, transform .5s";
    el.style.opacity    = "0";
    el.style.transform  = "translateX(120%)";
    setTimeout(() => el.remove(), 520);
  });
}, 4000);

// ── Upload Page: Drag & Drop + Preview ───────────────────────
const imageInput  = document.getElementById("imageInput");
const previewImg  = document.getElementById("previewImage");
const previewEmpty= document.getElementById("previewEmpty");
const fileInfo    = document.getElementById("fileInfo");
const fileNameEl  = document.getElementById("fileName");
const fileSizeEl  = document.getElementById("fileSize");
const uploadForm  = document.getElementById("uploadForm");
const analyzing   = document.getElementById("analyzing");
const dropZone    = document.getElementById("dropZone");

function handleFile(file) {
  if (!file) return;
  if (previewImg)   { previewImg.src = URL.createObjectURL(file); previewImg.style.display = "block"; }
  if (previewEmpty) previewEmpty.style.display = "none";
  if (fileInfo)     fileInfo.style.display = "block";
  if (fileNameEl)   fileNameEl.textContent = "📄 " + file.name;
  if (fileSizeEl)   fileSizeEl.textContent = "💾 " + (file.size / 1024).toFixed(1) + " KB";
}

if (imageInput) {
  imageInput.addEventListener("change", () => handleFile(imageInput.files[0]));
}
if (dropZone) {
  dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.style.borderColor = "var(--cyan)"; dropZone.style.background = "rgba(34,211,238,.06)"; });
  dropZone.addEventListener("dragleave", ()  => { dropZone.style.borderColor = ""; dropZone.style.background = ""; });
  dropZone.addEventListener("drop",      (e) => {
    e.preventDefault();
    dropZone.style.borderColor = ""; dropZone.style.background = "";
    if (e.dataTransfer.files.length) {
      const dt = new DataTransfer();
      dt.items.add(e.dataTransfer.files[0]);
      imageInput.files = dt.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });
}
if (uploadForm) {
  uploadForm.addEventListener("submit", () => {
    if (analyzing) { analyzing.style.display = "block"; }
    const btn = document.getElementById("analyzeBtn");
    if (btn) { btn.disabled = true; btn.textContent = "⏳ Analyzing…"; }
  });
}

// ── Animate confidence bars (result page) ─────────────────────
window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".frc-conf-fill, .conf-bar-fill").forEach(el => {
    const w = el.style.width;
    el.style.width = "0";
    setTimeout(() => { el.style.width = w; }, 300 + Math.random() * 200);
  });
});
