// COLESION-AI // CLINICAL DIAGNOSTIC WORKSTATION
// Keyboard-First HUD & Parallel AI Forecast Architecture

let allCasesData = [];
let casesData = [];
let currentCaseIndex = 0;
let caseStates = []; // Isolated state per case across A/D navigation

// 7 HAM10000 Diagnostic Classes + 1 Inconclusive Choice (Full Clinical Names in English)
const DIAG_CLASSES = [
  { key: "1", id: "mel", name: "Melanoma", isMalignant: true },
  { key: "2", id: "nv", name: "Melanocytic Nevus", isMalignant: false },
  { key: "3", id: "bcc", name: "Basal Cell Carcinoma", isMalignant: true },
  { key: "4", id: "akiec", name: "Actinic Keratosis", isMalignant: true },
  { key: "5", id: "bkl", name: "Benign Keratosis", isMalignant: false },
  { key: "6", id: "df", name: "Dermatofibroma", isMalignant: false },
  { key: "7", id: "vasc", name: "Vascular Lesion", isMalignant: false },
  { key: "8", id: "unclear", name: "Inconclusive / Non-diagnostic", isMalignant: false }
];

// Helper to look up DIAG_CLASS by ID or text string
function getDiagClassFromText(rawText) {
  if (!rawText) return null;
  const lower = rawText.toLowerCase().trim();

  // 1. Match exact ID within word boundaries or parentheses, e.g. "(nv)", "(mel)", "akiec"
  // This avoids substring false positives like "mel" matching inside "melanocytic"!
  for (const c of DIAG_CLASSES) {
    const idRegex = new RegExp(`\\b${c.id}\\b`, "i");
    if (idRegex.test(lower)) {
      return c;
    }
  }

  // 2. Match exact full clinical class name
  for (const c of DIAG_CLASSES) {
    if (lower.includes(c.name.toLowerCase())) {
      return c;
    }
  }

  return null;
}

// Threshold to decide whether model is confident or uncertain (close rivalry)
const CONFIDENCE_MARGIN_THRESHOLD = 0.20; // 20% margin delta

// Triple-Projection Anatomical Coordinates (Anterior, Posterior, Lateral)
const PROJECTION_COORDS = {
  "back": {
    front: null,
    back:  { x: 50, y: 28, visible: true },
    side:  { x: 80, y: 28, visible: true }
  },
  "lower extremity": {
    front: { x: 38, y: 78, visible: true },
    back:  null,
    side:  { x: 50, y: 78, visible: true }
  },
  "face / scalp": {
    front: { x: 50, y: 9, visible: true },
    back:  null,
    side:  { x: 30, y: 9, visible: true }
  },
  "face": {
    front: { x: 50, y: 9, visible: true },
    back:  null,
    side:  { x: 30, y: 9, visible: true }
  },
  "trunk": {
    front: { x: 50, y: 28, visible: true },
    back:  null,
    side:  { x: 30, y: 28, visible: true }
  },
  "abdomen": {
    front: { x: 50, y: 40, visible: true },
    back:  null,
    side:  { x: 32, y: 40, visible: true }
  },
  "upper extremity": {
    front: { x: 22, y: 38, visible: true },
    back:  null,
    side:  { x: 50, y: 38, visible: true }
  }
};

// Timers & States
let dwellTimerInterval = null;
let isWHeld = false;

// DOM Elements
const prevCaseBtn = document.getElementById("prevCaseBtn");
const nextCaseBtn = document.getElementById("nextCaseBtn");
const caseCounterEl = document.getElementById("caseCounter");
const specimenIdEl = document.getElementById("specimenId");
const caseLockStatusEl = document.getElementById("caseLockStatus");
const lockStatusTextEl = document.getElementById("lockStatusText");
const resetBtn = document.getElementById("resetBtn");

const diagButtonsGroup = document.getElementById("diagButtonsGroup");
const btnAiToggle = document.getElementById("btnAiToggle");
const aiToggleLabel = document.getElementById("aiToggleLabel");
const aiStatusCard = document.getElementById("aiStatusCard");
const btnCompareRival = document.getElementById("btnCompareRival");

const baseImageEl = document.getElementById("baseImage");
const heatmapImageEl = document.getElementById("heatmapImage");
const heatmapOpacitySlider = document.getElementById("heatmapOpacity");
const opacityValEl = document.getElementById("opacityVal");
const btnHeatmapDown = document.getElementById("btnHeatmapDown");
const btnHeatmapUp = document.getElementById("btnHeatmapUp");
const btnHoldWTwin = document.getElementById("btnHoldWTwin");

// Twins Panels (Left & Right 3+3)
const leftPanelNormal = document.getElementById("leftPanelNormal");
const leftPanelTwins = document.getElementById("leftPanelTwins");
const leftTwinsTitle = document.getElementById("leftTwinsTitle");
const leftTwinsSubtitle = document.getElementById("leftTwinsSubtitle");
const leftTwinsCardsRow = document.getElementById("leftTwinsCardsRow");
const btnCloseTwinsLeft = document.getElementById("btnCloseTwinsLeft");

const rightPanelNormal = document.getElementById("rightPanelNormal");
const rightPanelTwins = document.getElementById("rightPanelTwins");
const rightTwinsTitle = document.getElementById("rightTwinsTitle");
const rightTwinsSubtitle = document.getElementById("rightTwinsSubtitle");
const rightTwinsCardsRow = document.getElementById("rightTwinsCardsRow");
const btnCloseTwinsRight = document.getElementById("btnCloseTwinsRight");

const btnTwinsMode = document.getElementById("btnTwinsMode");

let isTwinsViewActive = false;
let twinsStrategy = "contrastive";
let currentLeftClass = null;
let currentRightClass = null; // "contrastive" (3 vs 3) or "latent" (6 nearest)

const pSpecimenId = document.getElementById("pSpecimenId");
const metaNameEl = document.getElementById("metaName");
const metaSexEl = document.getElementById("metaSex");
const metaAgeEl = document.getElementById("metaAge");
const metaSiteEl = document.getElementById("metaSite");

const pinFront = document.getElementById("pinFront");
const pinBack = document.getElementById("pinBack");
const pinSide = document.getElementById("pinSide");

const sliceWarningBox = document.getElementById("sliceWarningBox");
const sliceSiteVal = document.getElementById("sliceSiteVal");
const oodAlertBox = document.getElementById("oodAlertBox");

const dwellTimerBadge = document.getElementById("dwellTimerBadge");
const selectedDiagBadge = document.getElementById("selectedDiagBadge");
const btnSignOff = document.getElementById("btnSignOff");

// Application Initialization
async function initDashboard() {
  loadCalibratedMetrics();

  try {
    let response = await fetch("/api/cases");
    if (!response.ok) response = await fetch("data/cases.json");
    if (!response.ok) throw new Error("HTTP status " + response.status);
    allCasesData = await response.json();
    casesData = allCasesData;
  } catch (err) {
    console.error("Critical: Could not load case dataset:", err);
    return;
  }

  updateExplorerFilterCounts();
  resetAllCaseStates();
  setupGlobalEventListeners();
  setupExplorerControls();
  loadCase(0);
}

async function loadCalibratedMetrics() {
  const displayEl = document.getElementById("topMlAccuracyDisplay");
  const brandEl = document.getElementById("hudBrandContainer");

  try {
    const res = await fetch("/api/metrics");
    if (res.ok) {
      const m = await res.json();
      const valAcc = m.val_accuracy ?? m.val_acc;
      if (typeof valAcc === "number") {
        const accPct = (valAcc * 100).toFixed(1);
        if (displayEl) displayEl.textContent = `ML Accuracy ${accPct}%`;
        if (brandEl) {
          const eceStr = m.ece_calibrated ? ` | Calibrated ECE: ${(m.ece_calibrated * 100).toFixed(1)}%` : "";
          brandEl.title = `Model Accuracy: ${accPct}%${eceStr} (Computed from active ML run). Note: AI can be wrong!`;
        }
        return;
      }
    }
  } catch (err) {
    console.warn("API metrics fetch failed, falling back to dataset calculation:", err);
  }

  // Dynamic calculation fallback directly from dataset cases
  if (casesData && casesData.length > 0) {
    const correctCount = casesData.filter((c) => {
      const gt = c.ground_truth?.short_name || c.dx;
      const pred = c.ai_assessment?.predicted_class?.short_name || c.pred_class;
      return gt && pred && gt === pred;
    }).length;
    const accPct = ((correctCount / casesData.length) * 100).toFixed(1);
    if (displayEl) displayEl.textContent = `ML Accuracy ${accPct}%`;
    if (brandEl) {
      brandEl.title = `Model Accuracy: ${accPct}% (Computed across dataset cases). Note: AI can be wrong!`;
    }
  }
}

function updateExplorerFilterCounts() {
  const filterSelect = document.getElementById("explorerFilterSelect");
  if (!filterSelect) return;

  const total = allCasesData.length;
  const holdout = allCasesData.filter(c => c.split === "holdout").length;
  const pool = allCasesData.filter(c => c.split === "pool").length;
  const disagreements = allCasesData.filter(c => c.scenario_type === "disagreement").length;
  const rivalry = allCasesData.filter(c => c.scenario_type === "rivalry").length;
  const slices = allCasesData.filter(c => c.ai_assessment?.slice_competence?.warning === true).length;
  const oods = allCasesData.filter(c => c.ai_assessment?.ood?.is_ood === true).length;
  const mels = allCasesData.filter(c => c.ground_truth?.short_name === "mel" || c.ai_assessment?.predicted_class?.short_name === "mel").length;

  filterSelect.innerHTML = `
    <option value="all">All Cases (${total})</option>
    <option value="holdout">Holdout Split (${holdout})</option>
    <option value="pool">Pool Split (${pool})</option>
    <option value="disagreement">Disagreements / AI Errors (${disagreements})</option>
    <option value="rivalry">Close Rivalry Δ &lt; 20% (${rivalry})</option>
    <option value="slice">Subgroup Alerts (${slices})</option>
    <option value="ood">OOD Candidates (${oods})</option>
    <option value="mel">Melanoma Cases (${mels})</option>
  `;
}

function applyExplorerFilter(filterVal) {
  if (filterVal === "all") {
    casesData = allCasesData;
  } else if (filterVal === "holdout" || filterVal === "pool") {
    casesData = allCasesData.filter(c => c.split === filterVal);
  } else if (filterVal === "disagreement") {
    casesData = allCasesData.filter(c => c.scenario_type === "disagreement");
  } else if (filterVal === "rivalry") {
    casesData = allCasesData.filter(c => c.scenario_type === "rivalry");
  } else if (filterVal === "slice") {
    casesData = allCasesData.filter(c => c.ai_assessment?.slice_competence?.warning === true);
  } else if (filterVal === "ood") {
    casesData = allCasesData.filter(c => c.ai_assessment?.ood?.is_ood === true);
  } else if (filterVal === "mel") {
    casesData = allCasesData.filter(c => c.ground_truth?.short_name === "mel" || c.ai_assessment?.predicted_class?.short_name === "mel");
  }

  resetAllCaseStates();
  loadCase(0);
  showNotification(`Filter: ${casesData.length} case${casesData.length === 1 ? "" : "s"} loaded.`);
}

function setupExplorerControls() {
  const filterSelect = document.getElementById("explorerFilterSelect");
  const searchInput = document.getElementById("caseSearchInput");

  if (filterSelect) {
    filterSelect.addEventListener("change", (e) => {
      applyExplorerFilter(e.target.value);
    });
  }

  if (searchInput) {
    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const query = searchInput.value.trim().toLowerCase();
        if (!query) return;

        let idx = casesData.findIndex(c => 
          (c.id && c.id.toLowerCase().includes(query)) || 
          (c.image_id && c.image_id.toLowerCase().includes(query))
        );

        if (idx === -1) {
          idx = allCasesData.findIndex(c => 
            (c.id && c.id.toLowerCase().includes(query)) || 
            (c.image_id && c.image_id.toLowerCase().includes(query))
          );
          if (idx !== -1) {
            if (filterSelect) filterSelect.value = "all";
            casesData = allCasesData;
            resetAllCaseStates();
          }
        }

        if (idx !== -1) {
          loadCase(idx);
          searchInput.blur();
          showNotification(`Jumped to ${casesData[idx].id}`);
        } else {
          showNotification(`Case '${query}' not found!`);
        }
      }
    });
  }
}

function resetAllCaseStates() {
  caseStates = casesData.map(() => ({
    selectedClass: null,
    initialChoice: null,
    finalChoice: null,
    firstChoiceLatency: null,
    aiRevealed: false,
    aiRevealedAt: null,
    dwellSeconds: 0,
    startTime: null,
    isSignedOff: false,
    heatmapOpacity: 0
  }));
}

function setupGlobalEventListeners() {
  // Navigation Buttons
  prevCaseBtn.addEventListener("click", () => {
    if (currentCaseIndex > 0) loadCase(currentCaseIndex - 1);
  });

  nextCaseBtn.addEventListener("click", () => {
    if (currentCaseIndex < casesData.length - 1) loadCase(currentCaseIndex + 1);
  });

  // Global Workstation Reset (Resets all cases)
  resetBtn.addEventListener("click", () => {
    resetAllCases();
  });

  // AI Forecast Toggle Button (S)
  btnAiToggle.addEventListener("click", () => {
    toggleAiForecast();
  });

  // Heatmap Controls: Q / E step in 50% increments
  btnHeatmapDown.addEventListener("click", () => adjustHeatmapOpacity(-50));
  btnHeatmapUp.addEventListener("click", () => adjustHeatmapOpacity(50));

  // Slider fine adjustment with mouse (1% step)
  heatmapOpacitySlider.addEventListener("input", (e) => {
    setHeatmapOpacity(parseInt(e.target.value, 10));
  });

  // Toggle W Reference Twins Handler (Left & Right 3+3 Cases)
  btnHoldWTwin.addEventListener("click", () => toggleTwinsView());
  if (btnCompareRival) {
    btnCompareRival.addEventListener("click", () => toggleTwinsView());
  }
  if (btnCloseTwinsLeft) {
    btnCloseTwinsLeft.addEventListener("click", (e) => { e.stopPropagation(); toggleTwinsView(false); });
  }
  if (btnCloseTwinsRight) {
    btnCloseTwinsRight.addEventListener("click", (e) => { e.stopPropagation(); toggleTwinsView(false); });
  }

  const btnTwinsReturnRight = document.getElementById("btnTwinsReturnRight");
  if (btnTwinsReturnRight) {
    btnTwinsReturnRight.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleTwinsView(false);
      showNotification("Zurück im Hauptmenü. [Enter] zum Sign-off drücken.");
    });
  }

  // Clicking a column selects its corresponding hypothesis
  if (leftPanelTwins) {
    leftPanelTwins.addEventListener("click", (e) => {
      if (e.target.closest(".twins-close-btn") || e.target.closest(".twins-return-main-btn")) return;
      if (currentLeftClass) handleDiagnosisInput(currentLeftClass);
    });
  }
  if (rightPanelTwins) {
    rightPanelTwins.addEventListener("click", (e) => {
      if (e.target.closest(".twins-close-btn") || e.target.closest(".twins-return-main-btn")) return;
      if (currentRightClass) handleDiagnosisInput(currentRightClass);
    });
  }

  // Diagnosis Buttons (Click handler)
  const buttons = diagButtonsGroup.querySelectorAll(".diag-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const classId = btn.dataset.class;
      handleDiagnosisInput(classId);
    });
  });

  // Sign-Off Button (Enter)
  btnSignOff.addEventListener("click", () => {
    handleSignOff();
  });

  // Global Keyboard Shortcuts
  document.addEventListener("keydown", handleKeyDown);
  document.addEventListener("keyup", handleKeyUp);
}

function handleKeyDown(e) {
  if (e.target.tagName === "INPUT" && e.target.type === "text") return;

  const key = e.key;

  // 1-8: Diagnosis selection
  if (["1", "2", "3", "4", "5", "6", "7", "8"].includes(key)) {
    e.preventDefault();
    const item = DIAG_CLASSES.find((c) => c.key === key);
    if (item) handleDiagnosisInput(item.id);
    return;
  }

  // W / w: Toggle Reference Twins View (Left & Right 3+3 Panels)
  if (key === "w" || key === "W") {
    e.preventDefault();
    toggleTwinsView();
    return;
  }

  // Q / q: Heatmap Opacity -50%
  if (key === "q" || key === "Q") {
    e.preventDefault();
    adjustHeatmapOpacity(-50);
    return;
  }

  // E / e: Heatmap Opacity +50%
  if (key === "e" || key === "E") {
    e.preventDefault();
    adjustHeatmapOpacity(50);
    return;
  }

  // S / s: Toggle AI Forecast
  if (key === "s" || key === "S") {
    e.preventDefault();
    toggleAiForecast();
    return;
  }

  // A / a: Previous case
  if (key === "a" || key === "A") {
    e.preventDefault();
    if (currentCaseIndex > 0) loadCase(currentCaseIndex - 1);
    return;
  }

  // D / d: Next case
  if (key === "d" || key === "D") {
    e.preventDefault();
    if (currentCaseIndex < casesData.length - 1) loadCase(currentCaseIndex + 1);
    return;
  }

  // Enter: Sign-off or return from Twins view to Main Menu
  if (key === "Enter") {
    e.preventDefault();
    if (isTwinsViewActive) {
      toggleTwinsView(false);
      showNotification("Zurück im Hauptmenü. [Enter] zum Sign-off drücken.");
      return;
    }
    handleSignOff();
    return;
  }
}

function handleKeyUp(e) {
  // W is now a persistent toggle, keyup does nothing
}

// Backward compatibility alias
function showTwinsOverlay(show) {
  toggleTwinsView(show);
}

function loadCase(index) {
  if (index < 0 || index >= casesData.length) return;
  currentCaseIndex = index;

  const currentCase = casesData[currentCaseIndex];
  const state = caseStates[currentCaseIndex];

  // Stop previous timers
  clearInterval(dwellTimerInterval);

  // Initialize start time
  if (!state.startTime) {
    state.startTime = performance.now();
  }

  // Start dwell timer (only if not already signed off)
  updateDwellDisplay(state.dwellSeconds);
  if (!state.isSignedOff) {
    dwellTimerInterval = setInterval(() => {
      state.dwellSeconds += 0.2;
      updateDwellDisplay(state.dwellSeconds);
    }, 200);
  }

  // Header Nav & Lock Status
  caseCounterEl.textContent = `CASE ${currentCaseIndex + 1} OF ${casesData.length}`;
  specimenIdEl.textContent = currentCase.id;
  prevCaseBtn.disabled = currentCaseIndex === 0;
  nextCaseBtn.disabled = currentCaseIndex === casesData.length - 1;

  if (state.isSignedOff) {
    caseLockStatusEl.className = "hud-lock-badge locked";
    lockStatusTextEl.textContent = "SIGNED OFF & LOCKED";
  } else {
    caseLockStatusEl.className = "hud-lock-badge open";
    lockStatusTextEl.textContent = "IN PROGRESS";
  }

  // Specimen & Patient Metadata Table
  pSpecimenId.textContent = currentCase.id;
  if (metaNameEl) metaNameEl.textContent = currentCase.patient.name || "Anonymous";
  metaSexEl.textContent = currentCase.patient.sex || "--";
  metaAgeEl.textContent = currentCase.patient.age ? `${currentCase.patient.age} yrs` : "--";
  metaSiteEl.textContent = currentCase.patient.anatomical_site || "--";

  // Subgroup sensitivity alert & Anatomical Avatar
  updateSubgroupAlert(currentCase, state.aiRevealed);
  updateTripleAvatar(currentCase, state.aiRevealed);

  // OOD Detection: Simple and realistic clinical alert
  const isOod = currentCase.ai_assessment?.ood?.is_ood === true;
  if (isOod) {
    oodAlertBox.classList.remove("hidden");
  } else {
    oodAlertBox.classList.add("hidden");
  }

  // Images
  baseImageEl.src = currentCase.image;
  if (currentCase.heatmap) {
    heatmapImageEl.src = currentCase.heatmap;
    btnHeatmapDown.disabled = false;
    btnHeatmapUp.disabled = false;
    heatmapOpacitySlider.disabled = false;
  } else {
    heatmapImageEl.src = "";
    btnHeatmapDown.disabled = true;
    btnHeatmapUp.disabled = true;
    heatmapOpacitySlider.disabled = true;
  }

  // Restore Heatmap Opacity
  setHeatmapOpacity(state.heatmapOpacity);

  // Setup Reference Twins
  renderTwinsData(currentCase);
  updateTwinsColumnHighlighting(state.selectedClass);

  // Reset/Render Diagnosis Buttons Selection
  renderButtonsState(state);

  // STRICT PER-CASE AI FORECAST STATE RESTORATION:
  if (state.aiRevealed && !isOod) {
    applyAiForecast(currentCase);
  } else {
    hideAiForecast();
  }
  updateAiToggleStyle(currentCase, state.aiRevealed);

  // Update Sign-Off Panel
  updateSignOffPanel(state);
}

// DYNAMIC COLOR CODING FOR "S AI FORECAST" BASED ON CLINICAL WARNING LEVEL:
// - nichts -> grün (status-normal)
// - Close rivalry -> gelb (status-warning)
// - SUBGROUP SENSITIVITY -> gelb (status-warning)
// - ood -> rot (status-ood)
function updateAiToggleStyle(caseObj, isAiRevealed) {
  if (!caseObj || !btnAiToggle || !aiToggleLabel) return;

  const isOod = caseObj.ai_assessment?.ood?.is_ood === true;

  // Clear all dynamic status classes
  btnAiToggle.classList.remove("active", "status-normal", "status-warning", "status-ood");

  if (isOod) {
    btnAiToggle.classList.add("status-ood");
    aiToggleLabel.textContent = "AI Disabled (OOD)";
    return;
  }

  if (!isAiRevealed) {
    aiToggleLabel.textContent = "AI Forecast";
    return;
  }

  // AI Forecast is revealed and in-distribution:
  btnAiToggle.classList.add("active");
  aiToggleLabel.textContent = "AI Forecast On";

  const margin = caseObj.ai_assessment?.margin_delta ?? 1.0;
  const isCloseRivalry = margin < CONFIDENCE_MARGIN_THRESHOLD;
  const isSliceWarning = !!caseObj.ai_assessment?.slice_competence?.warning;

  if (isCloseRivalry || isSliceWarning) {
    // Close rivalry OR SUBGROUP SENSITIVITY -> gelb
    btnAiToggle.classList.add("status-warning");
  } else {
    // nichts -> grün
    btnAiToggle.classList.add("status-normal");
  }
}

// SUBGROUP SENSITIVITY ALERT (ONLY VISIBLE ONCE AI FORECAST IS REVEALED)
function updateSubgroupAlert(currentCase, isAiRevealed) {
  const isSliceWarning = !!currentCase.ai_assessment?.slice_competence?.warning;
  if (isSliceWarning && isAiRevealed) {
    sliceWarningBox.classList.remove("hidden");
    if (sliceSiteVal) {
      sliceSiteVal.textContent = currentCase.patient.anatomical_site || "Lower Extremity";
    }
  } else {
    sliceWarningBox.classList.add("hidden");
  }
}

// TOGGLE AI FORECAST (KEY S)
function toggleAiForecast() {
  const state = caseStates[currentCaseIndex];
  const currentCase = casesData[currentCaseIndex];
  const isOod = currentCase.ai_assessment?.ood?.is_ood === true;

  if (isOod) {
    showNotification("AI forecast is disabled for Out-of-Distribution cases.");
    return;
  }

  state.aiRevealed = !state.aiRevealed;

  if (state.aiRevealed) {
    state.aiRevealedAt = performance.now();
    applyAiForecast(currentCase);
    updateSubgroupAlert(currentCase, true);
    updateTripleAvatar(currentCase, true);
  } else {
    hideAiForecast();
    updateSubgroupAlert(currentCase, false);
    updateTripleAvatar(currentCase, false);
  }
  updateAiToggleStyle(currentCase, state.aiRevealed);
}

// HIDE AI FORECAST (CLEARS BARS, TEXT, AND STATUS)
function hideAiForecast() {
  const currentCase = casesData[currentCaseIndex];
  DIAG_CLASSES.forEach((c) => {
    const btn = diagButtonsGroup.querySelector(`[data-class="${c.id}"]`);
    const fillEl = document.getElementById(`fill-${c.id}`);
    const pctEl = document.getElementById(`pct-${c.id}`);

    if (fillEl) fillEl.style.width = "0%";
    if (pctEl) {
      pctEl.textContent = "";
      pctEl.classList.remove("revealed");
    }
    btn?.classList.remove("top-prediction");
  });

  aiStatusCard.classList.add("hidden");

  if (currentCase) {
    updateSubgroupAlert(currentCase, false);
    updateTripleAvatar(currentCase, false);
  }
}

// APPLY AI FORECAST
// PHILOSOPHY: If confident, color winner red with zero noise/clutter.
// If uncertain (close rivalry), mark NO winner and prompt: "Close rivalry: compare to rival -> [W]"
function applyAiForecast(caseObj) {
  const isOod = caseObj.ai_assessment?.ood?.is_ood === true;
  const probs = caseObj.ai_assessment?.calibrated_probabilities;
  const topKey = caseObj.ai_assessment?.predicted_class?.short_name;
  const margin = caseObj.ai_assessment?.margin_delta || 0;

  if (isOod || !probs) {
    hideAiForecast();
    return;
  }

  // Determine if model is confident or uncertain (close rivalry)
  const isConfident = margin >= CONFIDENCE_MARGIN_THRESHOLD;

  DIAG_CLASSES.forEach((c) => {
    const btn = diagButtonsGroup.querySelector(`[data-class="${c.id}"]`);
    const fillEl = document.getElementById(`fill-${c.id}`);
    const pctEl = document.getElementById(`pct-${c.id}`);

    if (c.id === "unclear") return;

    const probVal = probs[c.id] || 0;
    const pct = (probVal * 100).toFixed(1);

    if (fillEl) {
      // Anchored at right: 0, expanding to the left across the button body
      fillEl.style.width = `${Math.min(100, Math.max(0, probVal * 100))}%`;
    }

    if (pctEl) {
      pctEl.textContent = `${pct}%`;
      pctEl.classList.add("revealed");
    }

    // Only mark the winner red if the model is confident!
    // If the model is uncertain, no winner is marked red!
    if (isConfident && c.id === topKey) {
      btn?.classList.add("top-prediction");
    } else {
      btn?.classList.remove("top-prediction");
    }
  });

  // Zero noise when everything is good: hide status card completely if confident!
  // Only display alert when candidates are close to each other:
  if (isConfident) {
    aiStatusCard.classList.add("hidden");
  } else {
    aiStatusCard.classList.remove("hidden");
  }
}

function handleDiagnosisInput(classId) {
  const state = caseStates[currentCaseIndex];

  if (state.isSignedOff) {
    showNotification("Case is already signed off and locked.");
    return;
  }

  // First touch records initial hypothesis and latency
  if (!state.initialChoice) {
    state.initialChoice = classId;
    state.firstChoiceLatency = ((performance.now() - state.startTime) / 1000).toFixed(2);
    // Reveal AI forecast automatically on first diagnostic decision
    if (!state.aiRevealed) {
      toggleAiForecast();
    }
  }

  state.selectedClass = classId;
  renderButtonsState(state);
  updateSignOffPanel(state);

  // Update twins column highlighting immediately
  updateTwinsColumnHighlighting(classId);
}

function renderButtonsState(state) {
  const buttons = diagButtonsGroup.querySelectorAll(".diag-btn");
  buttons.forEach((btn) => {
    const classId = btn.dataset.class;
    if (classId === state.selectedClass) {
      btn.classList.add("selected");
    } else {
      btn.classList.remove("selected");
    }
    btn.disabled = state.isSignedOff;
  });
}

function handleSignOff() {
  const state = caseStates[currentCaseIndex];

  // Clinical Safety Gate: Sign-off only permitted in Main Menu
  if (isTwinsViewActive) {
    toggleTwinsView(false);
    showNotification("Zurück im Hauptmenü. [Enter] zum Sign-off drücken.");
    return;
  }

  // If already signed off, pressing Enter navigates to the next case
  if (state.isSignedOff) {
    if (currentCaseIndex < casesData.length - 1) {
      loadCase(currentCaseIndex + 1);
    }
    return;
  }

  // Require a selected diagnosis
  if (!state.selectedClass) {
    selectedDiagBadge.classList.add("empty");
    selectedDiagBadge.style.borderColor = "#ef4444";
    selectedDiagBadge.textContent = "Please select a diagnosis first (1-8)";
    setTimeout(() => {
      selectedDiagBadge.textContent = "No diagnosis selected yet (Press 1-8)";
      selectedDiagBadge.style.borderColor = "";
    }, 1500);
    return;
  }

  // Finalize case
  state.isSignedOff = true;
  state.finalChoice = state.selectedClass;

  clearInterval(dwellTimerInterval);

  caseLockStatusEl.className = "hud-lock-badge locked";
  lockStatusTextEl.textContent = "SIGNED OFF & LOCKED";

  renderButtonsState(state);
  updateSignOffPanel(state);

  if (currentCaseIndex < casesData.length - 1) {
    btnSignOff.innerHTML = `<span>Case Signed Off</span> <span class="hud-kbd">D for Next</span>`;
  } else {
    btnSignOff.innerHTML = `<span>All Cases Completed</span>`;
  }
}

function updateSignOffPanel(state) {
  if (state.selectedClass) {
    const diag = DIAG_CLASSES.find((c) => c.id === state.selectedClass);
    selectedDiagBadge.className = "selected-diag-badge";
    selectedDiagBadge.innerHTML = `<strong>[${diag.key}] ${diag.name}</strong>`;
  } else {
    selectedDiagBadge.className = "selected-diag-badge empty";
    selectedDiagBadge.textContent = "No diagnosis selected yet (Press 1-8)";
  }

  if (state.isSignedOff) {
    btnSignOff.classList.add("signed-off");
    btnSignOff.disabled = false;
  } else {
    btnSignOff.classList.remove("signed-off");
    btnSignOff.innerHTML = `<span class="btn-main-text">Sign-off Case</span> <span class="hud-kbd enter-kbd">ENTER</span>`;
  }
}

function updateDwellDisplay(sec) {
  const formatted = sec.toFixed(1);
  dwellTimerBadge.textContent = `${formatted}s`;
}

// HEATMAP STEPPING IN 50% INCREMENTS (KEY Q / E)
function adjustHeatmapOpacity(delta) {
  const state = caseStates[currentCaseIndex];
  const newVal = Math.min(100, Math.max(0, state.heatmapOpacity + delta));
  setHeatmapOpacity(newVal);
}

function setHeatmapOpacity(val) {
  const state = caseStates[currentCaseIndex];
  state.heatmapOpacity = val;
  heatmapOpacitySlider.value = val;
  heatmapImageEl.style.opacity = val / 100;
  opacityValEl.textContent = `${val}%`;
}

// TOGGLE REFERENCE TWINS (KEY W) - UNROLLS LEFT & RIGHT 3+3 PANELS
function toggleTwinsView(forceState = null) {
  if (forceState !== null) {
    isTwinsViewActive = forceState;
  } else {
    isTwinsViewActive = !isTwinsViewActive;
  }
  updateTwinsViewDisplay();
}

function updateTwinsViewDisplay() {
  if (isTwinsViewActive) {
    if (leftPanelNormal) leftPanelNormal.classList.add("hidden");
    if (leftPanelTwins) leftPanelTwins.classList.remove("hidden");
    if (rightPanelNormal) rightPanelNormal.classList.add("hidden");
    if (rightPanelTwins) rightPanelTwins.classList.remove("hidden");

    if (btnHoldWTwin) {
      btnHoldWTwin.classList.add("active");
      btnHoldWTwin.innerHTML = '<span class="hud-kbd">W</span> Twins: Active';
    }

    const currentCase = casesData[currentCaseIndex];
    renderTwinsData(currentCase);
    const state = caseStates[currentCaseIndex];
    updateTwinsColumnHighlighting(state ? state.selectedClass : null);
  } else {
    if (leftPanelNormal) leftPanelNormal.classList.remove("hidden");
    if (leftPanelTwins) leftPanelTwins.classList.add("hidden");
    if (rightPanelNormal) rightPanelNormal.classList.remove("hidden");
    if (rightPanelTwins) rightPanelTwins.classList.add("hidden");

    if (btnHoldWTwin) {
      btnHoldWTwin.classList.remove("active");
      btnHoldWTwin.innerHTML = '<span class="hud-kbd">W</span> Reference Twins';
    }
  }
}

// RENDER 6 REFERENCE TWINS (3 PREDICTED LEFT, 3 RIVAL RIGHT)
function renderTwinsData(caseObj) {
  if (!caseObj) return;
  const p = caseObj.precedents;

  const noTwinsHtml = `
    <div style="text-align: center; color: var(--text-muted); padding: 30px 10px; font-family: var(--font-mono); font-size: 11px;">
      No reference twins available (Out-of-Distribution Case).
    </div>
  `;

  if (!p || caseObj.ai_assessment?.ood?.is_ood) {
    currentLeftClass = null;
    currentRightClass = null;
    if (leftTwinsCardsRow) leftTwinsCardsRow.innerHTML = noTwinsHtml;
    if (rightTwinsCardsRow) rightTwinsCardsRow.innerHTML = noTwinsHtml;
    updateTwinsColumnHighlighting(null);
    return;
  }

  const predTwins = p.contrastive?.predicted_twins || (p.top_match ? [p.top_match] : []);
  const rivalTwins = p.contrastive?.rival_twins || (p.rival_match ? [p.rival_match] : []);

  const predShort = caseObj.ai_assessment?.predicted_class?.short_name || "mel";
  const predObj = DIAG_CLASSES.find(d => d.id === predShort);
  const predName = predObj ? predObj.name : predShort.toUpperCase();
  const predKey = predObj ? predObj.key : "1";

  const rivShort = p.rival_match?.dx_short || (caseObj.ai_assessment?.rival_class || "nv");
  const rivObj = DIAG_CLASSES.find(d => d.id === rivShort);
  const rivName = rivObj ? rivObj.name : rivShort.toUpperCase();
  const rivKey = rivObj ? rivObj.key : "2";

  currentLeftClass = predShort;
  currentRightClass = rivShort;

  // Update Left Header (Top Match - 2 Clean Lines)
  const leftTwinsKey = document.getElementById("leftTwinsKey");
  const leftTwinsLabel = document.getElementById("leftTwinsLabel");
  if (leftTwinsKey) leftTwinsKey.textContent = predKey;
  if (leftTwinsLabel) leftTwinsLabel.textContent = predName;

  // Update Right Header (Rival Match - 2 Clean Lines)
  const rightTwinsKey = document.getElementById("rightTwinsKey");
  const rightTwinsLabel = document.getElementById("rightTwinsLabel");
  if (rightTwinsKey) rightTwinsKey.textContent = rivKey;
  if (rightTwinsLabel) rightTwinsLabel.textContent = rivName;

  function renderCardList(twinsList) {
    if (!twinsList || twinsList.length === 0) {
      return `<div style="text-align: center; color: var(--text-muted); padding: 20px 10px; font-family: var(--font-mono); font-size: 11px;">No reference cases found</div>`;
    }
    return twinsList.map((twin, idx) => {
      return `
        <div class="twin-card-v2" title="Reference Twin #${idx + 1} (${twin.similarity_pct}% Similarity)">
          <div class="twin-card-top-row">
            <span class="twin-case-num">Case #${idx + 1}</span>
            <span class="twin-sim-badge">${twin.similarity_pct}% Similarity</span>
          </div>
          <div class="twin-img-container">
            <img src="${twin.image}" alt="Reference Specimen" loading="lazy" />
          </div>
        </div>
      `;
    }).join("");
  }

  if (leftTwinsCardsRow) leftTwinsCardsRow.innerHTML = renderCardList(predTwins);
  if (rightTwinsCardsRow) rightTwinsCardsRow.innerHTML = renderCardList(rivalTwins);

  const state = caseStates[currentCaseIndex];
  updateTwinsColumnHighlighting(state ? state.selectedClass : null);
}

// HIGHLIGHTS THE ENTIRE COLUMN OF THE CHOSEN DIAGNOSIS
function updateTwinsColumnHighlighting(selectedClass) {
  if (!leftPanelTwins || !rightPanelTwins) return;

  const leftBadge = document.getElementById("leftTwinsSelectedBadge");
  const rightBadge = document.getElementById("rightTwinsSelectedBadge");

  if (!selectedClass) {
    leftPanelTwins.classList.remove("column-selected", "column-dimmed");
    rightPanelTwins.classList.remove("column-selected", "column-dimmed");
    if (leftBadge) leftBadge.classList.add("hidden");
    if (rightBadge) rightBadge.classList.add("hidden");
    return;
  }

  if (selectedClass === currentLeftClass) {
    leftPanelTwins.classList.add("column-selected");
    leftPanelTwins.classList.remove("column-dimmed");
    if (leftBadge) leftBadge.classList.remove("hidden");

    rightPanelTwins.classList.remove("column-selected");
    rightPanelTwins.classList.add("column-dimmed");
    if (rightBadge) rightBadge.classList.add("hidden");
  } else if (selectedClass === currentRightClass) {
    rightPanelTwins.classList.add("column-selected");
    rightPanelTwins.classList.remove("column-dimmed");
    if (rightBadge) rightBadge.classList.remove("hidden");

    leftPanelTwins.classList.remove("column-selected");
    leftPanelTwins.classList.add("column-dimmed");
    if (leftBadge) leftBadge.classList.add("hidden");
  } else {
    // Other diagnosis chosen (e.g. 3 BCC)
    leftPanelTwins.classList.remove("column-selected");
    leftPanelTwins.classList.add("column-dimmed");
    rightPanelTwins.classList.remove("column-selected");
    rightPanelTwins.classList.add("column-dimmed");
    if (leftBadge) leftBadge.classList.add("hidden");
    if (rightBadge) rightBadge.classList.add("hidden");
  }
}

// UPDATE TRIPLE-PROJECTION ANATOMICAL AVATAR PINS
function updateTripleAvatar(caseObj, isAiRevealed) {
  if (!caseObj || !caseObj.patient) return;
  const siteKey = (caseObj.patient.anatomical_site || "").toLowerCase().trim();
  const coords = PROJECTION_COORDS[siteKey] || {
    front: null,
    back:  { x: 50, y: 28, visible: true },
    side:  { x: 80, y: 28, visible: true }
  };

  const isSliceAlert = isAiRevealed && !!caseObj.ai_assessment?.slice_competence?.warning;

  function applyPin(pinEl, cfg) {
    if (!pinEl) return;
    if (!cfg || !cfg.visible) {
      pinEl.style.display = "none";
      return;
    }
    pinEl.style.display = "flex";
    pinEl.style.left = `${cfg.x}%`;
    pinEl.style.top = `${cfg.y}%`;

    if (isSliceAlert) {
      pinEl.classList.add("alert");
    } else {
      pinEl.classList.remove("alert");
    }
  }

  applyPin(pinFront, coords.front);
  applyPin(pinBack, coords.back);
  applyPin(pinSide, coords.side);
}

// RESET ALL CASES (GLOBAL WORKSTATION RESET)
function resetAllCases() {
  resetAllCaseStates();
  loadCase(0);
  showNotification("Workstation reset: All cases cleared.");
}

function showNotification(msg) {
  if (!selectedDiagBadge) return;
  selectedDiagBadge.textContent = msg;
  setTimeout(() => {
    updateSignOffPanel(caseStates[currentCaseIndex]);
  }, 1600);
}

// Boot workstation on DOM ready
if (document.readyState === "complete" || document.readyState === "interactive") {
  initDashboard();
} else {
  window.addEventListener("DOMContentLoaded", initDashboard);
}
