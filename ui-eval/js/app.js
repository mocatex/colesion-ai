
let currentLeftClass = null;
let currentRightClass = null;

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
    leftPanelTwins.classList.remove("column-selected");
    leftPanelTwins.classList.add("column-dimmed");
    rightPanelTwins.classList.remove("column-selected");
    rightPanelTwins.classList.add("column-dimmed");
    if (leftBadge) leftBadge.classList.add("hidden");
    if (rightBadge) rightBadge.classList.add("hidden");
  }
}



// COLESION-AI // CLINICAL EVALUATION WORKPLACE (ui-eval)
// 4-Stage Decision Flow (Solo -> AI Reveal -> Deliberation -> Locked)
// Granular Telemetry & Keystroke Logger, Educational Onboarding Tutorial & Study

let studyPayload = null;
let tutorialCases = [];
let studyCases = [];
let globalMetadata = {};

let activePhase = "tutorial"; // "tutorial" | "study" | "completed"
let currentCaseIndex = 0;
let currentStage = 0; // 0: Solo, 1: AI Reveal, 2: Deliberation, 3: Locked
let participantId = "Dr_Muster_01";

// Live case runtime state
let currentCase = null;
let caseState = null;
let activeCasesList = [];

// Session Telemetry Store
let keystrokeLog = [];
let lastKeyTimestamp = null;
let caseResults = []; // Saved trial records
let sessionStartTime = null;

// 7 HAM10000 Diagnostic Classes + 1 Inconclusive Choice
const DIAG_CLASSES = [
  { key: "1", id: "mel", name: "Melanoma", isMalignant: true },
  { key: "2", id: "nv", name: "Melanocytic Nevus", isMalignant: false },
  { key: "3", id: "bcc", name: "Basal Cell Carcinoma", isMalignant: true },
  { key: "4", id: "akiec", name: "Actinic Keratosis", isMalignant: true },
  { key: "5", id: "bkl", name: "Benign Keratosis", isMalignant: false },
  { key: "6", id: "df", name: "Dermatofibroma", isMalignant: false },
  { key: "7", id: "vasc", name: "Vascular Lesion", isMalignant: false }
];

// Triple-Projection Anatomical Coordinates
const PROJECTION_COORDS = {
  "trunk": { front: { x: 50, y: 30, visible: true }, back: { x: 50, y: 30, visible: true }, side: { x: 50, y: 30, visible: true } },
  "upper extremity": { front: { x: 18, y: 33, visible: true }, back: { x: 18, y: 33, visible: true }, side: { x: 50, y: 33, visible: true } },
  "lower extremity": { front: { x: 38, y: 70, visible: true }, back: { x: 38, y: 70, visible: true }, side: { x: 62, y: 70, visible: true } },
  "head and neck": { front: { x: 50, y: 8, visible: true }, back: { x: 50, y: 8, visible: true }, side: { x: 49, y: 8, visible: true } },
  "head / neck": { front: { x: 50, y: 8, visible: true }, back: { x: 50, y: 8, visible: true }, side: { x: 49, y: 8, visible: true } },
  "face / scalp": { front: { x: 50, y: 8, visible: true }, back: { x: 50, y: 8, visible: true }, side: { x: 49, y: 8, visible: true } },
  "face": { front: { x: 50, y: 8, visible: true }, back: { x: 50, y: 8, visible: true }, side: { x: 49, y: 8, visible: true } },
  "anogenital region": { front: { x: 50, y: 49, visible: true }, back: { x: 50, y: 49, visible: true }, side: { x: 57, y: 49, visible: true } },
  "back": { front: null, back: { x: 50, y: 28, visible: true }, side: { x: 45, y: 28, visible: true } },
  "abdomen": { front: { x: 50, y: 38, visible: true }, back: null, side: { x: 35, y: 38, visible: true } },
  "chest": { front: { x: 50, y: 24, visible: true }, back: null, side: { x: 35, y: 24, visible: true } }
};

// Dwell timer
let dwellTimerInterval = null;
let isWHeld = false;
let twinOpenTimestamp = null;
let heatmapOpenTimestamp = null;

// DOM Elements
const prevCaseBtn = document.getElementById("prevCaseBtn");
const nextCaseBtn = document.getElementById("nextCaseBtn");
const caseCounterEl = document.getElementById("caseCounter");
const specimenIdEl = document.getElementById("specimenId");
const pSpecimenIdEl = document.getElementById("pSpecimenId");
const studyPhaseBadgeEl = document.getElementById("studyPhaseBadge");
const participantBadgeEl = document.getElementById("participantBadge");
const topMlAccuracyDisplayEl = document.getElementById("topMlAccuracyDisplay");
const caseLockStatusEl = document.getElementById("caseLockStatus");
const lockStatusTextEl = document.getElementById("lockStatusText");

// Pipeline pills
const pillStage0 = document.getElementById("pillStage0");
const pillStage1 = document.getElementById("pillStage1");
const pillStage2 = document.getElementById("pillStage2");
const pillStage3 = document.getElementById("pillStage3");

// Guidance Banner
const stageGuidanceTag = document.getElementById("stageGuidanceTag");
const stageGuidanceText = document.getElementById("stageGuidanceText");
const stageHotkeyHint = document.getElementById("stageHotkeyHint");

// Left panel
const aiConfidenceBadge = document.getElementById("aiConfidenceBadge");
const aiForecastBadgeText = document.getElementById("aiForecastBadgeText");
const aiStage0Notice = document.getElementById("aiStage0Notice");
const diagButtonsGroup = document.getElementById("diagButtonsGroup");
const deliberationCard = document.getElementById("deliberationCard");
const delibShiftBadge = document.getElementById("delibShiftBadge");
const delibHumanInitial = document.getElementById("delibHumanInitial");
const delibAiForecast = document.getElementById("delibAiForecast");
const aiStatusCard = document.getElementById("aiStatusCard");
const btnCompareRival = document.getElementById("btnCompareRival");

// Center panel / Viewport
const baseImageEl = document.getElementById("baseImage");
const heatmapImageEl = document.getElementById("heatmapImage");
const heatmapSliderDock = document.getElementById("heatmapSliderDock");
const heatmapOpacitySlider = document.getElementById("heatmapOpacity");
const opacityValEl = document.getElementById("opacityVal");
const btnHeatmapDown = document.getElementById("btnHeatmapDown");
const btnHeatmapUp = document.getElementById("btnHeatmapUp");
const btnHoldWTwin = document.getElementById("btnHoldWTwin");

// Right panel
const metaSexEl = document.getElementById("metaSex");
const metaAgeEl = document.getElementById("metaAge");
const metaSiteEl = document.getElementById("metaSite");
const pinFront = document.getElementById("pinFront");
const pinBack = document.getElementById("pinBack");
const pinSide = document.getElementById("pinSide");

const sliceWarningBox = document.getElementById("sliceWarningBox");
const sliceSiteVal = document.getElementById("sliceSiteVal");
const sliceNoteVal = document.getElementById("sliceNoteVal");
const oodAlertBox = document.getElementById("oodAlertBox");

const dwellTimerBadge = document.getElementById("dwellTimerBadge");
const selectedDiagBadge = document.getElementById("selectedDiagBadge");
const btnSignOff = document.getElementById("btnSignOff");
const btnSignOffText = document.getElementById("btnSignOffText");
const microAttrBox = document.getElementById("microAttrBox");
const btnNextAfterAttr = document.getElementById("btnNextAfterAttr");

// Twins Panels
const leftPanelNormal = document.getElementById("leftPanelNormal");
const leftPanelTwins = document.getElementById("leftPanelTwins");
const leftTwinsCardsRow = document.getElementById("leftTwinsCardsRow");
const btnCloseTwinsLeft = document.getElementById("btnCloseTwinsLeft");
const rightPanelNormal = document.getElementById("rightPanelNormal");
const rightPanelTwins = document.getElementById("rightPanelTwins");
const rightTwinsCardsRow = document.getElementById("rightTwinsCardsRow");
const btnCloseTwinsRight = document.getElementById("btnCloseTwinsRight");

// Modals
const modalWelcome = document.getElementById("modalWelcome");
const inputParticipantId = document.getElementById("inputParticipantId");
const btnStartTutorial = document.getElementById("btnStartTutorial");

const modalTutorialDebrief = document.getElementById("modalTutorialDebrief");
const debriefModalTitle = document.getElementById("debriefModalTitle");
const debriefGroundTruth = document.getElementById("debriefGroundTruth");
const debriefAiSuggestion = document.getElementById("debriefAiSuggestion");
const debriefLessonText = document.getElementById("debriefLessonText");
const debriefYourInitial = document.getElementById("debriefYourInitial");
const debriefYourFinal = document.getElementById("debriefYourFinal");
const btnContinueDebrief = document.getElementById("btnContinueDebrief");

const modalStudyComplete = document.getElementById("modalStudyComplete");
const kpiInitialAcc = document.getElementById("kpiInitialAcc");
const kpiFinalAcc = document.getElementById("kpiFinalAcc");
const kpiGainAcc = document.getElementById("kpiGainAcc");
const kpiSynergyRatio = document.getElementById("kpiSynergyRatio");
const kpiDecisionShifts = document.getElementById("kpiDecisionShifts");
const kpiAppropriateReliance = document.getElementById("kpiAppropriateReliance");
const btnDownloadJson = document.getElementById("btnDownloadJson");
const btnRestartStudy = document.getElementById("btnRestartStudy");

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  fetchStudyData();
});

async function fetchStudyData() {
  try {
    const res = await fetch("/api/study/cases");
    if (!res.ok) throw new Error("Failed to load evaluation study cases");
    studyPayload = await res.json();
    tutorialCases = studyPayload.tutorial_cases || [];
    studyCases = studyPayload.study_cases || [];
    globalMetadata = studyPayload.metadata || {};

    if (topMlAccuracyDisplayEl && globalMetadata.model_balanced_acc) {
      topMlAccuracyDisplayEl.textContent = `ML Accuracy ${(globalMetadata.model_balanced_acc * 100).toFixed(1)}%`;
    }

    logTelemetry("STUDY_DATA_LOADED", {
      total_tutorial: tutorialCases.length,
      total_study: studyCases.length
    });

    // Check if participant name already given
    if (!participantId || participantId === "Dr_Muster_01") {
      openWelcomeModal();
    } else {
      startTutorialPhase();
    }
  } catch (err) {
    console.error("Initialization error:", err);
    alert("Could not load study cases from server. Ensure ui-eval/server.py is running.");
  }
}

function openWelcomeModal() {
  if (modalWelcome) {
    modalWelcome.classList.remove("hidden");
    if (inputParticipantId) {
      inputParticipantId.value = participantId;
      inputParticipantId.focus();
    }
  }
}

function startTutorialPhase() {
  activePhase = "tutorial";
  activeCasesList = tutorialCases;
  currentCaseIndex = 0;
  sessionStartTime = Date.now();
  if (studyPhaseBadgeEl) {
    studyPhaseBadgeEl.className = "study-phase-badge tutorial";
    studyPhaseBadgeEl.textContent = `TUTORIAL CASE 1/${tutorialCases.length}`;
  }
  loadCase(0);
}

function startMainStudyPhase() {
  activePhase = "study";
  activeCasesList = studyCases;
  currentCaseIndex = 0;
  if (studyPhaseBadgeEl) {
    studyPhaseBadgeEl.className = "study-phase-badge study";
    studyPhaseBadgeEl.textContent = `STUDY CASE 1/${studyCases.length}`;
  }
  loadCase(0);
}

// --- CASE LOADING & STATE MANAGEMENT ---
function loadCase(index) {
  if (!activeCasesList || activeCasesList.length === 0) return;
  if (index < 0 || index >= activeCasesList.length) return;

  currentCaseIndex = index;
  currentCase = activeCasesList[index];

  // Initialize isolated case runtime state
  caseState = {
    caseObj: currentCase,
    phase: activePhase,
    caseIndex: currentCaseIndex,
    totalCasesInPhase: activeCasesList.length,
    stage: 0,
    initialDiagnosis: null,
    finalDiagnosis: null,
    decisionShift: false,
    microAttribution: null,
    t_stage0_start: Date.now(),
    t_init_ms: 0,
    t_stage1_start: null,
    t_stage2_start: null,
    t_delib_ms: 0,
    t_lock_ms: null,
    dwellHeatmapMs: 0,
    dwellTwinsMs: 0,
    heatmapUsed: false,
    twinsUsed: false,
    isLocked: false
  };

  // Close twins if open
  closeTwinsView();

  // Reset viewport & heatmap
  setHeatmapOpacity(0);

  // Update Top HUD Counter & Specimen ID
  const caseNum = currentCaseIndex + 1;
  const totalNum = activeCasesList.length;
  if (caseCounterEl) {
    caseCounterEl.textContent = `Case ${caseNum} of ${totalNum}`;
  }
  if (studyPhaseBadgeEl) {
    if (activePhase === "tutorial") {
      studyPhaseBadgeEl.className = "study-phase-badge tutorial";
      studyPhaseBadgeEl.textContent = `TUTORIAL • Case ${caseNum} of ${totalNum}`;
    } else {
      studyPhaseBadgeEl.className = "study-phase-badge study";
      studyPhaseBadgeEl.textContent = `STUDY • Case ${caseNum} of ${totalNum}`;
    }
  }
  if (specimenIdEl) specimenIdEl.textContent = currentCase.image_id;
  if (pSpecimenIdEl) pSpecimenIdEl.textContent = currentCase.image_id;
  if (participantBadgeEl) participantBadgeEl.textContent = `Participant: ${participantId}`;

  // Stream Base Dermoscopy Image & Heatmap
  if (baseImageEl) baseImageEl.src = currentCase.image;
  if (heatmapImageEl) heatmapImageEl.src = currentCase.heatmap || "";

  // Populate Clinical Metadata
  populatePatientMetadata(currentCase.patient);

  // Pre-render Twins
  renderTwinsData(currentCase);

  // Start Live Dwell Timer
  startDwellTimer();

  // Enter STAGE 0: SOLO
  transitionToStage(0);

  logTelemetry("CASE_LOADED", {
    phase: activePhase,
    case_index: currentCaseIndex,
    image_id: currentCase.image_id,
    ground_truth: currentCase.ground_truth.short_name,
    ai_predicted: currentCase.ai_assessment.predicted_class.short_name
  });
}

function populatePatientMetadata(patient) {
  if (!patient) return;
  if (metaSexEl) metaSexEl.textContent = patient.sex || "--";
  if (metaAgeEl) metaAgeEl.textContent = patient.age !== undefined && patient.age !== null ? patient.age : "--";
  if (metaSiteEl) metaSiteEl.textContent = patient.anatomical_site || "Unspecified";

  // Position Triple-Projection Avatar Pin
  const loc = (patient.anatomical_site || "").toLowerCase().trim();
  const isUnspecified = !loc || loc === "unspecified" || loc === "unknown" || loc === "--" || loc === "null";
  const coords = isUnspecified ? null : (PROJECTION_COORDS[loc] || null);

  function placePin(pinEl, data) {
    if (!pinEl) return;
    if (data && data.visible) {
      pinEl.style.display = "flex";
      pinEl.style.left = `${data.x}%`;
      pinEl.style.top = `${data.y}%`;
    } else {
      pinEl.style.display = "none";
    }
  }

  placePin(pinFront, coords ? coords.front : null);
  placePin(pinBack, coords ? coords.back : null);
  placePin(pinSide, coords ? coords.side : null);
}

// --- 4-STAGE PIPELINE TRANSITION LOGIC ---
function transitionToStage(newStage) {
  currentStage = newStage;
  caseState.stage = newStage;

  // 1. Update Pipeline Pills in Header
  [pillStage0, pillStage1, pillStage2, pillStage3].forEach((p, idx) => {
    if (!p) return;
    if (idx === newStage) {
      p.className = `stage-step-pill pill-${idx} active`;
    } else {
      p.className = `stage-step-pill pill-${idx}`;
    }
  });

  // 2. Update Lock Badge
  if (caseLockStatusEl && lockStatusTextEl) {
    if (newStage === 3) {
      caseLockStatusEl.className = "hud-lock-badge locked";
      lockStatusTextEl.textContent = "SIGNED OFF & LOCKED";
    } else {
      caseLockStatusEl.className = "hud-lock-badge open";
      const names = ["STAGE 0: SOLO", "STAGE 1: AI REVEAL", "STAGE 2: DELIBERATE", "STAGE 3: LOCKED"];
      lockStatusTextEl.textContent = names[newStage];
    }
  }

  // 3. Stage-Specific UI Display & Gating
  if (newStage === 0) {
    applyStage0UI();
  } else if (newStage === 1) {
    applyStage1UI();
  } else if (newStage === 2) {
    applyStage2UI();
  } else if (newStage === 3) {
    applyStage3UI();
  }

  logTelemetry("STAGE_TRANSITION", { stage: newStage });
}

function applyStage0UI() {
  // Guidance Banner
  if (stageGuidanceTag) stageGuidanceTag.className = "stage-tag stage-0-tag";
  if (stageGuidanceTag) stageGuidanceTag.textContent = "STAGE 0: SOLO";
  stageGuidanceText.textContent = "Examine specimen & metadata. AI forecast is blocked. Select your unassisted initial diagnosis using keys [1]–[7].";
  stageHotkeyHint.textContent = "Keys: [1]–[7] Initial Diagnosis";

  // Hide / Mask all AI indicators
  if (aiStage0Notice) aiStage0Notice.classList.remove("hidden");
  if (aiConfidenceBadge) aiConfidenceBadge.style.opacity = "0";
  if (deliberationCard) deliberationCard.classList.add("hidden");
  if (aiStatusCard) aiStatusCard.classList.add("hidden");
  if (sliceWarningBox) sliceWarningBox.classList.add("hidden");
  if (oodAlertBox) oodAlertBox.classList.add("hidden");
  if (microAttrBox) microAttrBox.classList.add("hidden");

  // Hide Heatmap Dock completely in Stage 0
  if (heatmapSliderDock) {
    heatmapSliderDock.classList.add("hidden");
    heatmapSliderDock.classList.add("ai-blocked-mask");
  }

  // Reset and show clean buttons without probability fill
  DIAG_CLASSES.forEach(d => {
    const btn = document.querySelector(`.diag-btn[data-class="${d.id}"]`);
    if (btn) {
      btn.classList.remove("selected", "ai-favored", "ai-rival");
      btn.style.pointerEvents = "auto";
      const fillEl = document.getElementById(`fill-${d.id}`);
      const pctEl = document.getElementById(`pct-${d.id}`);
      if (fillEl) fillEl.style.width = "0%";
      if (pctEl) {
        pctEl.textContent = "";
        pctEl.style.opacity = "0";
      }
    }
  });

  const signoffNormalSec = document.getElementById("signoffNormalSection");
  const stage3Sec = document.getElementById("stage3AttrSection");
  if (signoffNormalSec) signoffNormalSec.classList.remove("hidden");
  if (stage3Sec) stage3Sec.classList.add("hidden");

  if (selectedDiagBadge) {
    selectedDiagBadge.className = "selected-diag-badge empty";
    selectedDiagBadge.textContent = "Select initial diagnosis (Keys 1-7)";
  }

  if (btnSignOff) {
    btnSignOff.style.opacity = "0.4";
    btnSignOff.style.pointerEvents = "none";
    if (btnSignOffText) btnSignOffText.textContent = "Awaiting Initial Diagnosis";
  }
}

function applyStage1UI() {
  caseState.t_stage1_start = Date.now();
  caseState.t_init_ms = caseState.t_stage1_start - caseState.t_stage0_start;

  // Guidance Banner
  const predClass = currentCase.ai_assessment?.predicted_class?.full_name || "Prediction";
  const predProb = currentCase.ai_assessment?.calibrated_probabilities?.[currentCase.ai_assessment.predicted_class.short_name] || 0;
  const predPct = (predProb * 100).toFixed(0);

  if (stageGuidanceTag) stageGuidanceTag.className = "stage-tag stage-1-tag";
  if (stageGuidanceTag) stageGuidanceTag.textContent = "STAGE 1: AI REVEAL";
  stageGuidanceText.textContent = `AI Forecast Revealed: ${predClass} (${predPct}%). Inspect Heatmap [Q/E] and Precedent Twins [Hold W]. Press [SPACE] or any key to deliberate.`;
  stageHotkeyHint.textContent = "[Q/E] Heatmap | [Hold W] Twins | [SPACE] Deliberate";

  // Hide Stage 0 notice
  if (aiStage0Notice) aiStage0Notice.classList.add("hidden");

  // Reveal & Enable Heatmap Dock in Stage 1+
  if (heatmapSliderDock) {
    heatmapSliderDock.classList.remove("hidden");
    heatmapSliderDock.classList.remove("ai-blocked-mask");
  }

  // Unveil AI Probabilities and Fill Bars
  const probs = currentCase.ai_assessment?.calibrated_probabilities || {};
  const predShort = currentCase.ai_assessment?.predicted_class?.short_name;
  const rivalShort = currentCase.precedents?.rival_match?.dx_short || currentCase.ai_assessment?.rival_class;

  DIAG_CLASSES.forEach(d => {
    const btn = document.querySelector(`.diag-btn[data-class="${d.id}"]`);
    const p = probs[d.id] || 0;
    const pct = (p * 100).toFixed(1);

    if (btn) {
      if (d.id === predShort) btn.classList.add("ai-favored");
      else if (d.id === rivalShort) btn.classList.add("ai-rival");

      const fillEl = document.getElementById(`fill-${d.id}`);
      const pctEl = document.getElementById(`pct-${d.id}`);
      if (fillEl) fillEl.style.width = `${Math.min(100, p * 100)}%`;
      if (pctEl) {
        pctEl.textContent = `${pct}%`;
        pctEl.style.opacity = "1";
      }
    }
  });

  // Top AI Confidence Badge
  if (aiConfidenceBadge && aiForecastBadgeText) {
    aiConfidenceBadge.style.opacity = "1";
    aiForecastBadgeText.textContent = `AI: ${predClass} (${predPct}%)`;
  }

  // Display Subgroup Warning if active
  const sliceMeta = currentCase.ai_assessment?.slice_competence;
  if (sliceWarningBox) {
    if (sliceMeta && sliceMeta.warning) {
      sliceWarningBox.classList.remove("hidden");
      if (sliceSiteVal) sliceSiteVal.textContent = currentCase.patient.anatomical_site || "Subgroup";
      if (sliceNoteVal) sliceNoteVal.textContent = sliceMeta.message || "Reduced Historical Sensitivity";
    } else {
      sliceWarningBox.classList.add("hidden");
    }
  }

  // Display OOD Alert if active
  const oodMeta = currentCase.ai_assessment?.ood;
  if (oodAlertBox) {
    if (oodMeta && oodMeta.is_ood) {
      oodAlertBox.classList.remove("hidden");
    } else {
      oodAlertBox.classList.add("hidden");
    }
  }

  // Display Close Rivalry Alert if margin < 20%
  const margin = currentCase.ai_assessment?.margin_delta || 1.0;
  if (aiStatusCard) {
    if (margin < 0.20 && !oodMeta?.is_ood) {
      aiStatusCard.classList.remove("hidden");
    } else {
      aiStatusCard.classList.add("hidden");
    }
  }

  // Automatically transition to Stage 2 after 1.2s or upon interaction
  setTimeout(() => {
    if (currentStage === 1) {
      transitionToStage(2);
    }
  }, 1400);
}

function applyStage2UI() {
  caseState.t_stage2_start = Date.now();

  // Guidance Banner
  if (stageGuidanceTag) stageGuidanceTag.className = "stage-tag stage-2-tag";
  if (stageGuidanceTag) stageGuidanceTag.textContent = "STAGE 2: DELIBERATE";
  stageGuidanceText.textContent = "Deliberate and review evidence. Keep your initial diagnosis or change choice (Keys [1]–[7]). Press [ENTER] to lock & sign off.";
  stageHotkeyHint.textContent = "[1]–[7] Revise | [ENTER] Lock & Sign-Off";

  // Show Deliberation Comparison Card
  if (deliberationCard) {
    deliberationCard.classList.remove("hidden");
    const initObj = DIAG_CLASSES.find(d => d.id === caseState.initialDiagnosis);
    const aiObj = DIAG_CLASSES.find(d => d.id === currentCase.ai_assessment?.predicted_class?.short_name);

    if (delibHumanInitial) delibHumanInitial.textContent = initObj ? initObj.name : "--";
    if (delibAiForecast) delibAiForecast.textContent = aiObj ? aiObj.name : "--";

    updateDeliberationBadge();
  }

  // Enable Diagnosis Buttons for Revisions
  DIAG_CLASSES.forEach(d => {
    const btn = document.querySelector(`.diag-btn[data-class="${d.id}"]`);
    if (btn) {
      btn.style.pointerEvents = "auto";
      if (d.id === (caseState.finalDiagnosis || caseState.initialDiagnosis)) {
        btn.classList.add("selected");
      } else {
        btn.classList.remove("selected");
      }
    }
  });

  // Enable Sign-Off Button
  if (btnSignOff) {
    btnSignOff.style.opacity = "1";
    btnSignOff.style.pointerEvents = "auto";
    if (btnSignOffText) btnSignOffText.textContent = "Lock Final Diagnosis";
  }
}

function applyStage3UI() {
  caseState.t_lock_ms = Date.now();
  caseState.t_delib_ms = caseState.t_lock_ms - (caseState.t_stage1_start || caseState.t_stage0_start);
  caseState.isLocked = true;

  // Freeze Diagnosis Buttons
  DIAG_CLASSES.forEach(d => {
    const btn = document.querySelector(`.diag-btn[data-class="${d.id}"]`);
    if (btn) {
      btn.style.pointerEvents = "none";
    }
  });

  // Guidance Banner
  if (stageGuidanceTag) stageGuidanceTag.className = "stage-tag stage-3-tag";
  if (stageGuidanceTag) stageGuidanceTag.textContent = "STAGE 3: LOCKED";
  stageGuidanceText.textContent = "Decision locked. Press [ENTER] to proceed to the next case.";
  stageHotkeyHint.textContent = "Press [ENTER] to Proceed";

  // Streamlined Sign-Off Box: hide normal working diagnosis box, show in-situ Stage 3 attribution
  const signoffNormalSec = document.getElementById("signoffNormalSection");
  const stage3Sec = document.getElementById("stage3AttrSection");
  const lockedDiagEl = document.getElementById("lockedDiagName");

  if (signoffNormalSec) signoffNormalSec.classList.add("hidden");
  if (stage3Sec) {
    stage3Sec.classList.remove("hidden");
    const finalDiagObj = DIAG_CLASSES.find(d => d.id === caseState.finalDiagnosis);
    if (lockedDiagEl) {
      lockedDiagEl.textContent = finalDiagObj ? finalDiagObj.name : (caseState.finalDiagnosis || "--");
    }
    // Pre-select first attribution chip if none selected
    if (!caseState.microAttribution) {
      const firstChip = document.querySelector(".micro-attr-chip");
      if (firstChip) {
        document.querySelectorAll(".micro-attr-chip").forEach(c => c.classList.remove("selected"));
        firstChip.classList.add("selected");
        caseState.microAttribution = firstChip.getAttribute("data-attr") || "Clinical Intuition";
      }
    }
  }
}

function updateDeliberationBadge() {
  if (!delibShiftBadge) return;
  const currentChoice = caseState.finalDiagnosis || caseState.initialDiagnosis;
  if (currentChoice !== caseState.initialDiagnosis) {
    delibShiftBadge.className = "hud-kbd highlight-shift";
    delibShiftBadge.textContent = "DECISION SHIFT (REVISED)";
    caseState.decisionShift = true;
  } else {
    delibShiftBadge.className = "hud-kbd highlight-concordant";
    delibShiftBadge.textContent = "MAINTAINED INITIAL";
    caseState.decisionShift = false;
  }
}

// --- DIAGNOSIS SELECTION HANDLER ---
function handleDiagnosisSelection(classId) {
  if (currentStage === 3) return; // Locked

  const diagObj = DIAG_CLASSES.find(d => d.id === classId);
  if (!diagObj) return;

  if (currentStage === 0) {
    // Stage 0: Initial Solo Diagnosis
    caseState.initialDiagnosis = classId;
    caseState.finalDiagnosis = classId;

    if (selectedDiagBadge) {
      selectedDiagBadge.className = "selected-diag-badge active";
      selectedDiagBadge.textContent = diagObj.name;
    }

    logTelemetry("INITIAL_DIAGNOSIS_SUBMITTED", { diagnosis: classId });

    // Instantly transition to Stage 1 (AI Reveal)
    transitionToStage(1);
  } else if (currentStage === 1 || currentStage === 2) {
    // Deliberation Revision
    caseState.finalDiagnosis = classId;

    if (selectedDiagBadge) {
      selectedDiagBadge.className = "selected-diag-badge active";
      selectedDiagBadge.textContent = diagObj.name;
    }

    DIAG_CLASSES.forEach(d => {
      const btn = document.querySelector(`.diag-btn[data-class="${d.id}"]`);
      if (btn) {
        if (d.id === classId) btn.classList.add("selected");
        else btn.classList.remove("selected");
      }
    });

    updateDeliberationBadge();
    logTelemetry("DIAGNOSIS_REVISED", { new_diagnosis: classId });

    if (currentStage === 1) {
      transitionToStage(2);
    }
  }

  updateTwinsColumnHighlighting(classId);
}

// --- CASE SIGN-OFF & ADVANCE ---
function handleSignOffAction() {
  if (currentStage < 2) return;

  if (currentStage === 2) {
    // Lock Case & enter Stage 3
    transitionToStage(3);
    logTelemetry("CASE_SIGNED_OFF", {
      initial_dx: caseState.initialDiagnosis,
      final_dx: caseState.finalDiagnosis,
      decision_shift: caseState.decisionShift,
      t_init_ms: caseState.t_init_ms,
      t_delib_ms: caseState.t_delib_ms
    });
  } else if (currentStage === 3) {
    // Complete case and advance
    finalizeActiveCaseAndProceed();
  }
}

function finalizeActiveCaseAndProceed() {
  // Record trial result
  const record = {
    case_index: currentCaseIndex + 1,
    phase: activePhase,
    image_id: currentCase.image_id,
    ground_truth: currentCase.ground_truth.short_name,
    ai_prediction: currentCase.ai_assessment?.predicted_class?.short_name,
    ai_confidence: currentCase.ai_assessment?.calibrated_probabilities?.[currentCase.ai_assessment.predicted_class.short_name] || 0,
    ai_rival: currentCase.precedents?.rival_match?.dx_short || currentCase.ai_assessment?.rival_class,
    ai_margin: currentCase.ai_assessment?.margin_delta || 0,
    ai_ood_distance: currentCase.ai_assessment?.ood?.knn_distance || 0,
    initial_human_dx: caseState.initialDiagnosis,
    final_human_dx: caseState.finalDiagnosis,
    initial_correct: caseState.initialDiagnosis === currentCase.ground_truth.short_name,
    final_correct: caseState.finalDiagnosis === currentCase.ground_truth.short_name,
    ai_correct: currentCase.ai_assessment?.predicted_class?.short_name === currentCase.ground_truth.short_name,
    decision_shift: caseState.decisionShift,
    micro_attribution: caseState.microAttribution || "Clinical Intuition",
    t_init_ms: caseState.t_init_ms,
    t_delib_ms: caseState.t_delib_ms,
    t_total_ms: (caseState.t_init_ms || 0) + (caseState.t_delib_ms || 0),
    heatmap_used: caseState.heatmapUsed,
    twins_used: caseState.twinsUsed
  };

  caseResults.push(record);
  logTelemetry("TRIAL_RECORD_COMMITTED", record);

  // If Tutorial Phase: Show Educational Debrief Modal
  if (activePhase === "tutorial") {
    showTutorialDebriefModal(record);
    return;
  }

  // If Study Phase: Advance to next case or finish study
  if (currentCaseIndex < studyCases.length - 1) {
    loadCase(currentCaseIndex + 1);
  } else {
    finishStudySession();
  }
}

function showTutorialDebriefModal(record) {
  const meta = currentCase.study_meta || {};
  if (debriefModalTitle) debriefModalTitle.textContent = meta.title || "Tutorial Case Debrief";

  const gtObj = DIAG_CLASSES.find(d => d.id === record.ground_truth);
  const aiObj = DIAG_CLASSES.find(d => d.id === record.ai_prediction);
  const initObj = DIAG_CLASSES.find(d => d.id === record.initial_human_dx);
  const finalObj = DIAG_CLASSES.find(d => d.id === record.final_human_dx);

  if (debriefGroundTruth) debriefGroundTruth.textContent = gtObj ? gtObj.name : record.ground_truth;
  if (debriefAiSuggestion) debriefAiSuggestion.textContent = aiObj ? `${aiObj.name} (${(record.ai_confidence * 100).toFixed(0)}%)` : "--";
  if (debriefLessonText) debriefLessonText.textContent = meta.explanation || meta.learning_goal || "--";
  if (debriefYourInitial) debriefYourInitial.textContent = initObj ? initObj.name : "--";
  if (debriefYourFinal) debriefYourFinal.textContent = finalObj ? finalObj.name : "--";

  if (modalTutorialDebrief) {
    modalTutorialDebrief.classList.remove("hidden");
  }
}

function handleContinueDebrief() {
  if (modalTutorialDebrief) modalTutorialDebrief.classList.add("hidden");

  if (activePhase === "tutorial") {
    if (currentCaseIndex < tutorialCases.length - 1) {
      loadCase(currentCaseIndex + 1);
    } else {
      // Completed all 4 tutorial cases! Transition to main study!
      startMainStudyPhase();
    }
  }
}

function finishStudySession() {
  activePhase = "completed";
  stopDwellTimer();

  // Compute Summary Evaluation Metrics
  const studyTrials = caseResults.filter(r => r.phase === "study");
  const total = studyTrials.length || 1;

  const initCorr = studyTrials.filter(r => r.initial_correct).length;
  const finalCorr = studyTrials.filter(r => r.final_correct).length;
  const aiCorr = studyTrials.filter(r => r.ai_correct).length;

  const initAcc = initCorr / total;
  const finalAcc = finalCorr / total;
  const aiAcc = aiCorr / total;
  const gainAcc = finalAcc - initAcc;
  const maxSolo = Math.max(initAcc, aiAcc) || 0.001;
  const synergyRatio = finalAcc / maxSolo;
  const shiftsCount = studyTrials.filter(r => r.decision_shift).length;

  // Appropriate Reliance:
  // Trust Quota: user trusts when AI is correct
  const aiRightTrials = studyTrials.filter(r => r.ai_correct);
  const aiRightAgreed = aiRightTrials.filter(r => r.final_human_dx === r.ai_prediction).length;
  const trustQuota = aiRightTrials.length > 0 ? (aiRightAgreed / aiRightTrials.length) : 1.0;

  // Correction Quota: user overrides when AI is wrong
  const aiWrongTrials = studyTrials.filter(r => !r.ai_correct);
  const aiWrongOverridden = aiWrongTrials.filter(r => r.final_human_dx !== r.ai_prediction).length;
  const correctionQuota = aiWrongTrials.length > 0 ? (aiWrongOverridden / aiWrongTrials.length) : 1.0;

  const appropriateReliance = (trustQuota + correctionQuota) / 2.0;

  // Render KPI Dashboard
  if (kpiInitialAcc) kpiInitialAcc.textContent = `${(initAcc * 100).toFixed(1)}%`;
  if (kpiFinalAcc) kpiFinalAcc.textContent = `${(finalAcc * 100).toFixed(1)}%`;
  if (kpiGainAcc) kpiGainAcc.textContent = `${gainAcc >= 0 ? "+" : ""}${(gainAcc * 100).toFixed(1)}%`;
  if (kpiSynergyRatio) kpiSynergyRatio.textContent = synergyRatio.toFixed(3);
  if (kpiDecisionShifts) kpiDecisionShifts.textContent = `${shiftsCount} / ${total}`;
  if (kpiAppropriateReliance) kpiAppropriateReliance.textContent = `${(appropriateReliance * 100).toFixed(1)}%`;

  // Save session payload to server
  const sessionPayload = {
    participant_id: participantId,
    run_id: participantId,
    session_start_iso: new Date(sessionStartTime).toISOString(),
    session_end_iso: new Date().toISOString(),
    summary: {
      total_study_cases: total,
      initial_accuracy: initAcc,
      final_accuracy: finalAcc,
      ai_accuracy: aiAcc,
      diagnostic_gain: gainAcc,
      synergy_ratio: synergyRatio,
      decision_shifts: shiftsCount,
      appropriate_reliance_rate: appropriateReliance,
      trust_quota: trustQuota,
      correction_quota: correctionQuota
    },
    trials: caseResults,
    keystrokes_telemetry: keystrokeLog
  };

  saveTelemetryToServer(sessionPayload);

  if (modalStudyComplete) modalStudyComplete.classList.remove("hidden");
}

async function saveTelemetryToServer(payload) {
  try {
    const res = await fetch("/api/study/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    console.log("Telemetry auto-save response:", result);
  } catch (err) {
    console.error("Failed to auto-save telemetry:", err);
  }
}

function downloadTelemetryJson() {
  const sessionPayload = {
    participant_id: participantId,
    run_id: participantId,
    session_start_iso: new Date(sessionStartTime).toISOString(),
    session_end_iso: new Date().toISOString(),
    trials: caseResults,
    keystrokes_telemetry: keystrokeLog
  };

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(sessionPayload, null, 2));
  const downloadAnchor = document.createElement("a");
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `colesion_study_${participantId}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}

// --- GRANULAR KEYSTROKE & TIMING TELEMETRY ---
function logTelemetry(eventType, extraData = {}) {
  const now = Date.now();
  const delta = lastKeyTimestamp ? now - lastKeyTimestamp : 0;
  lastKeyTimestamp = now;

  keystrokeLog.push({
    event: eventType,
    timestamp_epoch_ms: now,
    delta_prev_ms: delta,
    phase: activePhase,
    case_index: currentCaseIndex,
    image_id: currentCase?.image_id || null,
    stage: currentStage,
    ...extraData
  });
}

// --- EVENT LISTENERS & HOTKEYS ---
function initEventListeners() {
  // Global Hotkey Listener
  document.addEventListener("keydown", (e) => {
    // If typing in input modal, ignore hotkeys
    if (e.target && e.target.tagName === "INPUT") return;

    logTelemetry("KEY_DOWN", { key: e.key, code: e.code });

    // In Stage 3, decisions are locked; press ENTER to proceed

    // Diagnosis Keys [1]–[7] (Stages 0, 1, 2)
    if (["1", "2", "3", "4", "5", "6", "7"].includes(e.key)) {
      const diagObj = DIAG_CLASSES.find(d => d.key === e.key);
      if (diagObj) {
        handleDiagnosisSelection(diagObj.id);
      }
      return;
    }

    // Sign-Off / Advance: Enter
    if (e.key === "Enter") {
      e.preventDefault();
      if (!modalTutorialDebrief.classList.contains("hidden")) {
        handleContinueDebrief();
      } else if (!modalWelcome.classList.contains("hidden")) {
        handleWelcomeStart();
      } else if (leftPanelTwins && !leftPanelTwins.classList.contains("hidden")) {
        // Return from twins to main deliberation view
        closeTwinsView();
      } else {
        handleSignOffAction();
      }
      return;
    }

    // Spacebar: Advance from Stage 1 to Stage 2
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      if (currentStage === 1) {
        transitionToStage(2);
      }
      return;
    }

    // Heatmap Controls [Q] / [E]
    if (e.key === "q" || e.key === "Q") {
      if (currentStage >= 1) stepHeatmapOpacity(-20);
      return;
    }
    if (e.key === "e" || e.key === "E") {
      if (currentStage >= 1) stepHeatmapOpacity(+20);
      return;
    }

    // Reference Twins [W] (Hold or Toggle)
    if (e.key === "w" || e.key === "W") {
      if (currentStage >= 1 && !isWHeld) {
        isWHeld = true;
        toggleTwinsView();
      }
      return;
    }

    // Navigation [A] / [D]
    // Case jumping keys (A/D) disabled in study evaluation
  });

  document.addEventListener("keyup", (e) => {
    if (e.key === "w" || e.key === "W") {
      isWHeld = false;
    }
  });

  // Diagnosis Buttons Clicks
  DIAG_CLASSES.forEach(d => {
    const btn = document.querySelector(`.diag-btn[data-class="${d.id}"]`);
    if (btn) {
      btn.addEventListener("click", () => handleDiagnosisSelection(d.id));
    }
  });

  // Twin Column Click Handlers: clicking column selects that hypothesis directly
  if (leftPanelTwins) {
    leftPanelTwins.addEventListener("click", (e) => {
      if (e.target.closest(".twins-close-btn")) return;
      if (currentLeftClass && currentStage < 3) {
        handleDiagnosisSelection(currentLeftClass);
      }
    });
  }
  if (rightPanelTwins) {
    rightPanelTwins.addEventListener("click", (e) => {
      if (e.target.closest(".twins-close-btn")) return;
      if (currentRightClass && currentStage < 3) {
        handleDiagnosisSelection(currentRightClass);
      }
    });
  }

  // Sign-off Button Click
  if (btnSignOff) {
    btnSignOff.addEventListener("click", handleSignOffAction);
  }
  const btnAdvanceNext = document.getElementById("btnAdvanceNext");
  if (btnAdvanceNext) {
    btnAdvanceNext.addEventListener("click", finalizeActiveCaseAndProceed);
  }

  // Micro-Attribution Chips Selection
  document.querySelectorAll(".micro-attr-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".micro-attr-chip").forEach(c => c.classList.remove("selected"));
      chip.classList.add("selected");
      caseState.microAttribution = chip.getAttribute("data-attr");
      logTelemetry("MICRO_ATTRIBUTION_SELECTED", { attribution: caseState.microAttribution });
    });
  });

  // Heatmap Controls
  if (btnHeatmapDown) btnHeatmapDown.addEventListener("click", () => stepHeatmapOpacity(-20));
  if (btnHeatmapUp) btnHeatmapUp.addEventListener("click", () => stepHeatmapOpacity(+20));
  if (heatmapOpacitySlider) {
    heatmapOpacitySlider.addEventListener("input", (e) => {
      setHeatmapOpacity(parseInt(e.target.value, 10));
    });
  }
  if (btnHoldWTwin) {
    btnHoldWTwin.addEventListener("click", toggleTwinsView);
  }

  // Twins Panel Close Buttons
  if (btnCloseTwinsLeft) btnCloseTwinsLeft.addEventListener("click", closeTwinsView);
  if (btnCloseTwinsRight) btnCloseTwinsRight.addEventListener("click", closeTwinsView);

  // Modal Buttons
  if (btnStartTutorial) {
    btnStartTutorial.addEventListener("click", handleWelcomeStart);
  }
  if (btnContinueDebrief) {
    btnContinueDebrief.addEventListener("click", handleContinueDebrief);
  }
  if (btnDownloadJson) {
    btnDownloadJson.addEventListener("click", downloadTelemetryJson);
  }
  if (btnRestartStudy) {
    btnRestartStudy.addEventListener("click", () => {
      if (modalStudyComplete) modalStudyComplete.classList.add("hidden");
      openWelcomeModal();
    });
  }

  // Navigation Prev / Next
  if (prevCaseBtn) {
    prevCaseBtn.addEventListener("click", () => {
      if (currentCaseIndex > 0 && currentStage === 3) loadCase(currentCaseIndex - 1);
    });
  }
  if (nextCaseBtn) {
    nextCaseBtn.addEventListener("click", () => {
      if (currentStage === 3) finalizeActiveCaseAndProceed();
    });
  }
}

function handleWelcomeStart() {
  const val = inputParticipantId ? inputParticipantId.value.trim() : "";
  if (val) {
    participantId = val.replace(/\s+/g, "_");
  } else {
    participantId = `Dr_Participant_${Math.floor(100 + Math.random() * 900)}`;
  }
  if (participantBadgeEl) participantBadgeEl.textContent = `Participant: ${participantId}`;
  if (modalWelcome) modalWelcome.classList.add("hidden");

  logTelemetry("STUDY_SESSION_STARTED", { participant_id: participantId });
  startTutorialPhase();
}

// --- HEATMAP OVERLAY LOGIC ---
function setHeatmapOpacity(val) {
  const clamped = Math.max(0, Math.min(100, val));
  if (heatmapImageEl) heatmapImageEl.style.opacity = clamped / 100;
  if (heatmapOpacitySlider) heatmapOpacitySlider.value = clamped;
  if (opacityValEl) opacityValEl.textContent = `${clamped}%`;

  if (clamped > 0 && caseState) {
    caseState.heatmapUsed = true;
  }
}

function stepHeatmapOpacity(delta) {
  if (currentStage === 0) return;
  const curr = parseInt(heatmapOpacitySlider.value, 10) || 0;
  setHeatmapOpacity(curr + delta);
}

// --- PRECEDENT TWINS TOGGLE ---
function toggleTwinsView() {
  if (currentStage === 0) return;
  const isHidden = leftPanelTwins.classList.contains("hidden");
  if (isHidden) {
    openTwinsView();
  } else {
    closeTwinsView();
  }
}

function openTwinsView() {
  if (leftPanelNormal) leftPanelNormal.classList.add("hidden");
  if (leftPanelTwins) leftPanelTwins.classList.remove("hidden");
  if (rightPanelNormal) rightPanelNormal.classList.add("hidden");
  if (rightPanelTwins) rightPanelTwins.classList.remove("hidden");

  if (caseState) caseState.twinsUsed = true;
  twinOpenTimestamp = Date.now();
  logTelemetry("TWINS_OPENED");
}

function closeTwinsView() {
  if (leftPanelNormal) leftPanelNormal.classList.remove("hidden");
  if (leftPanelTwins) leftPanelTwins.classList.add("hidden");
  if (rightPanelNormal) rightPanelNormal.classList.remove("hidden");
  if (rightPanelTwins) rightPanelTwins.classList.add("hidden");

  if (twinOpenTimestamp && caseState) {
    caseState.dwellTwinsMs += (Date.now() - twinOpenTimestamp);
    twinOpenTimestamp = null;
  }
  logTelemetry("TWINS_CLOSED");
}

function renderTwinsData(caseObj) {
  if (!caseObj || !caseObj.precedents) return;
  const p = caseObj.precedents;

  const predTwins = p.contrastive?.predicted_twins || (p.top_match ? [p.top_match] : []);
  const rivalTwins = p.contrastive?.rival_twins || (p.rival_match ? [p.rival_match] : []);

  currentLeftClass = caseObj.ai_assessment?.predicted_class?.short_name || (predTwins[0] ? predTwins[0].dx_short : null);
  currentRightClass = p.rival_match?.dx_short || caseObj.ai_assessment?.rival_class || (rivalTwins[0] ? rivalTwins[0].dx_short : null);

  const predObj = DIAG_CLASSES.find(d => d.id === currentLeftClass);
  const rivalObj = DIAG_CLASSES.find(d => d.id === currentRightClass);

  const leftTwinsKey = document.getElementById("leftTwinsKey");
  const leftTwinsLabel = document.getElementById("leftTwinsLabel");
  if (leftTwinsKey) leftTwinsKey.textContent = predObj ? predObj.key : "1";
  if (leftTwinsLabel) leftTwinsLabel.textContent = predObj ? predObj.name : "Top AI Match";

  const rightTwinsKey = document.getElementById("rightTwinsKey");
  const rightTwinsLabel = document.getElementById("rightTwinsLabel");
  if (rightTwinsKey) rightTwinsKey.textContent = rivalObj ? rivalObj.key : "2";
  if (rightTwinsLabel) rightTwinsLabel.textContent = rivalObj ? rivalObj.name : "Rival Match";

  function renderCardList(twinsList) {
    if (!twinsList || twinsList.length === 0) {
      return `<div style="text-align: center; color: var(--text-muted); padding: 20px 10px; font-family: var(--font-mono); font-size: 11px;">No reference twins found</div>`;
    }
    return twinsList.map((twin, idx) => `
      <div class="twin-card-v2" title="Precedent Twin #${idx + 1} (${twin.similarity_pct}% Similarity)">
        <div class="twin-card-top-row">
          <span class="twin-case-num">Case #${idx + 1}</span>
          <span class="twin-sim-badge">${twin.similarity_pct}% Similarity</span>
        </div>
        <div class="twin-img-container">
          <img src="${twin.image}" alt="Reference Specimen" loading="lazy" />
        </div>
      </div>
    `).join("");
  }

  if (leftTwinsCardsRow) leftTwinsCardsRow.innerHTML = renderCardList(predTwins);
  if (rightTwinsCardsRow) rightTwinsCardsRow.innerHTML = renderCardList(rivalTwins);

  const activeDx = caseState ? (caseState.finalDiagnosis || caseState.initialDiagnosis) : null;
  updateTwinsColumnHighlighting(activeDx);
}

// --- LIVE DWELL TIMER ---
function startDwellTimer() {
  stopDwellTimer();
  const startTime = Date.now();
  dwellTimerInterval = setInterval(() => {
    const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);
    if (dwellTimerBadge) dwellTimerBadge.textContent = `${elapsedSec}s`;
  }, 100);
}

function stopDwellTimer() {
  if (dwellTimerInterval) {
    clearInterval(dwellTimerInterval);
    dwellTimerInterval = null;
  }
}
