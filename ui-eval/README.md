# COLESION-AI // Formative Evaluation Workplace (`ui-eval`)

Evaluation Study Dashboard for the COLESION-AI bachelor thesis project.

## Quickstart

```bash
cd ui-eval
./run.sh
```

Launches the Evaluation Study workplace at **http://localhost:8001/**.

---

## Architecture & Differences from Data Explorer (`ui`)

While the original `ui` was designed as an exploratory visualizer, `ui-eval` operationalizes the **controlled experimental evaluation setup** specified in `concepts/experiment_setup_concept.md`:

1. **Participant Registration & Briefing (Modal on Start):**
   * Prompts for Participant ID / Run Name (e.g., `Dr_Muster_Run1`).
   * Displays the transparent known model baseline accuracy (~75–80% Balanced Accuracy).
   * Explains keyboard shortcuts and evaluation procedure.

2. **The 4 Decision Stages per Case:**
   * **Stage 0 (Solo Baseline):**
     * High-resolution dermoscopy image & clinical metadata only.
     * All AI displays, probability bars, heatmaps, and twin references are blocked.
     * User inputs unassisted initial diagnosis via keys `1`–`8`.
     * Records unassisted hypothesis $y_{H,\text{init}}$ and initial latency $t_{\text{init}}$.
   * **Stage 1 (AI Forecast Reveal):**
     * AI forecast and calibrated probabilities ($T = 1.29$) are unveiled.
     * Grad-CAM++ Saliency Heatmap (`Q`/`E` slider) and Precedent Twins (`Hold W`) become unlocked for inspection.
     * Subgroup slice alerts (e.g. acral / foot sensitivity warnings) and OOD alerts trigger if applicable.
   * **Stage 2 (Decision Deliberation & Shift):**
     * Side-by-side comparison between initial choice $y_{H,\text{init}}$ and AI recommendation $y_{ML}$.
     * User deliberates and can maintain or shift their decision (keys `1`–`8`).
     * User locks decision with `Enter`.
   * **Stage 3 (Case Sign-Off & In-Situ Micro-Attribution):**
     * Case is frozen (`SIGNED OFF & LOCKED`). Diagnosis buttons disabled.
     * 1-click micro-attribution: *"What had the decisive influence on your decision?"* (Clinical Intuition, AI Probability, Precedent Twins, Grad-CAM Heatmap, Subgroup/OOD Warning).
     * Press `Enter` or `D` to proceed to the next case.

3. **Onboarding Tutorial (4 Cases with Post-Case Debriefing):**
   * **T1 (`ISIC_0034533`, Nevus):** ML Correct routine concordance.
   * **T2 (`ISIC_0034584`, Melanoma):** Critical ML error (Melanoma misclassified as Nevus with 73.5% confidence) - tests protection against Automation Bias.
   * **T3 (`ISIC_0034527`, Nevus):** Close rivalry ($\Delta = 3.8\%$) - practicing twin comparison.
   * **T4 (`ISIC_0034814`, Vascular):** Out-of-Distribution ($d_{10} = 0.360 > \tau_{99}$) - practicing refusal gate and Option `8` (*Inconclusive / Biopsy referral*).
   * An educational feedback modal appears after each tutorial case explaining the ground truth pathology, why the AI behaved as it did, and the co-learning takeaway.

4. **Curated Evaluation Study (24 Cases):**
   * 12 Malignant / Premalignant (`mel`, `bcc`, `akiec`) vs. 12 Benign (`nv`, `bkl`, `df`, `vasc`, `ood`).
   * Strategically balances routine fast-paths, critical AI errors, subtle lesions where AI aids humans, and close differential rivalries.

5. **Granular Telemetry & Auto-Export:**
   * Every keystroke (`keydown`, `keyup`), timestamp, and inter-key latency ($\Delta t$) is tracked.
   * Live dwell times on heatmaps and precedent twins are measured.
   * On study completion, summary KPIs (Appropriate Reliance Rate, Accuracy Gain $\Delta_{Acc}$, Synergy Ratio $SR$, Decision Shifts, Reaction Times) are computed and automatically persisted to `ui-eval/results/colesion_study_<run_id>.json`.

---

## Key Controls

| Hotkey | Action |
| :--- | :--- |
| `1`–`8` | Select Diagnosis (`mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc`, `inconclusive`) |
| `W` | Precedent Twins (Hold / Toggle) |
| `Q` / `E` | Decrease / Increase Grad-CAM Heatmap Opacity (&plusmn;20%) |
| `Space` | Advance from Stage 1 to Deliberation Stage 2 |
| `Enter` | Lock Diagnosis / Confirm Sign-Off / Advance to Next Case |
