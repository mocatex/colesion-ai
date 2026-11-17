# COLESION-AI // Tactical HUD Data Explorer

> An interactive, high-performance diagnostic dashboard and zero-disk-duplication data explorer for clinical case analysis, model explainability, and human-AI decision calibration.

Built for Phase 3 of the CoLearning study to evaluate physician diagnostic behavior across **1,412 study cases** (`pool` + `holdout` splits) against real DINOv2 classifier outputs.

---

## Quickstart

Everything is automated. In your terminal:

```bash
cd ui
./run.sh
```

*(Alternatively from the repository root: `uv run ui/server.py` and navigate to `http://localhost:8000`).*

### What `run.sh` does automatically:
1. **Environment Setup:** Automatically copies `.env.example` to `.env` if not present.
2. **Autonomous ML Calibration:** Checks if the calibration cache exists. If missing, it runs `calibrate.py` in ~1.2 s to fit temperature scaling, compute OOD thresholds, detect subgroup drops, and index the 6-twin reference sets.
3. **Port Freeing:** Clears any stale process lingering on port `8000`.
4. **Browser Launch:** Boots the streaming server and automatically opens `http://localhost:8000/`.

---

## The Three Pillars of Decision Support

1. **Pillar 1: Temperature-Calibrated Probabilities**
   * Raw logits are scaled via validation-fitted temperature ($T = 1.2937$), reducing Expected Calibration Error (ECE) from **6.05% down to 1.49%**.
   * Probabilities are visually hidden until first physician touch or manual toggle (<kbd>S</kbd>) to prevent automation bias.

2. **Pillar 2: Real DINOv2 Grad-CAM Saliency Maps**
   * Overlaid directly on the 600×450 dermoscopic image with interactive opacity control (<kbd>Q</kbd> / <kbd>E</kbd> or mouse slider).
   * Converted on-the-fly from grayscale heatmaps to Turbo RGBA in ~2 ms via vectorized 3D LUTs.

3. **Pillar 3: 6 Reference Twins (Case-Based Reasoning)**
   * Contrastive twin retrieval across **4,559 histopathologically biopsy-verified cases** in the training corpus:
     * **Left Column:** Top 3 nearest biopsy twins of the **Predicted Class**.
     * **Right Column:** Top 3 nearest biopsy twins of the **Rival Runner-up Class**.
   * Normalized cosine similarity percentages are displayed per twin.

---

## Clinical Safety Gates

* **Out-of-Distribution (OOD) Gate:** Measures latent $k$-NN distance ($k=5$) in the 384-dimensional DINOv2 space. If distance exceeds the 95th validation percentile ($\tau_{95} = 0.2376$), a clinical OOD banner warns that the specimen diverges from training morphology.
* **Subgroup Sensitivity Warning:** Flags anatomical regions with statistically significant historical sensitivity drops (e.g. *Head & neck* lesions: 62.1% accuracy vs. 74.6% global baseline).
* **Two-Step Sign-Off Protocol:** Physicians cannot sign off while inside the twin menu. Pressing <kbd>Enter</kbd> returns to the main HUD so the clinician reviews metadata, triple-view anatomical avatar, and subgroup alerts before finalizing.

---

## Keyboard Shortcuts (Tactical HUD)

| Key | Action |
| :--- | :--- |
| <kbd>1</kbd> – <kbd>8</kbd> | Select Working Diagnosis (Melanoma, Nevus, BCC, AKIEC, BKL, DF, VASC, Other) |
| <kbd>W</kbd> | Toggle 6 Reference Twins View (Predicted vs. Rival Match) |
| <kbd>Q</kbd> / <kbd>E</kbd> | Adjust Grad-CAM Heatmap Opacity ($\pm 50\%$) |
| <kbd>S</kbd> | Toggle AI Forecast Confidence Bars |
| <kbd>A</kbd> / <kbd>D</kbd> | Navigate Previous / Next Case |
| <kbd>Enter</kbd> | **In Twins View:** Return to Main Menu<br>**In Main Menu:** Sign-Off Case |

---

## Configuration & Switching ML Runs

Settings are managed in `ui/.env`:

```bash
# Path to the exported ML run (must contain export/classifier_outputs.parquet and heatmaps)
ML_RUN_DIR="../runs/clf-finetune-dinov2/2026-11-11_22-10-36_s42"

# Root dataset directory (containing tables/items.parquet and images/)
DATA_BUILD_DIR="../data/build"

# Server Port
PORT=8000
```

### To switch to a new ML run:
1. Export the run:
   ```bash
   uv run colesion export runs/clf-finetune-dinov2/<your_new_run>
   ```
2. Update `ML_RUN_DIR` in `ui/.env`.
3. Recalibrate and start:
   ```bash
   ./run.sh --recalibrate
   ```

---

## Architecture: Zero Disk Duplication

* **`calibrate.py` (Offline Fitting):** Reads existing Parquet files, fits $T$ on `val`, indexes embeddings via cosine dot products in matrix form, and writes a compact 203 KB index to `ui/state/calibrated_cases.parquet`.
* **`server.py` (Zero-Copy HTTP Server):** Streams high-resolution JPGs directly from `data/build/images` and maps heatmaps on-the-fly. No images are copied or duplicated on disk.
