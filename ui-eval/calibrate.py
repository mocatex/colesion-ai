#!/usr/bin/env python3
"""
COLESION-AI // Mathematical Calibration & Precedent Indexing
Performs:
  1. Temperature Scaling on validation split (NLL minimization)
  2. OOD Manifold Distance & Tau_95 Threshold on validation split
  3. Subgroup / Clinical Slice Performance Analysis
  4. Histopathology Biopsy Twins Search Indexing (Top 3 Pred, Top 3 Rival, Top 6 Nearest Latent)
Outputs:
  - ui/state/calibrated_cases.parquet (compact case metadata + probabilities + 6-twins sets)
  - ui/state/metrics.json (calibration summary for thesis and monitoring)
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("colesion.calibrate")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis (akiec)",
    "bcc": "Basal Cell Carcinoma (bcc)",
    "bkl": "Benign Keratosis (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic Nevus (nv)",
    "vasc": "Vascular Lesion (vasc)",
}


def load_env(env_path: Path) -> dict:
    config = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip().strip('"').strip("'")
    return config


def softmax(z: np.ndarray, axis=-1) -> np.ndarray:
    e_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e_z / np.sum(e_z, axis=axis, keepdims=True)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total) * abs(bin_acc - bin_conf)
    return float(ece)


def main():
    ui_dir = Path(__file__).resolve().parent
    env = load_env(ui_dir / ".env")

    ml_run_dir = Path(env.get("ML_RUN_DIR", "../runs/clf-finetune-dinov2/2026-11-11_22-10-36_s42"))
    if not ml_run_dir.is_absolute():
        ml_run_dir = (ui_dir / ml_run_dir).resolve()

    data_build_dir = Path(env.get("DATA_BUILD_DIR", "../data/build"))
    if not data_build_dir.is_absolute():
        data_build_dir = (ui_dir / data_build_dir).resolve()

    log.info(f"Using ML run: {ml_run_dir}")
    log.info(f"Using dataset build: {data_build_dir}")

    parquet_path = ml_run_dir / "export" / "classifier_outputs.parquet"
    items_path = data_build_dir / "tables" / "items.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing classifier outputs: {parquet_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing items metadata: {items_path}")

    df_ml = pl.read_parquet(parquet_path)
    df_items = pl.read_parquet(items_path)

    df = df_ml.join(
        df_items.select(
            [
                "image_id",
                "dx",
                "dx_idx",
                "dx_type",
                "age",
                "sex",
                "localization",
                "image_path",
                "n_raters",
                "h_entropy",
            ]
        ),
        on="image_id",
        how="inner",
    )
    log.info(f"Loaded {df.height} merged records across all splits.")

    # 1. Temperature Scaling on Validation Split
    val_df = df.filter(pl.col("split") == "val")
    val_logits = val_df.select([f"logit_{c}" for c in CLASSES]).to_numpy()
    val_labels = val_df["dx_idx"].to_numpy()

    def nll(t_param):
        T = t_param[0]
        scaled = val_logits / T
        probs = softmax(scaled, axis=1)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        loss = -np.mean(np.log(probs[np.arange(len(val_labels)), val_labels]))
        return loss

    res = minimize(nll, x0=[1.0], bounds=[(0.05, 5.0)])
    optimal_T = float(res.x[0])

    val_raw_probs = softmax(val_logits, axis=1)
    val_cal_probs = softmax(val_logits / optimal_T, axis=1)

    val_acc = float(np.mean(np.argmax(val_logits, axis=1) == val_labels))
    ece_raw = compute_ece(val_raw_probs, val_labels)
    ece_cal = compute_ece(val_cal_probs, val_labels)

    log.info(f"[Temperature Scaling] Optimal T: {optimal_T:.4f}")
    log.info(
        f"[Temperature Scaling] Val Acc: {val_acc * 100:.2f}%, ECE raw: {ece_raw:.4f} -> calibrated: {ece_cal:.4f}"
    )

    # 2. OOD Calibration on 'fit' Embeddings
    fit_df = df.filter(pl.col("split") == "fit")
    fit_embs = np.vstack(fit_df["embedding"].to_numpy())  # (8524, 384)
    val_embs = np.vstack(val_df["embedding"].to_numpy())  # (1491, 384)

    k_nn = 10
    val_dists = []
    for i in range(0, len(val_embs), 250):
        batch = val_embs[i : i + 250]
        sims = batch @ fit_embs.T
        topk_sims = np.partition(sims, -k_nn, axis=1)[:, -k_nn:]
        kth_sim = np.min(topk_sims, axis=1)
        val_dists.extend(1.0 - kth_sim)

    val_dists = np.array(val_dists)
    tau_95 = float(np.percentile(val_dists, 95))
    mean_val_dist = float(np.mean(val_dists))
    log.info(f"[OOD Gate] Mean 10-NN dist on val: {mean_val_dist:.4f}, Tau_95: {tau_95:.4f}")

    # 3. Subgroup Slice Analysis on Validation Split
    val_preds = np.argmax(val_logits, axis=1)
    val_correct = val_preds == val_labels
    val_df = val_df.with_columns(correct=pl.Series(val_correct))

    slice_stats = (
        val_df.group_by("localization")
        .agg(count=pl.len(), acc=pl.col("correct").mean())
        .sort("acc")
    )

    sensitive_slices = {}
    for row in slice_stats.iter_rows(named=True):
        loc = row["localization"]
        cnt = row["count"]
        acc = row["acc"]
        if loc and cnt >= 20 and acc < (val_acc - 0.08):
            sensitive_slices[loc] = {
                "acc": float(acc),
                "count": int(cnt),
                "drop": float(val_acc - acc),
                "message": f"Historical sensitivity at '{loc}' is {acc * 100:.1f}% (below baseline of {val_acc * 100:.1f}%). Exercise caution with subtle lesions.",
            }
            log.info(
                f"[Subgroup Alert Triggered] Site '{loc}': {acc * 100:.1f}% vs baseline {val_acc * 100:.1f}%"
            )

    # 4. Precedents / Reference Twins Pool (Fit & Histopathology Verified)
    histo_fit = fit_df.filter(pl.col("dx_type") == "histo")
    log.info(f"[Twins Pool] {histo_fit.height} histopathologically verified cases in 'fit' set.")
    histo_embs = np.vstack(histo_fit["embedding"].to_numpy())  # (4559, 384)
    histo_dx = histo_fit["dx"].to_numpy()
    histo_ids = histo_fit["image_id"].to_list()

    # Pre-index indices by class for fast retrieval
    class_indices = {c: np.where(histo_dx == c)[0] for c in CLASSES}

    # 5. Process Study Cases (holdout + pool = 1,412 cases)
    study_df = df.filter(pl.col("split").is_in(["holdout", "pool"]))
    log.info(f"Processing {study_df.height} study cases (holdout + pool)...")

    study_logits = study_df.select([f"logit_{c}" for c in CLASSES]).to_numpy()
    study_embs = np.vstack(study_df["embedding"].to_numpy())
    study_cal_probs = softmax(study_logits / optimal_T, axis=1)

    # Storage for multi-twins (Top 3 Pred, Top 3 Rival, Top 6 Nearest Latent)
    pred_twin_ids_list, pred_twin_dxs_list, pred_twin_dists_list = [], [], []
    rival_twin_ids_list, rival_twin_dxs_list, rival_twin_dists_list = [], [], []
    nearest_twin_ids_list, nearest_twin_dxs_list, nearest_twin_dists_list = [], [], []

    # Backward compatibility single twins
    top_twin_ids, top_twin_dists, top_twin_dxs = [], [], []
    rival_twin_ids, rival_twin_dists, rival_twin_dxs = [], [], []

    case_knn_dists = []
    is_ood_flags = []
    margin_deltas = []
    pred_classes, pred_full_names, rival_classes = [], [], []
    slice_warnings, slice_messages = [], []

    # Batch compute k-NN distance to all fit cases for OOD
    for i in range(0, len(study_embs), 200):
        b = study_embs[i : i + 200]
        s = b @ fit_embs.T
        topk = np.partition(s, -k_nn, axis=1)[:, -k_nn:]
        k_sim = np.min(topk, axis=1)
        d = 1.0 - k_sim
        case_knn_dists.extend(d.tolist())
        is_ood_flags.extend((d > tau_95).tolist())

    # Precedent Twin search per case
    for idx in range(len(study_df)):
        q = study_cal_probs[idx]
        sorted_indices = np.argsort(q)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]

        top1_name = CLASSES[top1_idx]
        top2_name = CLASSES[top2_idx]
        delta = float(q[top1_idx] - q[top2_idx])

        pred_classes.append(top1_name)
        pred_full_names.append(CLASS_FULL_NAMES[top1_name])
        rival_classes.append(top2_name)
        margin_deltas.append(round(delta, 4))

        emb = study_embs[idx]

        # A) TOP 3 PREDICTED CLASS TWINS
        cand_top = class_indices.get(top1_name, np.array([]))
        if len(cand_top) > 0:
            top_sims = histo_embs[cand_top] @ emb
            best_rel_order = np.argsort(top_sims)[::-1][:3]
            best_abs = cand_top[best_rel_order]

            p_ids = [histo_ids[k] for k in best_abs]
            p_dxs = [top1_name] * len(p_ids)
            p_dists = [round(float(1.0 - top_sims[r]), 4) for r in best_rel_order]

            pred_twin_ids_list.append(p_ids)
            pred_twin_dxs_list.append(p_dxs)
            pred_twin_dists_list.append(p_dists)

            top_twin_ids.append(p_ids[0])
            top_twin_dxs.append(top1_name)
            top_twin_dists.append(p_dists[0])
        else:
            pred_twin_ids_list.append([])
            pred_twin_dxs_list.append([])
            pred_twin_dists_list.append([])
            top_twin_ids.append(None)
            top_twin_dxs.append(None)
            top_twin_dists.append(None)

        # B) TOP 3 RIVAL CLASS TWINS
        cand_riv = class_indices.get(top2_name, np.array([]))
        if len(cand_riv) > 0:
            riv_sims = histo_embs[cand_riv] @ emb
            best_rel_order_riv = np.argsort(riv_sims)[::-1][:3]
            best_abs_riv = cand_riv[best_rel_order_riv]

            r_ids = [histo_ids[k] for k in best_abs_riv]
            r_dxs = [top2_name] * len(r_ids)
            r_dists = [round(float(1.0 - riv_sims[r]), 4) for r in best_rel_order_riv]

            rival_twin_ids_list.append(r_ids)
            rival_twin_dxs_list.append(r_dxs)
            rival_twin_dists_list.append(r_dists)

            rival_twin_ids.append(r_ids[0])
            rival_twin_dxs.append(top2_name)
            rival_twin_dists.append(r_dists[0])
        else:
            rival_twin_ids_list.append([])
            rival_twin_dxs_list.append([])
            rival_twin_dists_list.append([])
            rival_twin_ids.append(None)
            rival_twin_dxs.append(None)
            rival_twin_dists.append(None)

        # C) TOP 6 NEAREST IN LATENT SPACE OVERALL
        all_sims = histo_embs @ emb
        top6_abs_order = np.argsort(all_sims)[::-1][:6]
        n_ids = [histo_ids[k] for k in top6_abs_order]
        n_dxs = [histo_dx[k] for k in top6_abs_order]
        n_dists = [round(float(1.0 - all_sims[k]), 4) for k in top6_abs_order]

        nearest_twin_ids_list.append(n_ids)
        nearest_twin_dxs_list.append(n_dxs)
        nearest_twin_dists_list.append(n_dists)

        # Subgroup check
        loc = study_df["localization"][idx]
        if loc in sensitive_slices:
            slice_warnings.append(True)
            slice_messages.append(sensitive_slices[loc]["message"])
        else:
            slice_warnings.append(False)
            slice_messages.append(
                "Standard subgroup performance within expected validation parameters."
            )

    prob_cols = {
        f"prob_{c}": study_cal_probs[:, i].astype(np.float32) for i, c in enumerate(CLASSES)
    }

    out_df = study_df.select(
        [
            "image_id",
            "split",
            "dx",
            "dx_idx",
            "dx_type",
            "age",
            "sex",
            "localization",
            "image_path",
            "heatmap",
            "n_raters",
            "h_entropy",
        ]
    ).with_columns(
        **prob_cols,
        pred_class=pl.Series(pred_classes),
        pred_full_name=pl.Series(pred_full_names),
        rival_class=pl.Series(rival_classes),
        margin_delta=pl.Series(margin_deltas),
        knn_distance=pl.Series(np.round(case_knn_dists, 4)),
        is_ood=pl.Series(is_ood_flags),
        slice_warning=pl.Series(slice_warnings),
        slice_message=pl.Series(slice_messages),
        # Legacy single twins
        top_twin_id=pl.Series(top_twin_ids),
        top_twin_dx=pl.Series(top_twin_dxs),
        top_twin_dist=pl.Series(top_twin_dists),
        rival_twin_id=pl.Series(rival_twin_ids),
        rival_twin_dx=pl.Series(rival_twin_dxs),
        rival_twin_dist=pl.Series(rival_twin_dists),
        # 6-Twin lists
        pred_twin_ids=pl.Series(pred_twin_ids_list),
        pred_twin_dxs=pl.Series(pred_twin_dxs_list),
        pred_twin_dists=pl.Series(pred_twin_dists_list),
        rival_twin_ids=pl.Series(rival_twin_ids_list),
        rival_twin_dxs=pl.Series(rival_twin_dxs_list),
        rival_twin_dists=pl.Series(rival_twin_dists_list),
        nearest_twin_ids=pl.Series(nearest_twin_ids_list),
        nearest_twin_dxs=pl.Series(nearest_twin_dxs_list),
        nearest_twin_dists=pl.Series(nearest_twin_dists_list),
    )

    state_dir = ui_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = state_dir / "calibrated_cases.parquet"
    out_df.write_parquet(out_parquet)
    log.info(
        f"Saved calibrated state with 6-twins: {out_parquet} ({out_df.height} rows, {out_parquet.stat().st_size / 1024:.1f} KB)"
    )

    metrics = {
        "temperature": optimal_T,
        "val_accuracy": val_acc,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "tau_95": tau_95,
        "val_mean_knn_dist": mean_val_dist,
        "sensitive_slices": sensitive_slices,
        "num_study_cases": out_df.height,
        "num_holdout_cases": int((out_df["split"] == "holdout").sum()),
        "num_pool_cases": int((out_df["split"] == "pool").sum()),
        "num_ood_flagged": int(out_df["is_ood"].sum()),
    }
    with open(state_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics summary to: {state_dir / 'metrics.json'}")
    log.info("Calibration & multi-twin indexing successfully completed!")


if __name__ == "__main__":
    main()
