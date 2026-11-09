"""`colesion evaluate <run> [--split val]`: how good is a finished run, class by class?

Uses the run's best epoch and the cached embeddings: no retraining, no images. Writes into
<run>/eval/<split>/: predictions.parquet (logits), metrics.json, per_class.csv, confusion.csv and
confusion.png, and adds <split>_<metric> to the run's summary.json (so `colesion report` shows it).
Choose models on `val`; run `holdout` only for the final candidates.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from matplotlib.figure import Figure
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from colesion_ai.core.run import read_json, write_json
from colesion_ai.data.dataset import CLASSES, load_split
from colesion_ai.data.embeddings import load_embeddings
from colesion_ai.models.classifier import load_run
from colesion_ai.train import classification_metrics

log = logging.getLogger(__name__)


def evaluate(run_dir: Path, split: str = "val") -> dict:
    cfg, head = load_run(run_dir)
    items = load_split(cfg.env.data_root, split)
    with torch.no_grad():
        logits = head(load_embeddings(cfg, items)).numpy()
    y, probs = items["dx_idx"].to_numpy(), softmax(logits, axis=1)
    pred = probs.argmax(1)
    out = run_dir / "eval" / split
    out.mkdir(parents=True, exist_ok=True)

    logit_columns = {f"logit_{c}": logits[:, i] for i, c in enumerate(CLASSES)}
    predictions = items.select("image_id", "dx_idx").with_columns(**logit_columns)
    predictions.write_parquet(out / "predictions.parquet")

    classes = range(len(CLASSES))
    precision, recall, _, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0
    )
    per_class = pl.DataFrame(
        {
            "class": CLASSES,
            "support": support,
            "recall": recall,
            "precision": precision,
            "auc": [roc_auc_score(y == c, probs[:, c]) for c in classes],  # one class vs the rest
        }
    )
    per_class.write_csv(out / "per_class.csv")

    mel = per_class.row(CLASSES.index("mel"), named=True)
    metrics = {
        **classification_metrics(y, logits),
        "macro_auc": float(per_class["auc"].mean()),
        "mel_recall": float(mel["recall"]),
        "mel_auc": float(mel["auc"]),
    }
    write_json(out / "metrics.json", metrics)

    counts = confusion_matrix(y, pred, labels=classes)
    table = pl.DataFrame(counts, schema=[f"pred_{c}" for c in CLASSES])
    table.insert_column(0, pl.Series("true", CLASSES)).write_csv(out / "confusion.csv")
    title = f"{run_dir.parent.name}/{run_dir.name} on {split}"
    plot_confusion(counts, out / "confusion.png", title)

    summary = read_json(run_dir / "summary.json")
    summary.update({f"{split}_{k}": v for k, v in metrics.items()})
    write_json(run_dir / "summary.json", summary)
    log.info("%s: %s", split, "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log.info("wrote %s", out)
    return metrics


def plot_confusion(counts: np.ndarray, path: Path, title: str) -> None:
    """Rows = true class, columns = predicted. Each cell: count and share of its row; the
    diagonal share is the recall. Colour = share, so the rare classes stay visible."""
    share = counts / counts.sum(1, keepdims=True).clip(min=1)
    fig = Figure(figsize=(6.4, 5.4), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(counts.shape):
        label = f"{counts[i, j]}\n{share[i, j]:.0%}"
        color = "white" if share[i, j] > 0.5 else "#1a1a19"
        ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
    ticks = range(len(CLASSES))
    ax.set(xticks=ticks, xticklabels=CLASSES, yticks=ticks, yticklabels=CLASSES)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("true", fontsize=9)
    ax.set_title(title, loc="left", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8, label="share of the true class")
    fig.savefig(path, dpi=150)
