import logging

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from colesion_ai.core.config import TrainEntryCfg
from colesion_ai.core.run import Run
from colesion_ai.data.dataset import (
    CLASSES,
    ImageDataset,
    ignore_signals,
    image_transform,
    load_split,
)
from colesion_ai.models.classifier import build_classifier
from colesion_ai.report import plot_curves

log = logging.getLogger(__name__)


def train(cfg: TrainEntryCfg, force: bool = False) -> Run:
    run = Run(cfg, force=force)
    if run.status == "done":
        log.info("%s is done, skipping it (--force starts a new run)", run.dir)
        return run

    with run:
        fit = load_split(cfg.env.data_root, "fit", cfg.data.max_fit)
        val = load_split(cfg.env.data_root, "val", cfg.data.max_val)
        fit_loader = make_loader(fit, cfg, train=True)
        val_loader = make_loader(val, cfg, train=False)

        model = build_classifier(cfg.model, len(CLASSES), cfg.data.image_size).to(run.device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        weights = class_weights(fit["dx_idx"].to_numpy(), cfg.train.class_weighting)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(run.device)

        state = run.restore(model=model, optimizer=optimizer)
        step = state.get("step", 0)
        best, best_epoch = state.get("best", -1.0), state.get("best_epoch")
        for epoch in range(state["start_epoch"], cfg.train.epochs):
            model.train()
            train_probs, train_labels = [], []
            for x, y in fit_loader:
                logits = model(x.to(run.device))
                loss = loss_fn(logits, y.to(run.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                train_probs.append(logits.detach().softmax(1).cpu())
                train_labels.append(y)
                if step % cfg.tracking.log_every_n_steps == 0:
                    run.log({"batch_loss": loss.item()}, step=step, epoch=epoch, split="train")

            # train metrics with the same definitions as val, so the curves can share a panel.
            # They are running values (augmented images, model changing during the epoch).
            # batch_loss is the class-weighted loss that is actually optimised.
            train_metrics = classification_metrics(
                torch.cat(train_labels).numpy(), torch.cat(train_probs).numpy()
            )
            probs = predict(model, val_loader, run.device)
            metrics = classification_metrics(val["dx_idx"].to_numpy(), probs)
            run.log(train_metrics, step=step, epoch=epoch, split="train")
            run.log(metrics, step=step, epoch=epoch, split="val")
            is_best = metrics["bal_acc"] > best
            if is_best:
                best, best_epoch = metrics["bal_acc"], epoch
            run.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=is_best,
                step=step,
                best=best,
                best_epoch=best_epoch,
            )
            run.summary.update(
                {f"val_{k}": v for k, v in metrics.items()},
                epochs_done=epoch + 1,
                best_epoch=best_epoch,
                best_val_bal_acc=best,
            )
            # redrawn every epoch, so the plot is there after a stop too
            plot_curves([run.dir], run.dir / "curves.png")

        probs = predict(model, val_loader, run.device)
        write_predictions(run.dir / "artifacts" / "predictions_val.parquet", val, probs)
    return run


def make_loader(items: pl.DataFrame, cfg: TrainEntryCfg, train: bool) -> DataLoader:
    dataset = ImageDataset(items, cfg.env.data_root, image_transform(cfg.data.image_size, train))
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=train,
        num_workers=cfg.env.num_workers,
        worker_init_fn=ignore_signals,
    )


def class_weights(labels: np.ndarray, scheme: str) -> torch.Tensor | None:
    if scheme == "none":
        return None
    if scheme == "inverse_freq":
        counts = np.bincount(labels, minlength=len(CLASSES))
        return torch.tensor(
            len(labels) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32
        )
    raise ValueError(f"train.class_weighting must be 'none' or 'inverse_freq', got {scheme!r}")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    return torch.cat([model(x.to(device)).softmax(1).cpu() for x, _ in loader]).numpy()


def classification_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]  # classes present, as in sklearn
    return {
        "loss": float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()),
        "acc": float(np.mean(pred == y)),
        "bal_acc": float(np.mean(recalls)),
    }


def write_predictions(path, items: pl.DataFrame, probs: np.ndarray) -> None:
    """Per-image probabilities: any metric can be recomputed later without retraining."""
    columns = {f"prob_{c}": probs[:, i] for i, c in enumerate(CLASSES)}
    items.select("image_id", "dx_idx").with_columns(**columns).write_parquet(path)
