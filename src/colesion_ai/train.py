"""`colesion train`: train the head on cached embeddings (`model.freeze: all`, seconds), or
fine-tune backbone + head on images (`model.freeze: none`). Both share the epoch loop below."""

import logging
import math

import numpy as np
import polars as pl
import torch
from scipy.special import softmax
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from colesion_ai.core.config import TrainEntryCfg
from colesion_ai.core.run import Run
from colesion_ai.data.dataset import (
    CLASSES,
    ImageDataset,
    ignore_signals,
    image_transform,
    load_split,
)
from colesion_ai.data.embeddings import load_embeddings
from colesion_ai.models.classifier import build_backbone, build_head
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
        size = cfg.data.image_size
        if cfg.model.freeze == "all":
            # the head alone, on cached embeddings: no augmentation, an epoch takes a moment
            fit_data = TensorDataset(load_embeddings(cfg, fit), labels(fit))
            val_data = TensorDataset(load_embeddings(cfg, val), labels(val))
            head = build_head(cfg.model.head, fit_data.tensors[0].shape[1])
            parts = {"head": head}
            groups = [{"params": head.parameters(), "lr": cfg.train.lr}]
            workers = 0  # the data is already in memory
        elif cfg.model.freeze == "none":
            # backbone + head end to end on images; the backbone learns more slowly
            fit_data = ImageDataset(fit, cfg.env.data_root, image_transform(size, train=True))
            val_data = ImageDataset(val, cfg.env.data_root, image_transform(size, train=False))
            backbone = build_backbone(cfg.model, size)
            head = build_head(cfg.model.head, backbone.num_features)
            parts = {"backbone": backbone, "head": head}
            groups = [
                {"params": backbone.parameters(), "lr": cfg.train.backbone_lr},
                {"params": head.parameters(), "lr": cfg.train.lr},
            ]
            workers = cfg.env.num_workers
        else:
            raise ValueError(f"model.freeze must be 'all' or 'none', got {cfg.model.freeze!r}")

        model = nn.Sequential(*parts.values()).to(run.device)
        fit_loader = make_loader(fit_data, cfg, workers, shuffle=True)
        val_loader = make_loader(val_data, cfg, workers, shuffle=False)
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
        scheduler = lr_schedule(optimizer, cfg.train.schedule, len(fit_loader), cfg.train.epochs)
        weights = class_weights(fit["dx_idx"].to_numpy(), cfg.train.class_weighting)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(run.device)

        state = run.restore(**parts, optimizer=optimizer, scheduler=scheduler)
        step = state.get("step", 0)
        best, best_epoch = state.get("best", -1.0), state.get("best_epoch")
        for epoch in range(state["start_epoch"], cfg.train.epochs):
            model.train()
            train_logits, train_labels = [], []
            for x, y in fit_loader:
                logits = model(x.to(run.device))
                loss = loss_fn(logits, y.to(run.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                train_logits.append(logits.detach().cpu())
                train_labels.append(y)
                if step % cfg.tracking.log_every_n_steps == 0:
                    run.log({"batch_loss": loss.item()}, step=step, epoch=epoch, split="train")

            # train metrics with the same definitions as val, so the curves can share a panel.
            # They are running values (augmented images, model changing during the epoch).
            # batch_loss is the class-weighted loss that is actually optimised.
            train_metrics = classification_metrics(
                torch.cat(train_labels).numpy(), torch.cat(train_logits).numpy()
            )
            # the head's lr at the end of the epoch; the backbone's follows the same schedule
            train_metrics["lr"] = optimizer.param_groups[-1]["lr"]
            val_logits = predict(model, val_loader, run.device)
            metrics = classification_metrics(val["dx_idx"].to_numpy(), val_logits)
            run.log(train_metrics, step=step, epoch=epoch, split="train")
            run.log(metrics, step=step, epoch=epoch, split="val")
            is_best = metrics["bal_acc"] > best
            if is_best:
                best, best_epoch = metrics["bal_acc"], epoch
            run.save_checkpoint(
                **parts,
                optimizer=optimizer,
                scheduler=scheduler,
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
    return run


def labels(items: pl.DataFrame) -> torch.Tensor:
    return torch.from_numpy(items["dx_idx"].to_numpy()).long()


def make_loader(dataset, cfg: TrainEntryCfg, workers: int, shuffle: bool) -> DataLoader:
    # Shuffling and worker seeds are drawn from torch's global RNG on every new epoch, and the
    # checkpoint restores that RNG: so a resumed run sees the same batches. (persistent_workers
    # would draw the worker seeds only once, in epoch 0, and break this.)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=ignore_signals,
    )


def lr_schedule(optimizer, schedule: str, steps_per_epoch: int, epochs: int) -> LambdaLR:
    """`train.schedule` as a factor on every learning rate, updated after each step.
    cosine: rises over the first epoch (a fresh head must not shake the pretrained backbone),
    then falls to 0 along a cosine, so the last epochs only fine-tune and the val curve settles."""
    warmup, total = steps_per_epoch, steps_per_epoch * epochs

    def cosine(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    if schedule == "constant":
        return LambdaLR(optimizer, lambda step: 1.0)
    if schedule == "cosine":
        return LambdaLR(optimizer, cosine)
    raise ValueError(f"train.schedule must be 'constant' or 'cosine', got {schedule!r}")


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
    """Logits for a whole split."""
    model.eval()
    return torch.cat([model(x.to(device)).cpu() for x, _ in loader]).numpy()


def classification_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = softmax(logits, axis=1)
    pred = probs.argmax(1)
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]  # classes present, as in sklearn
    return {
        "loss": float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()),
        "acc": float(np.mean(pred == y)),
        "bal_acc": float(np.mean(recalls)),
    }
