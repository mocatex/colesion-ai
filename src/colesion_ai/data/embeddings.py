"""Embedding cache: every image goes through the backbone once; heads then train in seconds.

One file per backbone version, with all 11,427 images (every split):
cache/embeddings/<backbone>_<image size>_<weights>.parquet, columns image_id + embedding.
"""

import logging
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colesion_ai.core.runtime import resolve_device
from colesion_ai.data.dataset import ImageDataset, ignore_signals, image_transform, load_items
from colesion_ai.models.classifier import build_backbone

log = logging.getLogger(__name__)


def load_embeddings(cfg, items: pl.DataFrame) -> torch.Tensor:
    """Embeddings for the rows of `items`, in their order. Computed and cached on first use."""
    path = cache_path(cfg)
    if not path.exists():
        compute_embeddings(cfg, path)
    cache = pl.read_parquet(path)
    rows = items.select("image_id").join(cache, on="image_id", how="left", maintain_order="left")
    return torch.from_numpy(rows["embedding"].to_numpy(writable=True))


def cache_path(cfg) -> Path:
    weights = cfg.model.weights
    if weights not in ("pretrained", "random"):  # a fine-tuned run: runs/<name>/<folder>
        weights = "-".join(Path(weights).parts[-2:])
    name = f"{cfg.model.backbone}_{cfg.data.image_size}_{weights}.parquet"
    return Path(cfg.env.cache_root) / "embeddings" / name


@torch.no_grad()
def compute_embeddings(cfg, path: Path) -> None:
    items = load_items(cfg.env.data_root)
    device = resolve_device(cfg.env.device)
    backbone = build_backbone(cfg.model, cfg.data.image_size).to(device).eval()
    transform = image_transform(cfg.data.image_size, train=False)
    loader = DataLoader(
        ImageDataset(items, cfg.env.data_root, transform),
        batch_size=64,
        num_workers=cfg.env.num_workers,
        worker_init_fn=ignore_signals,
    )
    log.info("embedding %d images with %s on %s", items.height, path.stem, device)
    batches = [backbone(x.to(device)).cpu() for x, _ in tqdm(loader, desc="embed")]

    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = torch.cat(batches).numpy()
    pl.DataFrame({"image_id": items["image_id"], "embedding": embeddings}).write_parquet(path)
    log.info("cached %s (%d x %d)", path, *embeddings.shape)
