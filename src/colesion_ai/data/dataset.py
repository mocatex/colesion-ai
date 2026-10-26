"""Minimal image loading for phase 1. Phase 2 adds the image and embedding caches."""

import json
import signal
from pathlib import Path

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

# dx_idx 0..6, fixed in the dataset card; never derive it from the data
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
# ImageNet statistics, what pretrained backbones expect
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_split(data_root, split: str, limit: int | None = None) -> pl.DataFrame:
    root = Path(data_root)
    built = tuple(json.loads((root / "BUILD.json").read_text())["classes"])
    if built != CLASSES:
        raise ValueError(f"class order in BUILD.json is {built}, the code expects {CLASSES}")
    items = pl.read_parquet(root / "tables" / "items.parquet").filter(pl.col("split") == split)
    if limit is not None and limit < items.height:
        items = items.sample(n=limit, seed=0)  # same subset for every seed and every resume
    return items


def image_transform(size: int, train: bool) -> T.Compose:
    if train:
        steps = [
            T.RandomResizedCrop(size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
    else:
        steps = [T.Resize(size), T.CenterCrop(size)]
    return T.Compose(
        [T.ToImage(), *steps, T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    )


class ImageDataset(Dataset):
    """(image tensor, dx_idx) for each row of `items`."""

    def __init__(self, items: pl.DataFrame, data_root, transform):
        self.paths = [Path(data_root) / p for p in items["image_path"]]
        self.labels = items["dx_idx"].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert("RGB")), self.labels[i]


def ignore_signals(worker_id: int) -> None:
    """`worker_init_fn` for DataLoaders. Ctrl-C (and SLURM's SIGTERM) reaches every process;
    workers ignore it, so only the main process reacts: it stops cleanly and shuts them down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
