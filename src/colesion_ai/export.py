"""`colesion export <run>`: what Jonas's dashboard needs from the classifier, from one finished
run (docs/classifier_pipeline_requirements.md). Writes into <run>/export/:

classifier_outputs.parquet, one row per image (all 11,427, every split), joins items on image_id:
    image_id, split, logit_akiec ... logit_vasc (uncalibrated), embedding (L2-normalised),
    heatmap (path relative to export/, only for the study cases in HEATMAP_SPLITS)
heatmaps/<image_id>.png: Grad-CAM of the top-1 class, same size as the original, turbo colours.

The heatmaps belong to this run's head: a retrained head gives other maps.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colesion_ai.core.runtime import resolve_device
from colesion_ai.data.dataset import CLASSES, ImageDataset, image_transform, load_items
from colesion_ai.data.embeddings import load_embeddings
from colesion_ai.models.classifier import build_backbone, load_run
from colesion_ai.models.saliency import grad_cam

log = logging.getLogger(__name__)

HEATMAP_SPLITS = ("pool", "holdout")  # the cases people see in the study


def export(run_dir: Path) -> Path:
    cfg, head = load_run(run_dir)
    items = load_items(cfg.env.data_root)
    embeddings = load_embeddings(cfg, items)
    with torch.no_grad():
        logits = head(embeddings).numpy()
    out = run_dir / "export"

    study_cases = pl.col("split").is_in(HEATMAP_SPLITS)
    write_heatmaps(cfg, head, items.filter(study_cases), out / "heatmaps")
    table = items.select("image_id", "split").with_columns(
        **{f"logit_{c}": logits[:, i] for i, c in enumerate(CLASSES)},
        embedding=pl.Series(F.normalize(embeddings, dim=1).numpy()),
        heatmap=pl.when(study_cases).then(pl.format("heatmaps/{}.png", "image_id")),
    )
    table.write_parquet(out / "classifier_outputs.parquet")
    log.info("wrote %s (%d rows)", out / "classifier_outputs.parquet", table.height)
    return out


def write_heatmaps(cfg, head, items: pl.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.env.device)
    backbone = build_backbone(cfg.model, cfg.data.image_size).to(device).eval()
    head = head.to(device)
    transform = image_transform(cfg.data.image_size, train=False)
    loader = DataLoader(ImageDataset(items, cfg.env.data_root, transform), batch_size=32)
    turbo = colormaps["turbo"]

    rows = items.select("image_id", "image_path").iter_rows()
    for x, _ in tqdm(loader, desc="heatmaps"):
        for cam, (image_id, image_path) in zip(grad_cam(backbone, head, x.to(device)), rows):
            width, height = Image.open(Path(cfg.env.data_root) / image_path).size
            cam = F.interpolate(cam[None, None], size=(height, width), mode="bilinear")[0, 0]
            rgb = (turbo(cam.clamp(0, 1).cpu().numpy())[..., :3] * 255).astype(np.uint8)
            Image.fromarray(rgb).save(folder / f"{image_id}.png")
