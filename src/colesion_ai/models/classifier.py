"""The classifier: a backbone (image -> embedding) and a linear head (embedding -> 7 logits).

The head always sees embeddings, so after the backbone is fixed, everything (head training,
evaluation, export, later the loop) runs on cached embeddings: data/embeddings.py.
"""

from pathlib import Path

import timm
import torch
from torch import nn

from colesion_ai.core.config import ModelCfg, load_config

# our name -> timm name
DINOV2 = {
    # DINOv2 ViT-S/14 (Oquab et al., 2023)
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    # the same with 4 register tokens (Darcet et al., 2024): without them, some background
    # patches carry huge values, which shows up as scattered hot spots in Grad-CAM maps
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
}


def build_backbone(cfg: ModelCfg, image_size: int) -> nn.Module:
    """A timm model without its own classification layer (`num_classes=0`): it returns one
    embedding of `.num_features` numbers per image. A new backbone is one more `if` here."""
    pretrained = cfg.weights == "pretrained"
    if cfg.backbone == "resnet18":
        backbone = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
    elif cfg.backbone in DINOV2:
        # timm's default input for DINOv2 is 518 px, so pass the size we use
        backbone = timm.create_model(
            DINOV2[cfg.backbone], pretrained=pretrained, num_classes=0, img_size=image_size
        )
    else:
        raise ValueError(f"unknown model.backbone {cfg.backbone!r}: add it to build_backbone()")

    if cfg.weights not in ("pretrained", "random"):  # a run folder that fine-tuned this backbone
        backbone.load_state_dict(load_checkpoint(Path(cfg.weights))["backbone"])
    return backbone


def build_head(name: str, in_features: int, num_classes: int = 7) -> nn.Module:
    """`model.head`: embedding -> one logit per class"""
    if name == "linear":
        return nn.Linear(in_features, num_classes)
    elif name == "mlp":
        return nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
    raise ValueError(f"unknown model.head {name!r}: add it to build_head()")


def load_run(run_dir: Path):
    """Config and trained head (best epoch) of a finished run.

    The config's `model.weights` then names the backbone that makes this head's embeddings:
    unchanged for a head-only run, the run itself for a fine-tuning run."""
    cfg = load_config(run_dir / "config.yaml")
    if cfg.model.freeze == "none":
        cfg.model.weights = str(run_dir)
    state = load_checkpoint(run_dir)["head"]
    in_features = next(iter(state.values())).shape[1]  # first layer's weight: (out, in)
    head = build_head(cfg.model.head, in_features)
    head.load_state_dict(state)
    return cfg, head.eval()


def load_checkpoint(run_dir: Path) -> dict:
    """The saved objects of the best epoch: "head", and "backbone" if the run fine-tuned one."""
    checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    return checkpoint["state"]
