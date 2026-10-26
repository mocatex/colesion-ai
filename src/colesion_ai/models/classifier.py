"""The classifier: a pretrained backbone (image -> feature vector) + a head (features -> 7 scores)."""

import timm
from torch import nn

from colesion_ai.core.config import ModelCfg


def build_backbone(name: str, pretrained: bool, image_size: int) -> nn.Module:
    """`model.backbone` -> a timm model without its own classification layer (`num_classes=0`).
    It returns `.num_features` numbers per image. A new backbone is one more `if` here."""
    if name == "resnet18":
        return timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
    if name == "dinov2_vits14":
        # DINOv2 ViT-S/14 (Meta, 2023). timm's default input is 518 px, so pass the size we use.
        return timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
    raise ValueError(f"unknown model.backbone {name!r}: add it to build_backbone()")


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone
        backbone.requires_grad_(not freeze_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # frozen means frozen: no batch-norm or dropout changes either
        return self

    def forward(self, x):
        return self.head(self.backbone(x))


def build_classifier(cfg: ModelCfg, num_classes: int, image_size: int) -> Classifier:
    if cfg.freeze not in ("all", "none"):
        raise ValueError(f"model.freeze must be 'all' or 'none', got {cfg.freeze!r}")
    if cfg.head != "linear":
        raise ValueError(f"model.head must be 'linear' (more come in phase 2), got {cfg.head!r}")
    backbone = build_backbone(cfg.backbone, cfg.pretrained, image_size)
    head = nn.Linear(backbone.num_features, num_classes)
    return Classifier(backbone, head, freeze_backbone=cfg.freeze == "all")
