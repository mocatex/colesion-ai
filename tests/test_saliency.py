import pytest
import torch
from torch import nn

from colesion_ai.core.config import ModelCfg
from colesion_ai.models.classifier import build_backbone
from colesion_ai.models.saliency import grad_cam


@pytest.mark.parametrize(
    ("backbone", "size", "grid"), [("resnet18", 64, 2), ("dinov2_vits14", 56, 4)]
)
def test_grad_cam_gives_one_map_per_image(backbone, size, grid):
    model = build_backbone(ModelCfg(backbone=backbone, weights="random"), image_size=size).eval()
    head = nn.Linear(model.num_features, 7)
    cams = grad_cam(model, head, torch.randn(3, 3, size, size))
    assert cams.shape == (3, grid, grid)  # CNN: 1/32 of the input, ViT/14: one cell per patch
    assert cams.min() >= 0 and cams.amax((1, 2)).allclose(torch.ones(3))
