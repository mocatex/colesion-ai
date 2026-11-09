"""Grad-CAM (Selvaraju et al., 2017): which image regions pushed the model to its top-1 class.

At one layer of the backbone, weight each feature map by how much the top-1 logit grows with it
(its average gradient), sum them, and keep the positive part.
"""

import torch
import torch.nn.functional as F
from torch import nn


def cam_layer(backbone: nn.Module) -> nn.Module:
    """The layer Grad-CAM looks at. CNN: the last residual block. ViT: the input of the last
    attention (after the last block, the patch tokens no longer reach the class token)."""
    return backbone.blocks[-1].norm1 if hasattr(backbone, "blocks") else backbone.layer4[-1]


def grad_cam(backbone: nn.Module, head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Maps of shape (batch, h, w) in [0, 1], on the layer's grid (e.g. 16 x 16 for a ViT/14
    at 224 px). Scale them to any image size with `F.interpolate`."""
    saved = {}
    hook = cam_layer(backbone).register_forward_hook(lambda m, i, out: saved.update(act=out))
    with torch.enable_grad():
        logits = head(backbone(x))
        saved["act"].retain_grad()
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    hook.remove()

    act, grad = saved["act"], saved["act"].grad
    if act.ndim == 3:  # ViT tokens (batch, prefix + h*w, channels) -> (batch, channels, h, w)
        side = int((act.shape[1] - backbone.num_prefix_tokens) ** 0.5)
        act, grad = (
            t[:, backbone.num_prefix_tokens :].mT.unflatten(2, (side, side)) for t in (act, grad)
        )
    weights = grad.mean((2, 3), keepdim=True)  # one weight per feature map
    cam = F.relu((weights * act).sum(1)).detach()
    return cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-8)
