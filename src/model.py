"""Model factory. Keeps architecture name in one place so training,
evaluation, and the API all rebuild the identical network."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    arch = architecture.lower()
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def freeze_backbone(model: nn.Module, architecture: str) -> None:
    arch = architecture.lower()
    if arch == "efficientnet_b0":
        for p in model.features.parameters():
            p.requires_grad = False
    elif arch == "resnet18":
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)