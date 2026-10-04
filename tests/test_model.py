import torch

from src.model import build_model, count_trainable, freeze_backbone


def test_build_efficientnet_head():
    m = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    x = torch.randn(1, 3, 256, 704)
    y = m(x)
    assert y.shape == (1, 2)


def test_freeze_backbone_reduces_trainable():
    m = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    before = count_trainable(m)
    freeze_backbone(m, "efficientnet_b0")
    after = count_trainable(m)
    assert after < before