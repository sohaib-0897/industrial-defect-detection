from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.common import CLASS_NAMES


def build_model(pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != list(CLASS_NAMES) or checkpoint["architecture"] != "resnet18":
        raise ValueError("Unsupported architecture or class mapping in checkpoint")
    if checkpoint["preprocessing"] != "letterbox224-imagenet-v1":
        raise ValueError("Unsupported preprocessing contract")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
