"""
Utilities for working with PyTorch models.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.config import Config


def get_device() -> torch.device:
    """
    Return the configured computation device.
    """
    return torch.device(Config.DEVICE)


def move_to_device(
    model: nn.Module,
) -> nn.Module:
    """
    Move a model to the configured device.
    """
    return model.to(get_device())


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    Count model parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def freeze_model(
    model: nn.Module,
) -> None:
    """
    Freeze all model parameters.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_model(
    model: nn.Module,
) -> None:
    """
    Unfreeze all model parameters.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True


def save_model(
    model: nn.Module,
    path: str | Path,
) -> None:
    """
    Save only the model weights.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path,
    )


def load_model(
    model: nn.Module,
    path: str | Path,
) -> nn.Module:
    """
    Load model weights.
    """
    state_dict = torch.load(
        path,
        map_location=get_device(),
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Perform inference.
    """
    model.eval()

    images = images.to(get_device())

    logits = model(images)

    return torch.argmax(
        logits,
        dim=1,
    )


def print_model_summary(
    model: nn.Module,
) -> None:
    """
    Print a concise model summary.
    """
    total = count_parameters(model)

    trainable = count_parameters(
        model,
        trainable_only=True,
    )

    print("=" * 50)
    print(model.__class__.__name__)
    print("=" * 50)
    print(f"Device               : {get_device()}")
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print("=" * 50)


if __name__ == "__main__":
    from .cnn import CIFAR10CNN

    model = move_to_device(CIFAR10CNN())

    print_model_summary(model)

    dummy = torch.randn(2, 3, 32, 32)

    predictions = predict(model, dummy)

    print("Predictions:", predictions.tolist())
