from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config


def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """
    Make experiments reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Keeps track of running averages.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.average = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.total += value * n
        self.count += n

        if self.count > 0:
            self.average = self.total / self.count

    def __str__(self) -> str:
        return f"{self.average:.4f}"


def accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute batch accuracy.
    """

    predictions = outputs.argmax(dim=1)

    correct = (predictions == labels).sum().item()

    return correct / labels.size(0)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path | str,
) -> None:
    """
    Save training checkpoint.
    """

    path = Path(filepath)

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    filepath: Path | str,
) -> dict[str, Any]:
    """
    Load a saved checkpoint.
    """

    checkpoint = torch.load(
        filepath,
        map_location=Config.DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def count_parameters(
    model: torch.nn.Module,
) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class History:
    """
    Stores training history.
    """

    def __init__(self) -> None:

        self.train_loss: list[float] = []

        self.train_accuracy: list[float] = []

        self.validation_loss: list[float] = []

        self.validation_accuracy: list[float] = []

    def add_train(
        self,
        loss: float,
        accuracy_value: float,
    ) -> None:

        self.train_loss.append(loss)

        self.train_accuracy.append(accuracy_value)

    def add_validation(
        self,
        loss: float,
        accuracy_value: float,
    ) -> None:

        self.validation_loss.append(loss)

        self.validation_accuracy.append(accuracy_value)

    def as_dict(self) -> dict[str, list[float]]:

        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
        }


def print_epoch_summary(
    epoch: int,
    epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    """
    Pretty epoch summary.
    """

    print(
        (
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
    )
