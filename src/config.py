from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Config:
    """
    Global project configuration.
    """

    # ------------------------------------------------------------------
    # Project directories
    # ------------------------------------------------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = PROJECT_ROOT / "data"

    MODEL_DIR: Path = PROJECT_ROOT / "models"

    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    DATASET_NAME: str = "CIFAR10"

    IMAGE_SIZE: int = 32

    NUM_CLASSES: int = 10

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    BATCH_SIZE: int = 64

    NUM_WORKERS: int = 2

    EPOCHS: int = 10

    LEARNING_RATE: float = 1e-3

    WEIGHT_DECAY: float = 1e-4

    RANDOM_SEED: int = 42

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    DEVICE: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    MODEL_FILE: Path = MODEL_DIR / "cnn_model.pth"

    BEST_MODEL_FILE: Path = MODEL_DIR / "best_model.pth"

    # ------------------------------------------------------------------
    # Create folders automatically
    # ------------------------------------------------------------------
    @staticmethod
    def create_directories() -> None:
        Config.DATA_DIR.mkdir(exist_ok=True)

        Config.MODEL_DIR.mkdir(exist_ok=True)

        Config.LOG_DIR.mkdir(exist_ok=True)
