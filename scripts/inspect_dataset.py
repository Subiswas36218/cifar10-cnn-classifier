from __future__ import annotations

from pathlib import Path

import torch

from src.config import Config
from src.dataset import CIFAR10DataModule
from src.utils import set_seed


def bytes_to_mb(num_bytes: int) -> float:
    """
    Convert bytes to megabytes.
    """
    return num_bytes / (1024 * 1024)


def print_header(title: str) -> None:
    """
    Print a formatted section header.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def inspect_directories() -> None:
    """
    Display important project directories.
    """
    print_header("PROJECT DIRECTORIES")

    directories: list[tuple[str, Path]] = [
        ("Project Root", Config.PROJECT_ROOT),
        ("Data Directory", Config.DATA_DIR),
        ("Model Directory", Config.MODEL_DIR),
        ("Log Directory", Config.LOG_DIR),
    ]

    for name, directory in directories:
        print(f"{name:<18}: {directory}")


def inspect_dataset() -> None:
    """
    Inspect dataset metadata.
    """
    print_header("DATASET INFORMATION")

    data_module = CIFAR10DataModule()

    train_dataset = data_module.train_dataset()
    test_dataset = data_module.test_dataset()

    print(f"Dataset Name       : {Config.DATASET_NAME}")
    print(f"Training Samples   : {len(train_dataset):,}")
    print(f"Testing Samples    : {len(test_dataset):,}")
    print(f"Number of Classes  : {Config.NUM_CLASSES}")
    print()

    print("Class Labels")

    for index, class_name in enumerate(train_dataset.classes):
        print(f"{index:>2} -> {class_name}")


def inspect_dataloaders() -> None:
    """
    Display DataLoader information.
    """
    print_header("DATALOADER")

    data_module = CIFAR10DataModule()

    train_loader = data_module.train_loader()
    test_loader = data_module.test_loader()

    print(f"Train Batches : {len(train_loader)}")
    print(f"Test Batches  : {len(test_loader)}")
    print(f"Batch Size    : {Config.BATCH_SIZE}")

    images, labels = next(iter(train_loader))

    print()
    print("First Batch")
    print(f"Images Shape  : {tuple(images.shape)}")
    print(f"Labels Shape  : {tuple(labels.shape)}")
    print(f"Images dtype  : {images.dtype}")
    print(f"Labels dtype  : {labels.dtype}")
    print(f"Device        : {Config.DEVICE}")


def inspect_tensor_statistics() -> None:
    """
    Print statistics of a batch.
    """
    print_header("BATCH STATISTICS")

    data_module = CIFAR10DataModule()

    images, labels = next(iter(data_module.train_loader()))

    print(f"Tensor Mean   : {images.mean().item():.4f}")
    print(f"Tensor Std    : {images.std().item():.4f}")
    print(f"Tensor Min    : {images.min().item():.4f}")
    print(f"Tensor Max    : {images.max().item():.4f}")

    unique_labels = torch.unique(labels)

    print(f"Unique Labels : {unique_labels.tolist()}")


def inspect_memory() -> None:
    """
    Estimate memory footprint of one batch.
    """
    print_header("MEMORY ESTIMATE")

    data_module = CIFAR10DataModule()

    images, _ = next(iter(data_module.train_loader()))

    memory = images.element_size() * images.nelement()

    print(f"Batch Memory : {bytes_to_mb(memory):.2f} MB")


def main() -> None:
    """
    Execute all inspection steps.
    """
    Config.create_directories()

    set_seed()

    inspect_directories()

    inspect_dataset()

    inspect_dataloaders()

    inspect_tensor_statistics()

    inspect_memory()

    print_header("DATASET INSPECTION COMPLETE")


if __name__ == "__main__":
    main()
