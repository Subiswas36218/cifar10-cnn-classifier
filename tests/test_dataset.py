from __future__ import annotations

import torch
from torchvision.datasets import CIFAR10

from src.config import Config
from src.dataset import CIFAR10DataModule


def test_directories_created() -> None:
    """
    Verify required project directories exist.
    """

    Config.create_directories()

    assert Config.DATA_DIR.exists()

    assert Config.MODEL_DIR.exists()

    assert Config.LOG_DIR.exists()


def test_train_dataset() -> None:
    """
    Verify the training dataset loads correctly.
    """

    data = CIFAR10DataModule()

    dataset = data.train_dataset()

    assert isinstance(dataset, CIFAR10)

    assert len(dataset) == 50000


def test_test_dataset() -> None:
    """
    Verify the testing dataset loads correctly.
    """

    data = CIFAR10DataModule()

    dataset = data.test_dataset()

    assert isinstance(dataset, CIFAR10)

    assert len(dataset) == 10000


def test_train_loader_batch_size() -> None:
    """
    Ensure DataLoader returns the configured batch size.
    """

    data = CIFAR10DataModule()

    loader = data.train_loader()

    images, labels = next(iter(loader))

    assert images.shape[0] == Config.BATCH_SIZE

    assert labels.shape[0] == Config.BATCH_SIZE


def test_image_dimensions() -> None:
    """
    Verify image dimensions.
    """

    data = CIFAR10DataModule()

    loader = data.train_loader()

    images, _ = next(iter(loader))

    assert images.shape[1] == 3

    assert images.shape[2] == 32

    assert images.shape[3] == 32


def test_label_dtype() -> None:
    """
    Labels should be integer tensors.
    """

    data = CIFAR10DataModule()

    loader = data.train_loader()

    _, labels = next(iter(loader))

    assert labels.dtype == torch.int64


def test_image_dtype() -> None:
    """
    Images should be float tensors.
    """

    data = CIFAR10DataModule()

    loader = data.train_loader()

    images, _ = next(iter(loader))

    assert images.dtype == torch.float32


def test_batch_is_tensor() -> None:
    """
    Images must be torch tensors.
    """

    data = CIFAR10DataModule()

    loader = data.train_loader()

    images, labels = next(iter(loader))

    assert isinstance(images, torch.Tensor)

    assert isinstance(labels, torch.Tensor)


def test_number_of_classes() -> None:
    """
    CIFAR-10 contains exactly ten classes.
    """

    data = CIFAR10DataModule()

    dataset = data.train_dataset()

    assert len(dataset.classes) == 10


def test_class_names() -> None:
    """
    Verify the expected class labels.
    """

    expected = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    data = CIFAR10DataModule()

    dataset = data.train_dataset()

    assert dataset.classes == expected


def test_test_loader_not_empty() -> None:
    """
    Test loader should contain at least one batch.
    """

    data = CIFAR10DataModule()

    loader = data.test_loader()

    assert len(loader) > 0


def test_train_loader_not_empty() -> None:
    """
    Train loader should contain at least one batch.
    """

    data = CIFAR10DataModule()

    loader = data.train_loader()

    assert len(loader) > 0


def test_device_available() -> None:
    """
    Config should always expose a valid device.
    """

    assert Config.DEVICE in {
        "cpu",
        "cuda",
        "mps",
    }


def test_random_sample() -> None:
    """
    Verify one sample can be retrieved.
    """

    data = CIFAR10DataModule()

    dataset = data.train_dataset()

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)

    assert isinstance(label, int)


def test_dataset_name() -> None:
    """
    Dataset name should remain unchanged.
    """

    assert Config.DATASET_NAME == "CIFAR10"
