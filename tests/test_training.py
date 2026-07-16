"""
Smoke tests for the training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

from src.models import (
    CIFAR10CNN,
    count_parameters,
)
from src.utils import (
    accuracy,
    save_checkpoint,
)


def test_model_forward() -> None:
    """
    Model should return logits with shape
    (batch_size, num_classes).
    """

    model = CIFAR10CNN()

    images = torch.randn(8, 3, 32, 32)

    outputs = model(images)

    assert outputs.shape == (8, 10)


def test_loss_computation() -> None:
    """
    CrossEntropyLoss should produce a scalar.
    """

    model = CIFAR10CNN()

    criterion = nn.CrossEntropyLoss()

    images = torch.randn(4, 3, 32, 32)

    labels = torch.randint(
        0,
        10,
        (4,),
    )

    outputs = model(images)

    loss = criterion(outputs, labels)

    assert loss.ndim == 0

    assert loss.item() > 0


def test_backward_pass() -> None:
    """
    Gradients should be computed.
    """

    model = CIFAR10CNN()

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    images = torch.randn(2, 3, 32, 32)

    labels = torch.randint(
        0,
        10,
        (2,),
    )

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]

    assert any(gradient is not None for gradient in gradients)


def test_optimizer_step() -> None:
    """
    Optimizer should execute without error.
    """

    model = CIFAR10CNN()

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    images = torch.randn(2, 3, 32, 32)

    labels = torch.randint(
        0,
        10,
        (2,),
    )

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    assert True


def test_accuracy_function() -> None:
    """
    Accuracy helper should return a value
    between 0 and 1.
    """

    outputs = torch.randn(
        16,
        10,
    )

    labels = torch.randint(
        0,
        10,
        (16,),
    )

    score = accuracy(
        outputs,
        labels,
    )

    assert 0.0 <= score <= 1.0


def test_parameter_count() -> None:
    """
    Model should contain trainable parameters.
    """

    model = CIFAR10CNN()

    assert count_parameters(model) > 0


def test_checkpoint_save(
    tmp_path,
) -> None:
    """
    Verify checkpoint saving.
    """

    model = CIFAR10CNN()

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint = tmp_path / "checkpoint.pth"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        loss=0.5,
        filepath=checkpoint,
    )

    assert checkpoint.exists()
