"""
Training script for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import Config
from src.dataset import CIFAR10DataModule
from src.models import (
    CIFAR10CNN,
    move_to_device,
    print_model_summary,
    save_model,
)
from src.utils import (
    AverageMeter,
    History,
    accuracy,
    print_epoch_summary,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """
    Train for one epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

    return loss_meter.average, acc_meter.average


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Evaluate the model.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        batch_acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

    return loss_meter.average, acc_meter.average


def main() -> None:
    """
    Train the CNN.
    """
    Config.create_directories()

    set_seed()

    data = CIFAR10DataModule()

    train_loader, test_loader = data.dataloaders()

    model = move_to_device(CIFAR10CNN())

    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    history = History()

    best_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(1, Config.EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = evaluate(
            model,
            test_loader,
            criterion,
        )

        history.add_train(train_loss, train_acc)
        history.add_validation(val_loss, val_acc)

        print_epoch_summary(
            epoch,
            Config.EPOCHS,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc

            save_model(
                model,
                Config.BEST_MODEL_FILE,
            )

    save_model(
        model,
        Config.MODEL_FILE,
    )

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Final model saved to: {Config.MODEL_FILE}")
    print(f"Best model saved to : {Config.BEST_MODEL_FILE}")


if __name__ == "__main__":
    main()
