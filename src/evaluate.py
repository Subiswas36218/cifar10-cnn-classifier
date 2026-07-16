"""
Evaluate a trained CIFAR-10 CNN model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.config import Config
from src.dataset import CIFAR10DataModule
from src.models import (
    CIFAR10CNN,
    load_model,
    move_to_device,
)


@torch.no_grad()
def evaluate() -> None:
    """
    Evaluate a saved model on the CIFAR-10 test set.
    """

    Config.create_directories()

    data_module = CIFAR10DataModule()

    test_loader = data_module.test_loader()

    model = move_to_device(CIFAR10CNN())

    load_model(
        model,
        Config.BEST_MODEL_FILE,
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0

    predictions: list[int] = []

    targets: list[int] = []

    for images, labels in tqdm(
        test_loader,
        desc="Evaluating",
    ):

        images = images.to(Config.DEVICE)

        labels = labels.to(Config.DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)

        predictions.extend(predicted.cpu().tolist())

        targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(test_loader)

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    print("\nEvaluation Results")
    print("=" * 50)

    print(f"Test Loss     : {avg_loss:.4f}")

    print(f"Accuracy      : {accuracy:.4f}")

    print("=" * 50)

    class_names = data_module.test_dataset().classes

    print("\nClassification Report\n")

    print(
        classification_report(
            targets,
            predictions,
            target_names=class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(
        targets,
        predictions,
    )

    print("Confusion Matrix\n")

    print(cm)


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
