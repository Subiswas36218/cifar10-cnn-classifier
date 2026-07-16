"""
Inference script for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.dataset import CIFAR10DataModule
from src.models import (
    CIFAR10CNN,
    load_model,
    move_to_device,
)

# Same normalization used during training
_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def load_image(path: str | Path) -> torch.Tensor:
    """
    Load an image from disk and prepare it for inference.
    """
    image = Image.open(path).convert("RGB")

    tensor = _transform(image)

    return tensor.unsqueeze(0)


@torch.no_grad()
def predict(image_path: str | Path) -> None:
    """
    Predict the class of a single image.
    """

    model = move_to_device(CIFAR10CNN())

    load_model(
        model,
        Config.BEST_MODEL_FILE,
    )

    image = load_image(image_path)

    image = image.to(Config.DEVICE)

    logits = model(image)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predicted_index = int(probabilities.argmax())

    confidence = float(probabilities.max())

    class_names = CIFAR10DataModule().train_dataset().classes

    print("\nPrediction")
    print("=" * 50)

    print(f"Image      : {image_path}")

    print(f"Class      : {class_names[predicted_index]}")

    print(f"Confidence : {confidence:.4f}")

    print("\nTop-5 Predictions")

    values, indices = torch.topk(
        probabilities,
        k=5,
    )

    for score, idx in zip(
        values.squeeze(),
        indices.squeeze(),
    ):
        print(f"{class_names[int(idx)]:<12}" f"{float(score):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 image.")

    parser.add_argument(
        "image",
        type=str,
        help="Path to an image.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    predict(args.image)


if __name__ == "__main__":
    main()
