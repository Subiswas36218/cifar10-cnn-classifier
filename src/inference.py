"""
Shared inference utilities for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

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

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)

_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=_MEAN,
            std=_STD,
        ),
    ]
)


class InferenceEngine:
    """
    Loads a trained model once and performs inference.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
    ) -> None:

        if model_path is None:
            model_path = Config.BEST_MODEL_FILE

        self.device = torch.device(Config.DEVICE)

        self.model = move_to_device(CIFAR10CNN())

        load_model(
            self.model,
            model_path,
        )

        self.model.eval()

        self.class_names = CIFAR10DataModule().train_dataset().classes

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Convert PIL image into a model input tensor.
        """

        image = image.convert("RGB")

        tensor = _transform(image)

        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> dict:
        """
        Predict the class of an image.
        """

        inputs = self.preprocess(image)

        logits = self.model(inputs)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        values, indices = torch.topk(
            probabilities,
            k=min(top_k, len(self.class_names)),
        )

        predictions = []

        for score, index in zip(
            values.squeeze(0),
            indices.squeeze(0),
        ):
            predictions.append(
                {
                    "class": self.class_names[int(index)],
                    "confidence": round(
                        float(score),
                        4,
                    ),
                }
            )

        return {
            "prediction": predictions[0]["class"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
        }


if __name__ == "__main__":

    image_path = Path("examples/airplane.jpg")

    if image_path.exists():

        engine = InferenceEngine()

        image = Image.open(image_path)

        result = engine.predict(image)

        print(result)

    else:
        print(
            "Example image not found.\n" "Create examples/airplane.jpg to run the demo."
        )