"""
CNN architecture for CIFAR-10 image classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, initialize_weights
from .classifier import ClassifierHead


class CIFAR10CNN(nn.Module):
    """
    Compact CNN for CIFAR-10.

    Input
        3 × 32 × 32

    Output
        10 class logits
    """

    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.classifier = ClassifierHead(
            in_features=128,
            hidden_features=256,
            num_classes=num_classes,
        )

        self.apply(initialize_weights)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard forward pass.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return predicted class indices.
        """
        self.eval()

        logits = self.forward(x)

        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return class probabilities.
        """
        self.eval()

        logits = self.forward(x)

        return torch.softmax(logits, dim=1)

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return CNN feature maps before classification.
        """
        return self.features(x)


if __name__ == "__main__":

    dummy = torch.randn(4, 3, 32, 32)

    model = CIFAR10CNN()

    logits = model(dummy)

    probabilities = model.predict_proba(dummy)

    predictions = model.predict(dummy)

    features = model.extract_features(dummy)

    print("Model Summary")
    print("-" * 40)

    print("Input Shape        :", tuple(dummy.shape))

    print("Feature Shape      :", tuple(features.shape))

    print("Logits Shape       :", tuple(logits.shape))

    print("Probability Shape  :", tuple(probabilities.shape))

    print("Prediction Shape   :", tuple(predictions.shape))

    total_params = sum(parameter.numel() for parameter in model.parameters())

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("Total Parameters   :", total_params)

    print("Trainable Params   :", trainable_params)
