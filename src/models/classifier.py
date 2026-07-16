"""
Classification head for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Fully-connected classification head.

    Architecture
    ------------
    AdaptiveAvgPool
        ↓
    Flatten
        ↓
    Linear
        ↓
    ReLU
        ↓
    Dropout
        ↓
    Linear
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Feature map from CNN backbone.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        x = self.pool(x)
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
        logits = self.forward(x)
        return logits.argmax(dim=1)


if __name__ == "__main__":
    dummy = torch.randn(8, 128, 4, 4)

    classifier = ClassifierHead(
        in_features=128,
        hidden_features=256,
        num_classes=10,
    )

    logits = classifier(dummy)

    predictions = classifier.predict(dummy)

    print("Input Shape       :", tuple(dummy.shape))
    print("Logits Shape      :", tuple(logits.shape))
    print("Predictions Shape :", tuple(predictions.shape))
    print("Predictions       :", predictions.tolist())
