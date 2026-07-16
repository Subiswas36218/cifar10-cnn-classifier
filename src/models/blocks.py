"""
Reusable neural network building blocks for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize model weights.

    - Conv2d: Kaiming Normal
    - Linear: Xavier Uniform
    - BatchNorm2d: weight=1, bias=0
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    """
    Standard convolutional block.

    Conv2D
        ↓
    BatchNorm
        ↓
    ReLU
        ↓
    MaxPool
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))

        self.block = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class Flatten(nn.Module):
    """
    Flatten feature maps before fully-connected layers.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)


class Identity(nn.Module):
    """
    Identity layer.
    Useful for experimentation.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x


if __name__ == "__main__":
    dummy = torch.randn(4, 3, 32, 32)

    block = ConvBlock(
        in_channels=3,
        out_channels=32,
    )

    block.apply(initialize_weights)

    output = block(dummy)

    print("Input :", tuple(dummy.shape))
    print("Output:", tuple(output.shape))
