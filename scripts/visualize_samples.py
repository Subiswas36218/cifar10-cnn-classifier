from __future__ import annotations

import argparse
from math import ceil

import matplotlib.pyplot as plt
import torch

from src.dataset import CIFAR10DataModule
from src.utils import set_seed

# CIFAR-10 normalization values used in dataset.py
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """
    Reverse normalization for visualization.
    """
    image = image.cpu()
    image = image * STD + MEAN
    image = image.clamp(0.0, 1.0)
    return image


def plot_samples(
    num_images: int,
    columns: int,
    save_path: str | None,
) -> None:
    """
    Display a grid of random CIFAR-10 images.
    """
    data_module = CIFAR10DataModule()

    loader = data_module.train_loader()

    dataset = data_module.train_dataset()

    class_names = dataset.classes

    images, labels = next(iter(loader))

    num_images = min(num_images, len(images))

    rows = ceil(num_images / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3, rows * 3),
    )

    # Flatten axes for consistent indexing
    if rows == 1 and columns == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for index in range(rows * columns):
        ax = axes[index]

        ax.axis("off")

        if index >= num_images:
            continue

        image = denormalize(images[index])

        image = image.permute(1, 2, 0).numpy()

        label = class_names[int(labels[index])]

        ax.imshow(image)

        ax.set_title(label, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CIFAR-10 samples.")

    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        help="Number of images to display.",
    )

    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns in the grid.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure.",
    )

    return parser.parse_args()


def main() -> None:
    set_seed()

    args = parse_args()

    plot_samples(
        num_images=args.num_images,
        columns=args.columns,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
