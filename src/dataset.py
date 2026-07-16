from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from .config import Config


class CIFAR10DataModule:
    """
    Handles downloading, preprocessing,
    and creating PyTorch DataLoaders.
    """

    def __init__(self) -> None:
        Config.create_directories()

        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32,
                    padding=4,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    def train_dataset(self) -> datasets.CIFAR10:
        """
        Returns training dataset.
        """

        return datasets.CIFAR10(
            root=Config.DATA_DIR,
            train=True,
            download=True,
            transform=self.train_transform,
        )

    def test_dataset(self) -> datasets.CIFAR10:
        """
        Returns testing dataset.
        """

        return datasets.CIFAR10(
            root=Config.DATA_DIR,
            train=False,
            download=True,
            transform=self.test_transform,
        )

    def train_loader(self) -> DataLoader:
        """
        Training DataLoader.
        """

        return DataLoader(
            dataset=self.train_dataset(),
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        """
        Testing DataLoader.
        """

        return DataLoader(
            dataset=self.test_dataset(),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    def dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns both loaders.
        """

        return (
            self.train_loader(),
            self.test_loader(),
        )


if __name__ == "__main__":
    data = CIFAR10DataModule()

    train_loader, test_loader = data.dataloaders()

    print(f"Training batches : {len(train_loader)}")

    print(f"Testing batches  : {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print(images.shape)

    print(labels.shape)
