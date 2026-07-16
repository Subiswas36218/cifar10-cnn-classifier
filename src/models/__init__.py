"""
Model package for the CIFAR-10 CNN Classifier.

This package exposes the public API for all model-related
components used throughout the project.
"""

from .blocks import (
    ConvBlock,
    Flatten,
    Identity,
    initialize_weights,
)
from .classifier import ClassifierHead
from .cnn import CIFAR10CNN
from .utils import (
    count_parameters,
    freeze_model,
    get_device,
    load_model,
    move_to_device,
    predict,
    print_model_summary,
    save_model,
    unfreeze_model,
)

__all__ = [
    # Core model
    "CIFAR10CNN",
    # Building blocks
    "ConvBlock",
    "Flatten",
    "Identity",
    "ClassifierHead",
    # Initialization
    "initialize_weights",
    # Utilities
    "get_device",
    "move_to_device",
    "count_parameters",
    "freeze_model",
    "unfreeze_model",
    "save_model",
    "load_model",
    "predict",
    "print_model_summary",
]

__version__ = "1.0.0"
