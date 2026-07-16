from src.models import (
    CIFAR10CNN,
    count_parameters,
    get_device,
    print_model_summary,
)

model = CIFAR10CNN()

print_model_summary(model)

print(f"Device: {get_device()}")

print(f"Parameters: {count_parameters(model):,}")
