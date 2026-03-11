"""Debug script to verify TTQ parameters get gradients during training."""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.config import DATASET_NUM_CLASSES
from experiments.models.factory import get_model


def check_ttq_gradients(model: nn.Module) -> tuple[int, int]:
    """Check how many TTQ parameters have gradients.

    Returns:
        (ttq_params_with_grad, total_ttq_params)
    """
    ttq_params_with_grad = 0
    total_ttq_params = 0

    for name, module in model.named_modules():
        if "TTQ" in type(module).__name__:
            # Check wp, wn, delta parameters
            for param_name in ["wp", "wn", "delta"]:
                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    total_ttq_params += 1
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        ttq_params_with_grad += 1
                        print(f"  ✓ {name}.{param_name}: grad_norm={param.grad.norm().item():.6f}")
                    else:
                        print(f"  ❌ {name}.{param_name}: NO GRADIENT")

    return ttq_params_with_grad, total_ttq_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--use-cifar-stem", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TTQ Gradient Flow Verification")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    num_classes = DATASET_NUM_CLASSES.get(args.dataset, 10)
    model = get_model(
        args.model,
        num_classes,
        bit_version=False,
        pretrained=False,
        skip_layers=set(),
        use_cifar_stem=args.use_cifar_stem,
        is_ttq=True,
    )
    model = model.to(device)

    # Count TTQ layers
    ttq_count = sum(1 for m in model.modules() if "TTQ" in type(m).__name__)
    print(f"Model: {args.model} with {ttq_count} TTQ layers\n")

    # Load a small batch of data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Get one batch
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    print(f"Data batch: {inputs.shape}, targets: {targets.shape}\n")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Single training step
    print("Running single training step...\n")
    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    print(f"Loss: {loss.item():.4f}")
    print(f"Output range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]")

    if torch.isnan(loss):
        print("❌ ERROR: NaN loss detected!")
        return

    loss.backward()

    # Check gradients
    print("\nChecking TTQ parameter gradients:\n")
    with_grad, total = check_ttq_gradients(model)

    print("\n" + "=" * 60)
    print(f"Result: {with_grad}/{total} TTQ parameters have gradients")

    if with_grad == 0:
        print("❌ CRITICAL: NO TTQ parameters are getting gradients!")
        print("   This means the quantization is broken or frozen")
    elif with_grad < total:
        print(f"⚠️  WARNING: Only {with_grad}/{total} TTQ parameters have gradients")
        print("   Some layers may be frozen or not in the computation graph")
    else:
        print("✅ SUCCESS: All TTQ parameters are getting gradients")

    # Check if weight parameters also have gradients
    weight_grads = sum(
        1 for n, p in model.named_parameters() if "weight" in n and p.grad is not None and p.grad.abs().sum() > 0
    )
    total_weights = sum(1 for n, p in model.named_parameters() if "weight" in n)
    print(f"\nWeight parameters with gradients: {weight_grads}/{total_weights}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
