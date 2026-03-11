"""Debug script to verify TTQ layers are being used."""

import argparse

import torch

from experiments.config import DATASET_NUM_CLASSES
from experiments.models.factory import get_model


def print_model_layers(model: torch.nn.Module, prefix: str = "") -> None:
    """Recursively print all layers in the model with their types."""
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"{full_name}: {type(module).__name__}")

        # Check if it's a TTQ or BitNet layer
        if "TTQ" in type(module).__name__:
            print(f"  ✓ TTQ layer detected: {type(module).__name__}")
        elif "Bit" in type(module).__name__:
            print(f"  ⚠ BitNet layer detected: {type(module).__name__}")

        # Recurse into children
        if len(list(module.children())) > 0:
            print_model_layers(module, full_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--ttq-version", action="store_true")
    parser.add_argument("--bit-version", action="store_true")
    parser.add_argument("--use-cifar-stem", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"TTQ Version: {args.ttq_version}")
    print(f"Bit Version: {args.bit_version}")
    print(f"CIFAR Stem: {args.use_cifar_stem}")
    print("=" * 60 + "\n")

    num_classes = DATASET_NUM_CLASSES.get(args.dataset, 10)
    model = get_model(
        args.model,
        num_classes,
        bit_version=args.bit_version,
        pretrained=False,
        skip_layers=set(),
        use_cifar_stem=args.use_cifar_stem,
        is_ttq=args.ttq_version,
    )

    print("\nModel Layer Structure:")
    print("-" * 60)
    print_model_layers(model)
    print("-" * 60)

    # Count layer types
    ttq_count = 0
    bit_count = 0
    linear_count = 0
    conv_count = 0

    for module in model.modules():
        if "TTQ" in type(module).__name__:
            ttq_count += 1
        elif "Bit" in type(module).__name__:
            bit_count += 1
        elif type(module).__name__ == "Linear":
            linear_count += 1
        elif type(module).__name__ == "Conv2d":
            conv_count += 1

    print("\nLayer Type Summary:")
    print(f"  TTQLinear/TTQConv2d: {ttq_count}")
    print(f"  BitLinear/BitConv2d: {bit_count}")
    print(f"  Standard Linear: {linear_count}")
    print(f"  Standard Conv2d: {conv_count}")
    print()

    if args.ttq_version and ttq_count == 0:
        print("❌ ERROR: --ttq-version specified but NO TTQ layers found!")
    elif args.ttq_version and ttq_count > 0:
        print(f"✅ SUCCESS: {ttq_count} TTQ layers found as expected")
    elif args.bit_version and bit_count == 0:
        print("❌ ERROR: --bit-version specified but NO BitNet layers found!")
    elif args.bit_version and bit_count > 0:
        print(f"✅ SUCCESS: {bit_count} BitNet layers found as expected")
    else:
        print("✅ Standard model (no quantization)")


if __name__ == "__main__":
    main()
