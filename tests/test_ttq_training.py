"""Local training test for TTQ - test different beta configurations."""

import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import both TTQ versions
from bitnet.nn.ttq_conv2d import TTQConv2d
from bitnet.nn.ttq_linear import TTQLinear


def create_small_resnet_ttq(num_classes=10, config="A"):
    """Create a tiny ResNet with TTQ layers for quick testing.

    Config options:
    - A: beta = 1.0 (current implementation)
    - B: beta = (wp+wn)/2 (original attempt)
    - C: beta = weight.abs().mean() (BitNet style)
    - D: Pure TTQ (no activation quant)
    """

    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, config="A"):
            super().__init__()
            self.config = config

            if config == "D":
                # Pure TTQ - no activation quantization
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            else:
                # Modified TTQ with activation quantization
                self.conv1 = TTQConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                self.conv2 = TTQConv2d(out_channels, out_channels, 3, padding=1, bias=False)

                # Store config for beta adjustment
                self.conv1._beta_config = config
                self.conv2._beta_config = config

            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
                    if config == "D"
                    else TTQConv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                if config != "D":
                    self.shortcut[0]._beta_config = config

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += self.shortcut(x)
            out = self.relu(out)
            return out

    class TinyResNet(nn.Module):
        def __init__(self, num_classes, config):
            super().__init__()
            self.config = config

            # CIFAR-adapted stem
            if config == "D":
                self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
            else:
                self.conv1 = TTQConv2d(3, 16, 3, stride=1, padding=1, bias=False)
                self.conv1._beta_config = config

            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)

            # Only 2 blocks per layer for speed
            self.layer1 = self._make_layer(16, 16, 2, 1, config)
            self.layer2 = self._make_layer(16, 32, 2, 2, config)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            if config == "D":
                self.fc = nn.Linear(32, num_classes)
            else:
                self.fc = TTQLinear(32, num_classes)
                self.fc._beta_config = config

        def _make_layer(self, in_channels, out_channels, num_blocks, stride, config):
            layers = []
            layers.append(BasicBlock(in_channels, out_channels, stride, config))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_channels, out_channels, 1, config))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return TinyResNet(num_classes, config)


def adjust_beta_in_forward(module):
    """Monkey-patch TTQ layers to use different beta configs."""
    if hasattr(module, "_beta_config") and hasattr(module, "forward"):
        config = module._beta_config

        if config == "B":
            # Original: beta = (wp+wn)/2
            def new_forward(x):
                import torch.nn.functional as f

                from bitnet.nn.quantization import dequantize, quantize_activations
                from bitnet.nn.ttq_quantization import ttq_quantize

                if isinstance(module, TTQLinear):
                    x = f.layer_norm(x, x.shape[1:])
                    x_quant, gamma = quantize_activations(x, module.num_bits)
                    w_quant, wp_pos, wn_pos = ttq_quantize(module.weight, module.wp, module.wn, module.delta)
                    beta = (wp_pos + wn_pos) / 2  # Config B
                    out = f.linear(x_quant, w_quant, module.bias)
                    return dequantize(out, gamma, beta, module.num_bits)
                else:  # TTQConv2d
                    x = f.layer_norm(x, x.shape[1:])
                    x_quant, gamma = quantize_activations(x, module.num_bits)
                    w_quant, wp_pos, wn_pos = ttq_quantize(module.weight, module.wp, module.wn, module.delta)
                    beta = (wp_pos + wn_pos) / 2  # Config B
                    out = f.conv2d(
                        x_quant, w_quant, module.bias, module.stride, module.padding, module.dilation, module.groups
                    )
                    return dequantize(out, gamma, beta, module.num_bits)

            module.forward = new_forward

        elif config == "C":
            # BitNet style: beta = weight.abs().mean()
            def new_forward(x):
                import torch.nn.functional as f

                from bitnet.nn.quantization import dequantize, quantize_activations
                from bitnet.nn.ttq_quantization import ttq_quantize

                if isinstance(module, TTQLinear):
                    x = f.layer_norm(x, x.shape[1:])
                    x_quant, gamma = quantize_activations(x, module.num_bits)
                    w_quant, wp_pos, wn_pos = ttq_quantize(module.weight, module.wp, module.wn, module.delta)
                    beta = module.weight.abs().mean()  # Config C - BitNet style
                    out = f.linear(x_quant, w_quant, module.bias)
                    return dequantize(out, gamma, beta, module.num_bits)
                else:  # TTQConv2d
                    x = f.layer_norm(x, x.shape[1:])
                    x_quant, gamma = quantize_activations(x, module.num_bits)
                    w_quant, wp_pos, wn_pos = ttq_quantize(module.weight, module.wp, module.wn, module.delta)
                    beta = module.weight.abs().mean()  # Config C - BitNet style
                    out = f.conv2d(
                        x_quant, w_quant, module.bias, module.stride, module.padding, module.dilation, module.groups
                    )
                    return dequantize(out, gamma, beta, module.num_bits)

            module.forward = new_forward


def train_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["A", "B", "C", "D"],
        help="Beta configuration: A=1.0, B=(wp+wn)/2, C=weight.abs().mean(), D=pure_ttq",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Testing TTQ Config {args.config}")
    print(f"{'='*60}\n")

    # CIFAR-10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_small_resnet_ttq(num_classes=10, config=args.config).to(device)

    # Apply beta config adjustments
    if args.config in ["B", "C"]:
        model.apply(adjust_beta_in_forward)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print(f"\n[EPOCH {epoch+1}] NaN loss detected at batch {batch_idx}!")
                return

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] Batch {batch_idx}/{len(trainloader)}: "
                    f"Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%"
                )

        train_acc = 100.0 * correct / total

        # Test
        model.eval()
        test_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total

        print(
            f"\n[EPOCH {epoch+1}/{args.epochs}] "
            f"Train: Loss={train_loss/len(trainloader):.4f}, Acc={train_acc:.2f}% | "
            f"Test: Loss={test_loss/len(testloader):.4f}, Acc={test_acc:.2f}%\n"
        )

        scheduler.step()

    print(f"\n{'='*60}")
    print(f"Config {args.config} Final Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_test()
