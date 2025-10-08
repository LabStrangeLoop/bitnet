"""Training script for standard and bit-quantized models."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from experiments.datasets.factory import get_dataset
from experiments.models.factory import get_model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_optimizer(model: nn.Module, opt_name: str, lr: float, momentum: float = 0.9,
                  weight_decay: float = 1e-4) -> optim.Optimizer:
    """Create optimizer."""
    if opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
    elif opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str,
                  epochs: int) -> CosineAnnealingLR | StepLR | LambdaLR:
    """Create learning rate scheduler."""
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'none':
        return LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                writer: SummaryWriter | None = None) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': acc})

        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss_batch', loss.item(), global_step)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                   best_acc: float, output_dir: Path, is_best: bool = False) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }

    # Save regular checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = output_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved with accuracy: {best_acc:.2f}%")


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_suffix = "_bit" if args.bit_version else "_standard"
    run_name = f"{args.model}{model_suffix}_{args.dataset}_seed{args.seed}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {output_dir / 'config.json'}")

    # Setup tensorboard
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(output_dir / 'tensorboard')
        print(f"Tensorboard logs: {output_dir / 'tensorboard'}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = get_dataset(args.dataset, split='train', root=args.data_dir)
    test_dataset = get_dataset(args.dataset, split='test', root=args.data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(cast(list, train_dataset))}")
    print(f"Test samples: {len(cast(list, test_dataset))}")

    # Create model
    print(f"\nCreating {args.model} ({'bit-quantized' if args.bit_version else 'standard'})...")
    num_classes = 10 if args.dataset in ['cifar10', 'mnist'] else 100
    if args.dataset == 'tiny_imagenet':
        num_classes = 200

    model = get_model(
        args.model,
        num_classes=num_classes,
        bit_version=args.bit_version,
        pretrained=args.pretrained
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.momentum, args.weight_decay)
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_acc = 0.0
    history: dict[str, list[float]] = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Save results
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/train_acc', train_acc, epoch)
            writer.add_scalar('epoch/test_loss', test_loss, epoch)
            writer.add_scalar('epoch/test_acc', test_acc, epoch)
            writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        print("\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        # Save checkpoint
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, best_acc, output_dir, is_best)

    # Save final results
    results = {
        'best_acc': best_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'history': history
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_dir}")

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification models')

    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Model architecture (from timm)')
    parser.add_argument('--bit-version', action='store_true',
                       help='Use 1.58-bit quantized version')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')

    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='results/raw',
                       help='Output directory')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Enable tensorboard logging')

    args = parser.parse_args()
    main(args)
