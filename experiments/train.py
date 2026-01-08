"""Training script for standard and bit-quantized models."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from experiments.datasets.factory import get_dataset
from experiments.models.factory import get_model

DATASET_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_scheduler(
    optimizer: optim.Optimizer, name: str, epochs: int, warmup_epochs: int = 0
) -> SequentialLR | CosineAnnealingLR | StepLR | LambdaLR:
    def make_main_scheduler() -> CosineAnnealingLR | StepLR | LambdaLR:
        main_epochs = epochs - warmup_epochs
        schedulers = {
            "cosine": lambda: CosineAnnealingLR(optimizer, T_max=main_epochs),
            "step": lambda: StepLR(optimizer, step_size=30, gamma=0.1),
            "none": lambda: LambdaLR(optimizer, lambda _: 1.0),
        }
        if name not in schedulers:
            raise ValueError(f"Unknown scheduler: {name}")
        return schedulers[name]()

    if warmup_epochs > 0:
        warmup = LambdaLR(optimizer, lambda e: (e + 1) / warmup_epochs)
        main = make_main_scheduler()
        return SequentialLR(optimizer, [warmup, main], milestones=[warmup_epochs])
    return make_main_scheduler()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    best_acc: float,
    history: dict,
) -> None:
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "history": history,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pth")


def load_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
) -> tuple[int, float, dict]:
    checkpoint_path = output_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        return 0, 0.0, {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Resumed from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint["best_acc"], checkpoint["history"]


def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_set = get_dataset(config["dataset"], "train", root=config["data_dir"])
    test_set = get_dataset(config["dataset"], "test", root=config["data_dir"])
    train_loader = DataLoader(
        train_set,
        config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        config["batch_size"] * 2,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Model
    num_classes = DATASET_NUM_CLASSES.get(config["dataset"], 10)
    model = get_model(config["model"], num_classes, config["bit_version"], config["pretrained"])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
        nesterov=True,
    )
    scheduler = get_scheduler(
        optimizer, config["scheduler"], config["epochs"], config.get("warmup_epochs", 0)
    )

    # Output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Resume from checkpoint if exists
    start_epoch, best_acc, history = load_checkpoint(output_dir, model, optimizer, scheduler)

    # TensorBoard (purge old logs if starting fresh)
    tb_dir = output_dir / "tensorboard"
    if start_epoch == 0 and tb_dir.exists():
        import shutil
        shutil.rmtree(tb_dir)
    writer = SummaryWriter(tb_dir) if config.get("tensorboard") else None

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if writer:
            for name, val in [
                ("train_loss", train_loss),
                ("train_acc", train_acc),
                ("test_loss", test_loss),
                ("test_acc", test_acc),
            ]:
                writer.add_scalar(f"epoch/{name}", val, epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
            f"Test: {test_loss:.4f} / {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state, output_dir / "best_model.pth")

        # Save checkpoint every epoch
        save_checkpoint(output_dir, epoch, model, optimizer, scheduler, best_acc, history)

    if writer:
        writer.close()

    results = {
        "best_acc": best_acc,
        "final_test_acc": test_acc,
        "history": history,
        "config": config,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Clean up checkpoint after successful completion
    (output_dir / "checkpoint.pth").unlink(missing_ok=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--dataset", default="cifar10", choices=list(DATASET_NUM_CLASSES.keys()))
    parser.add_argument("--bit-version", action="store_true")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    args = parser.parse_args()

    config = vars(args)
    version = "bit" if args.bit_version else "std"
    config["output_dir"] = f"{args.output_dir}/{args.model}_{version}_{args.dataset}_s{args.seed}"

    train(config)


if __name__ == "__main__":
    main()
