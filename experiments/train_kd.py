"""Training script with Knowledge Distillation for BitNet models."""

import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from experiments.config import (
    ABLATION_SKIP_LAYERS,
    DATASET_NUM_CLASSES,
    DEFAULTS,
    AblationMode,
    TrainConfig,
    Version,
)
from experiments.datasets.factory import AUGMENT_CHOICES, get_dataset
from experiments.models.factory import get_model
from experiments.paths import ExperimentType, get_experiment_dir, should_skip_experiment
from experiments.training import checkpoint, logging_config
from experiments.training.kd_loss import KDLoss
from experiments.training.loops import evaluate, get_scheduler

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_teacher(model_name: str, num_classes: int, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load pre-trained FP32 teacher model."""
    teacher = get_model(model_name, num_classes, bit_version=False, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    teacher.load_state_dict(state_dict)
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def train_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    criterion: KDLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    quiet: bool = False,
) -> tuple[float, float, dict[str, float]]:
    """Run one training epoch with knowledge distillation."""
    student.train()
    total_loss, correct, total = 0.0, 0, 0
    loss_accum = {"hard_loss": 0.0, "soft_loss": 0.0}

    iterator = loader if quiet else tqdm(loader, desc="Training (KD)", leave=False)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)

        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        # Forward pass student
        optimizer.zero_grad()
        student_logits = student(inputs)
        loss, metrics = criterion(student_logits, teacher_logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += student_logits.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

        for key in loss_accum:
            loss_accum[key] += metrics[key] * inputs.size(0)

    avg_metrics = {k: v / total for k, v in loss_accum.items()}
    return total_loss / total, 100.0 * correct / total, avg_metrics


def train_kd(config: TrainConfig, teacher_path: Path, temperature: float, alpha: float) -> dict:
    output_dir = Path(config.output_dir)
    results_path = output_dir / "results.json"

    # Skip if already completed
    if results_path.exists():
        log.warning("Skipping %s (already completed)", output_dir)
        return json.loads(results_path.read_text())

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_config.setup(output_dir, resume=checkpoint.exists(output_dir), quiet=config.quiet)
    log.info("=" * 60)
    log.info("KD Experiment: %s %s on %s (seed=%d)", config.model, config.version.value, config.dataset, config.seed)
    log.info("Teacher: %s", teacher_path)
    log.info("KD params: temperature=%.1f, alpha=%.2f", temperature, alpha)
    log.info("Device: %s (GPUs: %d)", device.type, torch.cuda.device_count())
    log.info("=" * 60)

    # Data
    train_set = get_dataset(config.dataset, "train", root=config.data_dir, augment=config.augment)
    test_set = get_dataset(config.dataset, "test", root=config.data_dir)
    log.info("Dataset: %s (train=%d, test=%d)", config.dataset, len(train_set), len(test_set))  # type: ignore[arg-type]
    train_loader = DataLoader(
        train_set, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, config.batch_size * 2, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    # Models
    num_classes = DATASET_NUM_CLASSES.get(config.dataset, 10)
    teacher = load_teacher(config.model, num_classes, teacher_path, device)
    log.info("Teacher loaded from: %s", teacher_path)

    skip_layers = ABLATION_SKIP_LAYERS.get(config.ablation, set())
    student = get_model(config.model, num_classes, bit_version=True, pretrained=False, skip_layers=skip_layers)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)
    student = student.to(device)
    if config.ablation != AblationMode.NONE:
        log.info("Student: %s (bit, ablation=%s)", config.model, config.ablation.value)
    else:
        log.info("Student: %s (bit)", config.model)

    # Training setup
    criterion = KDLoss(temperature=temperature, alpha=alpha)
    ce_criterion = nn.CrossEntropyLoss()  # For evaluation
    optimizer: optim.AdamW | optim.SGD
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:  # Default to SGD
        optimizer = optim.SGD(
            student.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay, nesterov=True
        )
    scheduler = get_scheduler(optimizer, config.scheduler, config.epochs, config.warmup_epochs)

    # Save config
    config_dict = dataclasses.asdict(config)
    config_dict["version"] = config.version.value
    config_dict["ablation"] = config.ablation.value
    config_dict["kd"] = {"teacher_path": str(teacher_path), "temperature": temperature, "alpha": alpha}
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    start_epoch, best_acc, history = checkpoint.load(output_dir, student, optimizer, scheduler)
    history.setdefault("hard_loss", [])
    history.setdefault("soft_loss", [])

    writer = SummaryWriter(output_dir / "tensorboard") if config.tensorboard else None

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss, train_acc, kd_metrics = train_epoch_kd(
            student, teacher, train_loader, criterion, optimizer, device, quiet=config.quiet
        )
        test_loss, test_acc = evaluate(student, test_loader, ce_criterion, device, quiet=config.quiet)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["hard_loss"].append(kd_metrics["hard_loss"])
        history["soft_loss"].append(kd_metrics["soft_loss"])

        log.info(
            "Epoch %d/%d: train_acc=%.2f%%, test_acc=%.2f%%, hard=%.4f, soft=%.4f",
            epoch,
            config.epochs,
            train_acc,
            test_acc,
            kd_metrics["hard_loss"],
            kd_metrics["soft_loss"],
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("KD/hard_loss", kd_metrics["hard_loss"], epoch)
            writer.add_scalar("KD/soft_loss", kd_metrics["soft_loss"], epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            torch.save(state, output_dir / "best_model.pth")
            log.info("New best model saved (acc=%.2f%%)", best_acc)

        checkpoint.save(output_dir, epoch, student, optimizer, scheduler, best_acc, history)

    if writer:
        writer.close()

    results = {"best_acc": best_acc, "final_test_acc": test_acc, "history": history, "config": config_dict}
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    checkpoint.cleanup(output_dir)

    log.warning("Training complete! Best accuracy: %.2f%%", best_acc)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BitNet with Knowledge Distillation")
    parser.add_argument("--model", default=DEFAULTS.model)
    parser.add_argument("--dataset", default=DEFAULTS.dataset, choices=list(DATASET_NUM_CLASSES.keys()))
    parser.add_argument("--teacher-path", type=Path, required=True, help="Path to teacher best_model.pth")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature (default: 4.0)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Weight for soft loss (default: 0.9)")
    parser.add_argument("--ablation", default="none", choices=[m.value for m in AblationMode])
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.lr)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--scheduler", default=DEFAULTS.scheduler, choices=["cosine", "step", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--augment", default="basic", choices=AUGMENT_CHOICES)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--data-dir", default=DEFAULTS.data_dir)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    # Auto-generate output dir if not specified
    if not args.output_dir:
        experiment_dir = get_experiment_dir(
            ExperimentType.KD,
            args.dataset,
            args.model,
            args.seed,
            ablation=args.ablation,
            lr=args.lr,
            optimizer=args.optimizer,
            kd_temperature=args.temperature,
            kd_alpha=args.alpha,
        )
        args.output_dir = str(experiment_dir)

    # Skip if experiment already exists (unless --force)
    if should_skip_experiment(Path(args.output_dir), force=args.force):
        return

    config = TrainConfig(
        model=args.model,
        dataset=args.dataset,
        version=Version.BIT,  # Always BitNet for KD student
        ablation=AblationMode(args.ablation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        augment=args.augment,
        seed=args.seed,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tensorboard=args.tensorboard,
        quiet=args.quiet,
    )
    train_kd(config, args.teacher_path, args.temperature, args.alpha)


if __name__ == "__main__":
    main()
