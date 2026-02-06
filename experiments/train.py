"""Training script for standard and bit-quantized models."""

import argparse
import dataclasses
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from experiments.config import (
    ABLATION_SKIP_LAYERS,
    DATASET_NUM_CLASSES,
    DEFAULTS,
    AblationMode,
    EpochMetrics,
    TrainConfig,
    Version,
)
from experiments.datasets.factory import AUGMENT_CHOICES, get_dataset
from experiments.models.factory import get_model
from experiments.paths import should_skip_experiment
from experiments.training import checkpoint, logging_config
from experiments.training.loops import evaluate, get_scheduler, train_epoch

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(config: TrainConfig) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_config.setup(output_dir, resume=checkpoint.exists(output_dir), quiet=config.quiet)
    logging_config.log_experiment_header(config, device)

    # Data
    train_set = get_dataset(config.dataset, "train", root=config.data_dir, augment=config.augment)
    test_set = get_dataset(config.dataset, "test", root=config.data_dir)
    log.info("Dataset: %s (train=%d, test=%d)", config.dataset, len(train_set), len(test_set))  # type: ignore[arg-type]
    train_loader = DataLoader(
        train_set,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Model
    num_classes = DATASET_NUM_CLASSES.get(config.dataset, 10)
    is_bit = config.version == Version.BIT
    skip_layers = ABLATION_SKIP_LAYERS.get(config.ablation, set())
    model = get_model(config.model, num_classes, is_bit, config.pretrained, skip_layers)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    log.info("Model: %s (%s)", config.model, config.version.value)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer: optim.AdamW | optim.SGD
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:  # Default to SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    scheduler = get_scheduler(optimizer, config.scheduler, config.epochs, config.warmup_epochs)
    config_dict = dataclasses.asdict(config)
    config_dict["version"] = config.version.value  # Serialize enum
    config_dict["ablation"] = config.ablation.value  # Serialize enum
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    start_epoch, best_acc, history = checkpoint.load(output_dir, model, optimizer, scheduler)

    # TensorBoard
    tb_dir = output_dir / "tensorboard"
    if start_epoch == 0 and tb_dir.exists():
        shutil.rmtree(tb_dir)
    writer = SummaryWriter(tb_dir) if config.tensorboard else None

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, quiet=config.quiet)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, quiet=config.quiet)
        scheduler.step()

        metrics = EpochMetrics(train_loss, train_acc, test_loss, test_acc)
        for key, val in metrics.as_dict().items():
            history[key].append(val)
        logging_config.log_epoch(epoch, metrics, writer, total_epochs=config.epochs if config.quiet else 0)

        if metrics.test_acc > best_acc:
            best_acc = metrics.test_acc
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, output_dir / "best_model.pth")
            log.info("New best model saved (acc=%.2f%%)", best_acc)

        checkpoint.save(output_dir, epoch, model, optimizer, scheduler, best_acc, history)

    if writer:
        writer.close()

    results = {
        "best_acc": best_acc,
        "final_test_acc": test_acc,
        "history": history,
        "config": config_dict,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    checkpoint.cleanup(output_dir)

    # Use warning level so it shows even in quiet mode
    log.warning("Training complete! Best accuracy: %.2f%%", best_acc)
    log.info("Results saved to: %s", output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULTS.model)
    parser.add_argument("--dataset", default=DEFAULTS.dataset, choices=list(DATASET_NUM_CLASSES.keys()))
    parser.add_argument("--bit-version", action="store_true")
    parser.add_argument(
        "--ablation",
        default="none",
        choices=[m.value for m in AblationMode],
        help="Layer-wise ablation: keep specified layer in FP32",
    )
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.lr)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", default=DEFAULTS.scheduler, choices=["cosine", "step", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--augment", default="basic", choices=AUGMENT_CHOICES)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--data-dir", default=DEFAULTS.data_dir)
    parser.add_argument("--output-dir", default="")  # Empty = auto-generate
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    config = TrainConfig(
        model=args.model,
        dataset=args.dataset,
        version=Version.from_bool(args.bit_version),
        ablation=AblationMode(args.ablation),
        pretrained=args.pretrained,
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

    # Skip if experiment already exists (unless --force)
    if should_skip_experiment(Path(config.output_dir), force=args.force):
        return

    train(config)


if __name__ == "__main__":
    main()
