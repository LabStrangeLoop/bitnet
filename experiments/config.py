"""Experiment configuration: models, datasets, and seeds."""

MODELS = ["resnet18", "resnet50", "vgg16", "mobilenetv2_100", "efficientnet_b0"]
DATASETS = ["cifar10", "cifar100", "imagenet"]
SEEDS = [42, 123, 456]

# Dataset-specific settings
DATASET_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}
DATASET_EPOCHS = {"cifar10": 200, "cifar100": 200, "imagenet": 90}
