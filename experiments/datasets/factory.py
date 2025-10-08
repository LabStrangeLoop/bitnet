"""Dataset factory for loading and preprocessing datasets."""


import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset


def get_transforms(transform_type: str, dataset_name: str) -> transforms.Compose:
    """Get transforms for a dataset."""
    if transform_type == "standard":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    raise ValueError(f"Unknown transform type: {transform_type}")


def get_dataset(
    name: str, split: str, transform_type: str = "standard", root: str = "./data"
) -> VisionDataset:
    """Get dataset from torchvision or HuggingFace."""
    transform = get_transforms(transform_type, name)
    train = split == "train"

    torchvision_datasets: dict[str, type[VisionDataset]] = {
        "cifar10": tv_datasets.CIFAR10,
        "cifar100": tv_datasets.CIFAR100,
    }

    if name in torchvision_datasets:
        return torchvision_datasets[name](
            root=root, train=train, transform=transform, download=True # type: ignore
        )

    raise ValueError(f"Unknown dataset: {name}")
