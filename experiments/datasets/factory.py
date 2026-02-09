"""Dataset factory for loading datasets."""

import datasets as hf_datasets
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms

# Use ImageNet normalization for all datasets when fine-tuning pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation levels
AUGMENT_CHOICES = ["basic", "randaug", "cutout", "full"]


def get_cifar_train_transform(augment: str = "basic") -> transforms.Compose:
    """Get CIFAR training transform with specified augmentation level.

    Augmentation levels:
    - basic: RandomCrop + RandomHorizontalFlip (baseline)
    - randaug: basic + RandAugment
    - cutout: basic + RandomErasing (Cutout equivalent)
    - full: RandAugment + RandomErasing (SOTA)
    """
    base = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if augment == "randaug":
        base.insert(0, transforms.RandAugment(num_ops=2, magnitude=9))
    elif augment == "full":
        base.insert(0, transforms.RandAugment(num_ops=2, magnitude=9))

    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if augment in ("cutout", "full"):
        base.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)))

    return transforms.Compose(base)


def get_cifar_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_imagenet_train_transform(image_size: int = 224, augment: str = "basic") -> transforms.Compose:
    """Get ImageNet training transform with specified augmentation level."""
    base = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
    ]

    if augment in ("randaug", "full"):
        base.insert(0, transforms.RandAugment(num_ops=2, magnitude=9))

    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if augment in ("cutout", "full"):
        base.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)))

    return transforms.Compose(base)


def get_imagenet_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_tiny_imagenet_train_transform(augment: str = "basic") -> transforms.Compose:
    """Get Tiny ImageNet (64x64) training transform."""
    base = [
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
    ]

    if augment in ("randaug", "full"):
        base.insert(0, transforms.RandAugment(num_ops=2, magnitude=9))

    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if augment in ("cutout", "full"):
        base.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)))

    return transforms.Compose(base)


def get_tiny_imagenet_eval_transform() -> transforms.Compose:
    """Get Tiny ImageNet eval transform (no resize needed, already 64x64)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HFImageNetDataset(Dataset):
    """ImageNet dataset wrapper for HuggingFace datasets."""

    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def _reorganize_tiny_imagenet_val(val_dir: str) -> None:
    """Reorganize Tiny ImageNet val set into class folders (run once)."""
    import shutil
    from pathlib import Path

    val_path = Path(val_dir)
    annotations_file = val_path / "val_annotations.txt"

    # Check if already reorganized (class folders exist)
    if not annotations_file.exists():
        return

    # Read annotations and move images to class folders
    with open(annotations_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, class_id = parts[0], parts[1]
            class_dir = val_path / class_id
            class_dir.mkdir(exist_ok=True)

            src = val_path / "images" / img_name
            dst = class_dir / img_name
            if src.exists():
                shutil.move(str(src), str(dst))

    # Clean up
    images_dir = val_path / "images"
    if images_dir.exists() and not list(images_dir.iterdir()):
        images_dir.rmdir()
    annotations_file.unlink(missing_ok=True)


def get_dataset(name: str, split: str, root: str = "./data", augment: str = "basic") -> Dataset:
    """Load dataset with specified augmentation level.

    Args:
        name: Dataset name (cifar10, cifar100, imagenet, tiny_imagenet)
        split: train or test
        root: Data directory
        augment: Augmentation level (basic, randaug, cutout, full)
    """
    is_train = split == "train"

    if name in ("cifar10", "cifar100"):
        transform = get_cifar_train_transform(augment) if is_train else get_cifar_eval_transform()
        dataset_cls = tv_datasets.CIFAR10 if name == "cifar10" else tv_datasets.CIFAR100
        return dataset_cls(root, train=is_train, transform=transform, download=True)  # type: ignore[no-any-return]

    if name == "tiny_imagenet":
        from pathlib import Path

        tiny_root = Path(root) / "tiny-imagenet-200"
        if is_train:
            transform = get_tiny_imagenet_train_transform(augment)
            data_dir = tiny_root / "train"
        else:
            transform = get_tiny_imagenet_eval_transform()
            val_dir = tiny_root / "val"
            _reorganize_tiny_imagenet_val(str(val_dir))
            data_dir = val_dir
        return tv_datasets.ImageFolder(str(data_dir), transform=transform)  # type: ignore[no-any-return]

    if name == "imagenet":
        if is_train:
            transform = get_imagenet_train_transform(augment=augment)
        else:
            transform = get_imagenet_eval_transform()
        hf_split = "train" if is_train else "validation"
        hf_dataset = hf_datasets.load_dataset("ILSVRC/imagenet-1k", split=hf_split, cache_dir=root)
        result: Dataset = HFImageNetDataset(hf_dataset, transform)
        return result

    raise ValueError(f"Unknown dataset: {name}")
