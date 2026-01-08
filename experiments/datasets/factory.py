"""Dataset factory for loading datasets."""

from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms

# Use ImageNet normalization for all datasets when fine-tuning pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cifar_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_cifar_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


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


def get_dataset(name: str, split: str, root: str = "./data") -> Dataset:
    is_train = split == "train"

    if name in ("cifar10", "cifar100"):
        transform = get_cifar_train_transform() if is_train else get_cifar_eval_transform()
        dataset_cls = tv_datasets.CIFAR10 if name == "cifar10" else tv_datasets.CIFAR100
        return dataset_cls(root, train=is_train, transform=transform, download=True)

    if name == "imagenet":
        import datasets

        transform = get_imagenet_train_transform() if is_train else get_imagenet_eval_transform()
        hf_split = "train" if is_train else "validation"
        hf_dataset = datasets.load_dataset("ILSVRC/imagenet-1k", split=hf_split, cache_dir=root)
        return HFImageNetDataset(hf_dataset, transform)

    raise ValueError(f"Unknown dataset: {name}")
