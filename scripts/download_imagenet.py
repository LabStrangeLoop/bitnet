#!/usr/bin/env python3
"""Download ImageNet-1k dataset from HuggingFace.

Requires:
1. HuggingFace account: https://huggingface.co
2. Accept ImageNet terms: https://huggingface.co/datasets/imagenet-1k
3. Login: huggingface-cli login
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ImageNet-1k from HuggingFace")
    parser.add_argument("--data-dir", default="./data", help="Directory to save dataset")
    parser.add_argument("--split", default="all", choices=["train", "validation", "all"])
    args = parser.parse_args()

    from datasets import load_dataset

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"Downloading {split} split...")
        load_dataset("imagenet-1k", split=split, cache_dir=str(data_dir))
        print(f"Downloaded {split} split to {data_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
