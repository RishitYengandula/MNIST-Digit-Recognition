"""
Dataset pipeline for MNIST: loading, splitting, conservative augmentation,
normalization, and PyTorch DataLoader generation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.utils import ensure_dir, load_config


class TransformedSubset(Dataset):
    """
    Wraps a torch.utils.data.Subset to apply transforms on-the-fly.
    
    This ensures that raw images can be split before applying distinct
    transformations (e.g., augmentation for training vs. clean scaling for validation).
    """

    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, target = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)

    def __len__(self) -> int:
        return len(self.subset)

    @property
    def targets(self) -> torch.Tensor:
        """Extract targets efficiently without executing image transforms."""
        base_dataset = self.subset.dataset
        indices = self.subset.indices
        if hasattr(base_dataset, "targets"):
            return base_dataset.targets[indices]
        return torch.tensor([self.subset[i][1] for i in range(len(self.subset))])


def get_train_transforms(config: Optional[Dict[str, Any]] = None) -> transforms.Compose:
    """
    Construct training transformations: conservative augmentation + MNIST normalization.

    Augmentations applied:
      - Rotation: within +/- 10 degrees
      - Translation: within +/- 7% (~2 pixels on 28x28)
      - Scale: 95% to 105%
      - NO horizontal or vertical flipping (preserves digit semantics)
    """
    deg = 10
    trans = (0.07, 0.07)
    scale = (0.95, 1.05)
    mean = [0.1307]
    std = [0.3081]

    if config and "dataset" in config:
        aug_cfg = config["dataset"].get("augmentation", {})
        deg = aug_cfg.get("rotation_degrees", deg)
        trans = tuple(aug_cfg.get("translate", trans))
        scale = tuple(aug_cfg.get("scale", scale))
        norm_cfg = config["dataset"].get("normalization", {})
        mean = norm_cfg.get("mean", mean)
        std = norm_cfg.get("std", std)

    return transforms.Compose([
        transforms.RandomAffine(
            degrees=(-deg, deg),
            translate=trans,
            scale=scale,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_test_transforms(config: Optional[Dict[str, Any]] = None) -> transforms.Compose:
    """
    Construct validation/test transformations: strictly scaling and normalization (no augmentation).
    """
    mean = [0.1307]
    std = [0.3081]

    if config and "dataset" in config:
        norm_cfg = config["dataset"].get("normalization", {})
        mean = norm_cfg.get("mean", mean)
        std = norm_cfg.get("std", std)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_mnist_datasets(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Download/load official MNIST and produce deterministic (train, validation, test) datasets.

    - Canonical 60,000 training set is partitioned into 50,000 train and 10,000 validation.
    - Canonical 10,000 test set is loaded independently.
    - Augmentation is strictly bound to the 50,000 training set.
    """
    if config is None:
        config = load_config()

    data_dir = config.get("paths", {}).get("data_dir", "data")
    seed = config.get("reproducibility", {}).get("seed", 42)
    train_size = config.get("dataset", {}).get("train_samples", 50000)
    val_size = config.get("dataset", {}).get("val_samples", 10000)

    ensure_dir(data_dir)

    # Load canonical 60,000 training partition without transforms initially
    raw_train_val = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )

    if len(raw_train_val) != train_size + val_size:
        raise ValueError(
            f"Expected {train_size + val_size} total samples in raw training set, found {len(raw_train_val)}"
        )

    # Deterministic split into 50,000 train and 10,000 validation
    split_generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        raw_train_val,
        [train_size, val_size],
        generator=split_generator,
    )

    # Attach transforms: augmentation to train ONLY, clean transforms to val
    train_transforms = get_train_transforms(config)
    val_test_transforms = get_val_test_transforms(config)

    train_dataset = TransformedSubset(train_subset, transform=train_transforms)
    val_dataset = TransformedSubset(val_subset, transform=val_test_transforms)

    # Canonical 10,000 test set
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=val_test_transforms,
    )

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(
    config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for train, validation, and test datasets.
    """
    if config is None:
        config = load_config()

    batch_size = config.get("training", {}).get("batch_size", 64)
    train_ds, val_ds, test_ds = load_mnist_datasets(config)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_dataset_sizes(
    train_ds: Dataset, val_ds: Dataset, test_ds: Dataset
) -> Dict[str, int]:
    """Return dictionary of dataset sizes."""
    return {
        "train": len(train_ds),
        "validation": len(val_ds),
        "test": len(test_ds),
        "total": len(train_ds) + len(val_ds) + len(test_ds),
    }


def get_class_distribution(dataset: Union[Dataset, Subset, TransformedSubset]) -> Dict[int, int]:
    """
    Calculate the count of samples per class (0-9) in the given dataset or subset.
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]

    counts = {digit: 0 for digit in range(10)}
    for label in targets:
        counts[int(label)] += 1
    return counts


if __name__ == "__main__":
    import sys
    from src.utils import get_device, set_seed

    print("=" * 60)
    print("MNIST Dataset Verification Script")
    print("=" * 60)

    cfg = load_config()
    set_seed(cfg.get("reproducibility", {}).get("seed", 42))

    device = get_device()
    print(f"Python Version:     {sys.version.split()[0]}")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"Torchvision Version:{transforms.__file__ and 'available'}")
    print(f"NumPy Version:      {np.__version__}")
    print(f"Device Detected:    {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("\nLoading datasets...")
    train_data, val_data, test_data = load_mnist_datasets(cfg)
    sizes = get_dataset_sizes(train_data, val_data, test_data)

    print("\nDataset Sizes:")
    print(f"  - Training samples:   {sizes['train']:,}")
    print(f"  - Validation samples: {sizes['validation']:,}")
    print(f"  - Test samples:       {sizes['test']:,}")
    print(f"  - Total samples:      {sizes['total']:,}")

    img, lbl = train_data[0]
    print(f"\nSample Inspection (Train sample #0):")
    print(f"  - Tensor Shape:       {list(img.shape)}")
    print(f"  - Tensor Data Type:   {img.dtype}")
    print(f"  - Value Range (min/max): [{img.min():.4f}, {img.max():.4f}]")
    print(f"  - Label:              {lbl}")

    print("\nClass Distribution (Digit counts per split):")
    train_dist = get_class_distribution(train_data)
    val_dist = get_class_distribution(val_data)
    test_dist = get_class_distribution(test_data)
    print(f"{'Digit':<6}{'Train':<12}{'Validation':<12}{'Test':<12}")
    print("-" * 42)
    for d in range(10):
        print(f"{d:<6}{train_dist[d]:<12}{val_dist[d]:<12}{test_dist[d]:<12}")

    print("\nBatch Verification:")
    train_ldr, _, _ = get_data_loaders(cfg)
    batch_imgs, batch_lbls = next(iter(train_ldr))
    print(f"  - Batch Images Shape: {list(batch_imgs.shape)}")
    print(f"  - Batch Labels Shape: {list(batch_lbls.shape)}")
    print("=" * 60)
    print("Verification complete. Dataset is ready for Phase 2.")

