"""
Unit tests for the MNIST dataset loading, deterministic splitting, and preprocessing pipeline.
"""

import pytest
import torch
from torchvision import transforms

from src.dataset import (
    TransformedSubset,
    get_class_distribution,
    get_data_loaders,
    get_dataset_sizes,
    load_mnist_datasets,
)
from src.utils import load_config, set_seed


@pytest.fixture(scope="module")
def dataset_bundle():
    """Load datasets once for all tests in this module."""
    set_seed(42)
    config = load_config("config/config.yaml")
    train_ds, val_ds, test_ds = load_mnist_datasets(config)
    return {
        "config": config,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }


def test_dataset_sizes(dataset_bundle):
    """Confirm exact canonical sizes: 50,000 train, 10,000 validation, 10,000 test."""
    train_ds = dataset_bundle["train_ds"]
    val_ds = dataset_bundle["val_ds"]
    test_ds = dataset_bundle["test_ds"]

    sizes = get_dataset_sizes(train_ds, val_ds, test_ds)
    assert sizes["train"] == 50000, f"Expected 50000 train samples, got {sizes['train']}"
    assert sizes["validation"] == 10000, f"Expected 10000 val samples, got {sizes['validation']}"
    assert sizes["test"] == 10000, f"Expected 10000 test samples, got {sizes['test']}"
    assert sizes["total"] == 70000, f"Expected 70000 total samples, got {sizes['total']}"


def test_tensor_shapes_and_label_bounds(dataset_bundle):
    """Verify single-sample tensor shape [1, 28, 28] and label range [0, 9]."""
    for split_name, ds in [
        ("train", dataset_bundle["train_ds"]),
        ("val", dataset_bundle["val_ds"]),
        ("test", dataset_bundle["test_ds"]),
    ]:
        image, label = ds[0]
        assert isinstance(image, torch.Tensor), f"{split_name} sample image is not a torch.Tensor"
        assert image.shape == (1, 28, 28), f"{split_name} sample shape mismatch: {image.shape}"
        assert image.dtype == torch.float32, f"{split_name} sample dtype mismatch: {image.dtype}"
        assert isinstance(label, int), f"{split_name} label must be int, got {type(label)}"
        assert 0 <= label <= 9, f"{split_name} label {label} outside expected range [0, 9]"


def test_normalized_tensor_values(dataset_bundle):
    """Confirm normalized tensors contain sensible numerical ranges without NaNs or Infs."""
    for split_name, ds in [
        ("train", dataset_bundle["train_ds"]),
        ("val", dataset_bundle["val_ds"]),
        ("test", dataset_bundle["test_ds"]),
    ]:
        image, _ = ds[0]
        assert not torch.isnan(image).any(), f"NaN detected in {split_name} tensor"
        assert not torch.isinf(image).any(), f"Inf detected in {split_name} tensor"
        # Background pixels (0.0 in [0, 1]) normalize to (0 - 0.1307)/0.3081 ~= -0.4242
        # Max white pixels (1.0 in [0, 1]) normalize to (1 - 0.1307)/0.3081 ~= 2.8215
        assert image.min() >= -1.0, f"{split_name} min pixel value {image.min()} unusually low"
        assert image.max() <= 4.0, f"{split_name} max pixel value {image.max()} unusually high"


def test_train_val_split_disjoint(dataset_bundle):
    """Verify train and validation subsets have zero index overlap and cover the 60,000 raw training pool."""
    train_ds = dataset_bundle["train_ds"]
    val_ds = dataset_bundle["val_ds"]

    assert isinstance(train_ds, TransformedSubset)
    assert isinstance(val_ds, TransformedSubset)

    train_indices = set(train_ds.subset.indices)
    val_indices = set(val_ds.subset.indices)

    intersection = train_indices.intersection(val_indices)
    assert len(intersection) == 0, f"Found {len(intersection)} overlapping indices between train and val"
    assert len(train_indices) + len(val_indices) == 60000, "Indices do not sum to canonical 60,000 set"


def test_augmentation_attached_only_to_train(dataset_bundle):
    """Confirm data augmentation is attached ONLY to training data, not validation or test."""
    train_transform = dataset_bundle["train_ds"].transform
    val_transform = dataset_bundle["val_ds"].transform
    test_transform = dataset_bundle["test_ds"].transform

    def has_random_affine(t_compose):
        if not isinstance(t_compose, transforms.Compose):
            return False
        return any(isinstance(step, transforms.RandomAffine) for step in t_compose.transforms)

    assert has_random_affine(train_transform), "Training pipeline must include RandomAffine augmentation"
    assert not has_random_affine(val_transform), "Validation pipeline must NOT include augmentation"
    assert not has_random_affine(test_transform), "Test pipeline must NOT include augmentation"


def test_class_distributions_balanced(dataset_bundle):
    """Ensure every digit (0-9) is well-represented in each partition."""
    train_dist = get_class_distribution(dataset_bundle["train_ds"])
    val_dist = get_class_distribution(dataset_bundle["val_ds"])
    test_dist = get_class_distribution(dataset_bundle["test_ds"])

    assert len(train_dist) == 10
    assert len(val_dist) == 10
    assert len(test_dist) == 10

    # In MNIST each class has approximately 10% representation (~4500-5500 for train, ~900-1100 for val/test)
    for digit in range(10):
        assert train_dist[digit] >= 4000, f"Class {digit} underrepresented in train: {train_dist[digit]}"
        assert val_dist[digit] >= 800, f"Class {digit} underrepresented in val: {val_dist[digit]}"
        assert test_dist[digit] >= 800, f"Class {digit} underrepresented in test: {test_dist[digit]}"


def test_dataloaders_batch_generation(dataset_bundle):
    """Verify DataLoaders generate batches of correct dimensions and types."""
    config = dataset_bundle["config"]
    train_loader, val_loader, test_loader = get_data_loaders(config, num_workers=0)

    batch_x, batch_y = next(iter(train_loader))
    batch_size = config["training"]["batch_size"]

    assert batch_x.shape == (batch_size, 1, 28, 28), f"Batch X shape mismatch: {batch_x.shape}"
    assert batch_y.shape == (batch_size,), f"Batch Y shape mismatch: {batch_y.shape}"
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64

