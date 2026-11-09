"""
Unit tests for the evaluation pipeline, metrics calculation, and visualization generators.
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import (
    evaluate_test_set,
    plot_confusion_matrices,
    plot_error_analysis,
    plot_training_curves,
)
from src.model import MNISTCNN
from src.utils import set_seed


@pytest.fixture(autouse=True)
def init_seed():
    """Ensure deterministic random seed across tests."""
    set_seed(42)


def test_evaluate_metrics_on_synthetic_data():
    """Verify evaluation metric calculations and output structure on synthetic batch."""
    device = torch.device("cpu")
    model = MNISTCNN().to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # 10 samples (1 for each class 0-9)
    dummy_x = torch.randn(10, 1, 28, 28)
    dummy_y = torch.tensor(list(range(10)), dtype=torch.long)
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=5)

    results = evaluate_test_set(model, loader, criterion, device)

    assert "test_loss" in results
    assert "test_accuracy" in results
    assert "macro_f1" in results
    assert "weighted_f1" in results
    assert "per_class_metrics" in results
    assert "confusion_matrix" in results
    assert "confusion_pairs" in results
    assert "misclassified_records" in results

    assert results["total_samples"] == 10
    assert 0.0 <= results["test_accuracy"] <= 1.0
    assert len(results["confusion_matrix"]) == 10
    assert len(results["confusion_matrix"][0]) == 10


def test_confusion_matrix_plotting():
    """Verify confusion matrix plotting generates valid raw and normalized image files."""
    cm = np.zeros((10, 10), dtype=int)
    np.fill_diagonal(cm, 950)
    cm[4, 9] = 30
    cm[3, 5] = 20

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = Path(tmp_dir) / "cm_raw.png"
        norm_path = Path(tmp_dir) / "cm_norm.png"

        plot_confusion_matrices(cm, raw_path, norm_path)

        assert raw_path.is_file() and raw_path.stat().st_size > 0
        assert norm_path.is_file() and norm_path.stat().st_size > 0


def test_training_curves_plotting():
    """Verify learning curve plotter handles history dictionary and outputs image."""
    dummy_history = {
        "best_epoch": 2,
        "history": {
            "epochs": [1, 2, 3],
            "train_loss": [0.4, 0.2, 0.1],
            "val_loss": [0.2, 0.1, 0.15],
            "train_acc": [0.90, 0.95, 0.98],
            "val_acc": [0.94, 0.97, 0.96],
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        curve_path = Path(tmp_dir) / "curves.png"
        plot_training_curves(dummy_history, curve_path)

        assert curve_path.is_file()
        assert curve_path.stat().st_size > 0


def test_error_analysis_plotting():
    """Verify error analysis visualizer generates grid for misclassified samples."""
    dummy_records = [
        {
            "index": 42,
            "actual": 4,
            "predicted": 9,
            "confidence": 0.88,
            "tensor": torch.randn(1, 28, 28),
        },
        {
            "index": 105,
            "actual": 3,
            "predicted": 5,
            "confidence": 0.72,
            "tensor": torch.randn(1, 28, 28),
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        error_path = Path(tmp_dir) / "errors.png"
        plot_error_analysis(dummy_records, error_path, num_samples=16)

        assert error_path.is_file()
        assert error_path.stat().st_size > 0


def test_training_history_file_validity():
    """Verify that models/training_history.json generated during Phase 3 is well-structured."""
    history_file = Path("models/training_history.json")
    assert history_file.is_file(), "models/training_history.json not found"

    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "best_epoch" in data
    assert "best_val_loss" in data
    assert "best_val_acc" in data
    assert "history" in data
    assert "epochs" in data["history"]
    assert "val_loss" in data["history"]
    assert len(data["history"]["epochs"]) == data["epochs_completed"]

