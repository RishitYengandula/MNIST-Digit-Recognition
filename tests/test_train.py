"""
Unit tests for the training pipeline components, checkpointing, and single-batch execution.
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MNISTCNN
from src.train import load_checkpoint, train_one_epoch, validate
from src.utils import get_device, load_config, set_seed


@pytest.fixture(autouse=True)
def init_seed():
    """Ensure deterministic random seed across tests."""
    set_seed(42)


def test_training_config_completeness():
    """Verify config.yaml contains all hyperparameters required for training."""
    config = load_config("config/config.yaml")

    assert "training" in config
    train_cfg = config["training"]
    assert "batch_size" in train_cfg and train_cfg["batch_size"] > 0
    assert "learning_rate" in train_cfg and train_cfg["learning_rate"] > 0
    assert "weight_decay" in train_cfg and train_cfg["weight_decay"] >= 0
    assert "max_epochs" in train_cfg and train_cfg["max_epochs"] > 0
    assert "early_stopping_patience" in train_cfg and train_cfg["early_stopping_patience"] > 0
    assert "lr_scheduler" in train_cfg
    assert "factor" in train_cfg["lr_scheduler"]
    assert "patience" in train_cfg["lr_scheduler"]

    assert "paths" in config
    paths_cfg = config["paths"]
    assert "best_model_path" in paths_cfg
    assert "training_history_path" in paths_cfg


def test_single_batch_training_step():
    """Verify train_one_epoch executes forward, backward, and updates parameters without errors."""
    device = torch.device("cpu")
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic batch of 8 samples
    dummy_x = torch.randn(8, 1, 28, 28)
    dummy_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    synthetic_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)

    # Capture initial weights of first conv layer
    initial_weight = model.conv_block1[0].weight.clone()

    loss, acc = train_one_epoch(
        model=model,
        dataloader=synthetic_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # Parameter must have been updated by backprop
    updated_weight = model.conv_block1[0].weight
    assert not torch.equal(initial_weight, updated_weight), "Weights were not updated after training step"
    assert loss > 0.0, f"Expected positive loss, got {loss}"
    assert 0.0 <= acc <= 1.0, f"Accuracy must be in [0, 1], got {acc}"


def test_validation_step_does_not_modify_weights():
    """Verify validation executes without mutating model weights."""
    device = torch.device("cpu")
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    dummy_x = torch.randn(8, 1, 28, 28)
    dummy_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    synthetic_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)

    initial_weight = model.conv_block1[0].weight.clone()

    loss, acc = validate(
        model=model,
        dataloader=synthetic_loader,
        criterion=criterion,
        device=device,
    )

    assert torch.equal(initial_weight, model.conv_block1[0].weight), "Validation modified model weights"
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0


def test_checkpoint_save_and_reload():
    """Verify checkpoint structure can be saved to disk and reloaded faithfully."""
    device = torch.device("cpu")
    original_model = MNISTCNN().to(device)
    original_model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_checkpoint_path = Path(tmp_dir) / "test_checkpoint.pth"

        checkpoint_data = {
            "epoch": 5,
            "model_state_dict": original_model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.045,
            "val_acc": 0.987,
            "train_loss": 0.062,
            "train_acc": 0.981,
            "config": {"test": True},
        }
        torch.save(checkpoint_data, tmp_checkpoint_path)

        # Reload
        loaded_model, loaded_meta = load_checkpoint(
            checkpoint_path=str(tmp_checkpoint_path),
            device=device,
        )

        assert loaded_meta["epoch"] == 5
        assert loaded_meta["val_loss"] == 0.045
        assert loaded_meta["val_acc"] == 0.987

        # Compare outputs on same input
        dummy_input = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            out_orig = original_model(dummy_input)
            out_loaded = loaded_model(dummy_input)

        assert torch.allclose(out_orig, out_loaded), "Reloaded model produced different predictions"

