"""
Training pipeline for MNIST digit recognition using PyTorch, AdamW,
learning rate scheduling, and early stopping.
"""

from pathlib import Path
import json
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_data_loaders
from src.model import MNISTCNN
from src.utils import ensure_dir, get_device, load_config, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute one training epoch over the dataset.

    Returns:
        Tuple[float, float]: (average_loss, accuracy_fraction)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute validation without gradient calculation.

    Returns:
        Tuple[float, float]: (average_loss, accuracy_fraction)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_pipeline(
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate full training cycle: data loading, optimization, early stopping,
    checkpointing the best validation loss model, and recording metrics history.

    Args:
        config: Configuration dictionary (loads config/config.yaml if None).
        verbose: Whether to print progress tables to stdout.

    Returns:
        Dict[str, Any]: Comprehensive training summary and history.
    """
    if config is None:
        config = load_config()

    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    device = get_device()

    # Paths
    paths_cfg = config.get("paths", {})
    models_dir = ensure_dir(paths_cfg.get("models_dir", "models"))
    best_model_path = Path(paths_cfg.get("best_model_path", "models/best_model.pth"))
    history_path = Path(paths_cfg.get("training_history_path", "models/training_history.json"))
    metadata_path = Path(paths_cfg.get("model_metadata_path", "models/model_metadata.json"))

    # Training parameters
    train_cfg = config.get("training", {})
    max_epochs = train_cfg.get("max_epochs", 15)
    learning_rate = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 3)

    sched_cfg = train_cfg.get("lr_scheduler", {})
    sched_factor = sched_cfg.get("factor", 0.5)
    sched_patience = sched_cfg.get("patience", 2)
    sched_min_lr = float(sched_cfg.get("min_lr", 1e-6))

    if verbose:
        print("=" * 78)
        print("Starting MNIST CNN Training Pipeline")
        print("=" * 78)
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        print(f"Device:                 {device.type.upper()} ({device_name})")
        print(f"Random Seed:            {seed}")
        print(f"Max Epochs:             {max_epochs}")
        print(f"Batch Size:             {train_cfg.get('batch_size', 64)}")
        print(f"Initial Learning Rate:  {learning_rate}")
        print(f"Weight Decay:           {weight_decay}")
        print(f"Early Stopping Patience:{early_stopping_patience} epochs")
        print(f"Checkpoint Target:      {best_model_path}")
        print("=" * 78)

    # Data loaders
    train_loader, val_loader, _ = get_data_loaders(config)

    # Model, Loss, Optimizer, Scheduler
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_factor,
        patience=sched_patience,
        min_lr=sched_min_lr,
    )

    history: Dict[str, list] = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
        "epoch_duration_seconds": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0
    total_training_start = time.time()

    if verbose:
        header = f"{'Epoch':<7}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<12}{'LR':<11}{'Time':<8}{'Status'}"
        print(header)
        print("-" * 78)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        duration = time.time() - t0

        # Step learning rate scheduler based on validation loss
        scheduler.step(val_loss)

        # Record history
        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["learning_rate"].append(round(current_lr, 8))
        history["epoch_duration_seconds"].append(round(duration, 2))

        status = ""
        # Checkpoint if lowest validation loss achieved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "config": config,
            }
            torch.save(checkpoint, best_model_path)
            status = "* Best Checkpoint Saved"
        else:
            epochs_no_improve += 1
            status = f"No improve ({epochs_no_improve}/{early_stopping_patience})"

        if verbose:
            row = (
                f"{epoch:<7}"
                f"{train_loss:<12.4f}"
                f"{train_acc * 100:<11.2f}%"
                f"{val_loss:<12.4f}"
                f"{val_acc * 100:<11.2f}%"
                f"{current_lr:<11.6f}"
                f"{duration:<7.1f}s"
                f"{status}"
            )
            print(row)

        if epochs_no_improve >= early_stopping_patience:
            if verbose:
                print("-" * 78)
                print(f"Early stopping triggered at epoch {epoch} (no validation loss improvement for {early_stopping_patience} consecutive epochs).")
            break

    total_training_duration = time.time() - total_training_start

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc": round(best_val_acc, 6),
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "epochs_completed": len(history["epochs"]),
        "total_training_time_seconds": round(total_training_duration, 2),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "checkpoint_path": str(best_model_path),
        "history": history,
    }

    # Save training history JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save metadata JSON companion
    metadata = {
        "project": config.get("project", {}).get("name", "mnist-digit-recognition"),
        "model_architecture": "MNISTCNN",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc": round(best_val_acc, 6),
        "epochs_trained": len(history["epochs"]),
        "training_duration_seconds": round(total_training_duration, 2),
        "normalization": config.get("dataset", {}).get("normalization", {}),
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print("=" * 78)
        print(f"Training Complete in {total_training_duration:.2f}s")
        print(f"  - Best Epoch:             {best_epoch}")
        print(f"  - Best Validation Loss:   {best_val_loss:.4f}")
        print(f"  - Best Validation Acc:    {best_val_acc * 100:.2f}%")
        print(f"  - Checkpoint Saved To:    {best_model_path}")
        print(f"  - History Saved To:       {history_path}")
        print("=" * 78)

    return summary


def load_checkpoint(
    checkpoint_path: str = "models/best_model.pth",
    device: Optional[torch.device] = None,
) -> Tuple[MNISTCNN, Dict[str, Any]]:
    """
    Safely load a trained model checkpoint and instantiate the model in eval mode.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Target torch device (defaults to get_device()).

    Returns:
        Tuple[MNISTCNN, Dict[str, Any]]: (instantiated model in eval mode, checkpoint dict)
    """
    if device is None:
        device = get_device()

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")

    checkpoint = torch.load(path, map_location=device)
    model = MNISTCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


if __name__ == "__main__":
    train_pipeline()
