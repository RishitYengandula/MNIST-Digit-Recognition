"""
Utility functions for reproducibility, device selection, and configuration loading.
"""

from pathlib import Path
import os
import random
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: The integer seed for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Detect and return the appropriate PyTorch computing device.

    Args:
        prefer_cuda: Whether to use CUDA if available.

    Returns:
        torch.device: 'cuda' if available and requested, otherwise 'cpu'.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Filepath to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the target directory.

    Returns:
        Path: Resolved Path object.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target

