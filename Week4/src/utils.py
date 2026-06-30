"""Seeding, device selection, and relative-L2 error metrics."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the CUDA device if available (and preferred), else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Relative L2 error ||pred - true||_2 / ||true||_2 as a Python float."""
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    num = torch.linalg.norm(pred - true)
    den = torch.linalg.norm(true)
    return (num / den).item()
