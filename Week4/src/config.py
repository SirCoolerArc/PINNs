"""Domain bounds, hyperparameters, seeds, and output paths."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# Repository-relative locations for saved artefacts.
WEEK4_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = WEEK4_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.json"


@dataclass
class Config:
    """Hyperparameters and domain for the Taylor-Green forward PINN."""

    # --- Physics / domain ---
    nu: float = 0.01                      # true kinematic viscosity
    x_min: float = 0.0
    x_max: float = 2.0 * math.pi
    y_min: float = 0.0
    y_max: float = 2.0 * math.pi
    t_min: float = 0.0
    t_max: float = 2.0                     # spans a few decay e-foldings of exp(-2*nu*t)

    # --- Sampling counts ---
    n_collocation: int = 20000             # interior PDE-residual points
    n_initial: int = 4000                  # points on the t = 0 slice (IC)
    n_boundary: int = 2000                 # paired points per periodic-edge constraint

    # --- Model ---
    n_layers: int = 8                      # hidden layers
    n_units: int = 64                      # units per hidden layer
    n_inputs: int = 3                      # (x, y, t)
    n_outputs: int = 2                     # (psi, p) via the stream-function head

    # --- Optimization ---
    adam_iters: int = 20000
    adam_lr: float = 1.0e-3
    lr_decay_factor: float = 0.5           # plateau scheduler
    lr_decay_patience: int = 1000
    lbfgs_iters: int = 5000

    # --- Loss weights ---
    w_ic: float = 1.0
    w_bc: float = 1.0
    w_pde: float = 1.0

    # --- Misc ---
    seed: int = 1234
    prefer_cuda: bool = True
    log_every: int = 500

    # Filled in lazily so the tuple is not shared across instances.
    domain_lows: tuple = field(default=None, repr=False)
    domain_highs: tuple = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.domain_lows = (self.x_min, self.y_min, self.t_min)
        self.domain_highs = (self.x_max, self.y_max, self.t_max)
