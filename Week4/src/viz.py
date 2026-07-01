"""Field / error / vorticity / pressure plots and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _reshape(arr, shape):
    return np.asarray(arr).reshape(shape)


def plot_field_comparison(x, y, pred, true, shape, title, path: Path, cmap="RdBu_r"):
    """Predicted vs true vs absolute-error maps for one scalar field."""
    X = _reshape(x, shape)
    Y = _reshape(y, shape)
    P = _reshape(pred, shape)
    T = _reshape(true, shape)
    err = np.abs(P - T)

    vmin = min(P.min(), T.min())
    vmax = max(P.max(), T.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, field, name, kw in (
        (axes[0], T, "Exact", dict(cmap=cmap, vmin=vmin, vmax=vmax)),
        (axes[1], P, "PINN", dict(cmap=cmap, vmin=vmin, vmax=vmax)),
        (axes[2], err, "Absolute error", dict(cmap="viridis")),
    ):
        pc = ax.pcolormesh(X, Y, field, shading="auto", **kw)
        ax.set_title(f"{title} - {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history, path: Path):
    """Total / PDE / IC / BC loss versus optimizer iteration (log scale)."""
    its = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for key in ("total", "pde", "ic", "bc"):
        ax.semilogy(its, [h[key] for h in history], label=key)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_param_curve(values, true_value, path: Path, name="nu"):
    """Convergence of a learnable parameter (e.g. viscosity) toward its truth."""
    its = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(its, values, label=f"estimated {name}")
    ax.axhline(true_value, color="k", ls="--", label=f"true {name} = {true_value:g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(name)
    ax.set_title(f"{name} convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)
