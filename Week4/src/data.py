"""Taylor-Green analytical fields; sparse + noisy sampling; collocation points."""

from __future__ import annotations

import math

import torch

from .config import Config


# ---------------------------------------------------------------------------
# Taylor-Green vortex: exact solution of 2D incompressible Navier-Stokes on
# [0, 2*pi]^2 with periodic boundaries.
#   u(x,y,t) = -cos(x) sin(y) exp(-2 nu t)
#   v(x,y,t) =  sin(x) cos(y) exp(-2 nu t)
#   p(x,y,t) = -1/4 (cos(2x) + cos(2y)) exp(-4 nu t)
# ---------------------------------------------------------------------------


def tgv_velocity(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: float):
    """Analytical TGV velocity (u, v) at the given points."""
    decay = torch.exp(-2.0 * nu * t)
    u = -torch.cos(x) * torch.sin(y) * decay
    v = torch.sin(x) * torch.cos(y) * decay
    return u, v


def tgv_pressure(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: float):
    """Analytical TGV pressure p at the given points."""
    return -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * nu * t)


def tgv_fields(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: float):
    """Analytical TGV (u, v, p) at the given points."""
    u, v = tgv_velocity(x, y, t, nu)
    p = tgv_pressure(x, y, t, nu)
    return u, v, p


def _uniform(n: int, low: float, high: float, device, generator):
    return low + (high - low) * torch.rand(n, 1, device=device, generator=generator)


def sample_collocation(cfg: Config, device, generator):
    """Interior (x, y, t) points where the PDE residual is enforced."""
    x = _uniform(cfg.n_collocation, cfg.x_min, cfg.x_max, device, generator)
    y = _uniform(cfg.n_collocation, cfg.y_min, cfg.y_max, device, generator)
    t = _uniform(cfg.n_collocation, cfg.t_min, cfg.t_max, device, generator)
    return x, y, t


def sample_initial(cfg: Config, device, generator):
    """Points on the t = 0 slice with their analytical (u, v) targets (the IC)."""
    x = _uniform(cfg.n_initial, cfg.x_min, cfg.x_max, device, generator)
    y = _uniform(cfg.n_initial, cfg.y_min, cfg.y_max, device, generator)
    t = torch.full_like(x, cfg.t_min)
    u, v = tgv_velocity(x, y, t, cfg.nu)
    return x, y, t, u, v


def sample_boundary(cfg: Config, device, generator):
    """Matched point pairs on opposite edges for the periodic BC.

    Returns two (x, y, t) triplets; the loss enforces equal network output on
    each pair. With period 2*pi the left/right and bottom/top edges coincide.
    """
    period_x = cfg.x_max - cfg.x_min
    period_y = cfg.y_max - cfg.y_min

    # Left (x = x_min) paired with right (x = x_max), random y, t.
    y_lr = _uniform(cfg.n_boundary, cfg.y_min, cfg.y_max, device, generator)
    t_lr = _uniform(cfg.n_boundary, cfg.t_min, cfg.t_max, device, generator)
    left = (torch.full_like(y_lr, cfg.x_min), y_lr, t_lr)
    right = (torch.full_like(y_lr, cfg.x_min + period_x), y_lr, t_lr)

    # Bottom (y = y_min) paired with top (y = y_max), random x, t.
    x_bt = _uniform(cfg.n_boundary, cfg.x_min, cfg.x_max, device, generator)
    t_bt = _uniform(cfg.n_boundary, cfg.t_min, cfg.t_max, device, generator)
    bottom = (x_bt, torch.full_like(x_bt, cfg.y_min), t_bt)
    top = (x_bt, torch.full_like(x_bt, cfg.y_min + period_y), t_bt)

    return left, right, bottom, top


def grid_eval_points(cfg: Config, t: float, n: int, device):
    """A dense regular (x, y) grid at a fixed time, for evaluation / plotting.

    Returns flattened column tensors (x, y, t) plus the grid shape (n, n).
    """
    xs = torch.linspace(cfg.x_min, cfg.x_max, n, device=device)
    ys = torch.linspace(cfg.y_min, cfg.y_max, n, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    x = gx.reshape(-1, 1)
    y = gy.reshape(-1, 1)
    tt = torch.full_like(x, t)
    return x, y, tt, (n, n)
