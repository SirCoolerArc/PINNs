"""Two-stage Adam -> L-BFGS training loop, loss assembly, and logging."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .config import Config
from . import data, inverse, pde


def _requires_grad(*tensors):
    return [t.requires_grad_(True) for t in tensors]


def forward_loss(model, batch, cfg: Config):
    """Physics-driven forward loss: IC + periodic BC + PDE residual.

    `batch` is a dict of pre-sampled tensors produced by `make_forward_batch`.
    No interior velocity labels are used -- the interior field is determined by
    the physics alone.
    """
    # --- PDE residual on interior collocation points ---
    xc, yc, tc = batch["coll"]
    f_u, f_v = pde.navier_stokes_residual(model, xc, yc, tc, cfg.nu)
    loss_pde = (f_u.pow(2).mean() + f_v.pow(2).mean())

    # --- Initial condition: match (u, v) at t = 0 ---
    xi, yi, ti, ui, vi = batch["ic"]
    u_i, v_i, _ = pde.predict_uvp(model, xi, yi, ti)
    loss_ic = F.mse_loss(u_i, ui) + F.mse_loss(v_i, vi)

    # --- Periodic BC: equal velocity on matched opposite-edge points ---
    left, right, bottom, top = batch["bc"]
    ul, vl, _ = pde.predict_uvp(model, *left)
    ur, vr, _ = pde.predict_uvp(model, *right)
    ub, vb, _ = pde.predict_uvp(model, *bottom)
    ut, vt, _ = pde.predict_uvp(model, *top)
    loss_bc = (
        F.mse_loss(ul, ur) + F.mse_loss(vl, vr)
        + F.mse_loss(ub, ut) + F.mse_loss(vb, vt)
    )

    total = cfg.w_pde * loss_pde + cfg.w_ic * loss_ic + cfg.w_bc * loss_bc
    parts = {
        "total": total.item(),
        "pde": loss_pde.item(),
        "ic": loss_ic.item(),
        "bc": loss_bc.item(),
    }
    return total, parts


def make_forward_batch(cfg: Config, device, generator):
    """Sample the fixed point sets used throughout forward training."""
    xc, yc, tc = data.sample_collocation(cfg, device, generator)
    xc, yc, tc = _requires_grad(xc, yc, tc)

    xi, yi, ti, ui, vi = data.sample_initial(cfg, device, generator)
    xi, yi, ti = _requires_grad(xi, yi, ti)

    left, right, bottom, top = data.sample_boundary(cfg, device, generator)
    bc = tuple(
        tuple(_requires_grad(*edge)) for edge in (left, right, bottom, top)
    )

    return {"coll": (xc, yc, tc), "ic": (xi, yi, ti, ui, vi), "bc": bc}


def train(model, cfg: Config, device, generator, log_fn=print):
    """Run Adam then L-BFGS on the forward loss. Returns the loss history."""
    batch = make_forward_batch(cfg, device, generator)
    history = []

    # --- Stage 1: Adam ---
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.lr_decay_factor, patience=cfg.lr_decay_patience
    )
    t0 = time.time()
    for it in range(1, cfg.adam_iters + 1):
        optimizer.zero_grad()
        loss, parts = forward_loss(model, batch, cfg)
        loss.backward()
        optimizer.step()
        scheduler.step(parts["total"])
        history.append(parts)
        if it % cfg.log_every == 0 or it == 1:
            lr = optimizer.param_groups[0]["lr"]
            log_fn(
                f"[adam {it:>6}/{cfg.adam_iters}] "
                f"total={parts['total']:.3e} pde={parts['pde']:.3e} "
                f"ic={parts['ic']:.3e} bc={parts['bc']:.3e} lr={lr:.1e}"
            )
    log_fn(f"Adam stage done in {time.time() - t0:.1f}s")

    # --- Stage 2: L-BFGS full-batch fine-tune ---
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=cfg.lbfgs_iters,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )
    state = {"it": 0}

    def closure():
        optimizer.zero_grad()
        loss, parts = forward_loss(model, batch, cfg)
        loss.backward()
        state["it"] += 1
        history.append(parts)
        if state["it"] % cfg.log_every == 0 or state["it"] == 1:
            log_fn(
                f"[lbfgs {state['it']:>6}] total={parts['total']:.3e} "
                f"pde={parts['pde']:.3e} ic={parts['ic']:.3e} bc={parts['bc']:.3e}"
            )
        return loss

    t0 = time.time()
    optimizer.step(closure)
    log_fn(f"L-BFGS stage done in {time.time() - t0:.1f}s ({state['it']} evals)")

    return history


def make_inverse_data(cfg: Config, n_meas, noise, device, generator):
    """Sample the sparse noisy measurements and the collocation points."""
    xm, ym, tm, um, vm = data.sample_measurements(n_meas, noise, cfg, device, generator)
    xm, ym, tm = _requires_grad(xm, ym, tm)
    meas = (xm, ym, tm, um, vm)

    xc, yc, tc = data.sample_collocation(cfg, device, generator)
    coll = tuple(_requires_grad(xc, yc, tc))
    return meas, coll


def train_inverse(model, nu_module, cfg: Config, meas, coll, log_fn=print):
    """Adam then L-BFGS on the inverse loss; jointly fits weights and nu.

    Returns (history, nu_trajectory) where nu_trajectory is nu after every step.
    """
    params = list(model.parameters()) + list(nu_module.parameters())
    history = []
    nu_traj = []

    # --- Stage 1: Adam ---
    optimizer = torch.optim.Adam(params, lr=cfg.adam_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.lr_decay_factor, patience=cfg.lr_decay_patience
    )
    t0 = time.time()
    for it in range(1, cfg.adam_iters + 1):
        optimizer.zero_grad()
        loss, parts = inverse.inverse_loss(model, nu_module, meas, coll)
        loss.backward()
        optimizer.step()
        scheduler.step(parts["total"])
        history.append(parts)
        nu_traj.append(parts["nu"])
        if it % cfg.log_every == 0 or it == 1:
            lr = optimizer.param_groups[0]["lr"]
            log_fn(
                f"[adam {it:>6}/{cfg.adam_iters}] total={parts['total']:.3e} "
                f"data={parts['data']:.3e} pde={parts['pde']:.3e} "
                f"nu={parts['nu']:.5f} lr={lr:.1e}"
            )
    log_fn(f"Adam stage done in {time.time() - t0:.1f}s")

    # --- Stage 2: L-BFGS full-batch fine-tune ---
    optimizer = torch.optim.LBFGS(
        params,
        max_iter=cfg.lbfgs_iters,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )
    state = {"it": 0}

    def closure():
        optimizer.zero_grad()
        loss, parts = inverse.inverse_loss(model, nu_module, meas, coll)
        loss.backward()
        state["it"] += 1
        history.append(parts)
        nu_traj.append(parts["nu"])
        if state["it"] % cfg.log_every == 0 or state["it"] == 1:
            log_fn(
                f"[lbfgs {state['it']:>6}] total={parts['total']:.3e} "
                f"data={parts['data']:.3e} pde={parts['pde']:.3e} nu={parts['nu']:.5f}"
            )
        return loss

    t0 = time.time()
    optimizer.step(closure)
    log_fn(f"L-BFGS stage done in {time.time() - t0:.1f}s ({state['it']} evals)")

    return history, nu_traj
