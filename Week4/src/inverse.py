"""Learnable viscosity nu and inverse-problem loss / metrics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pde


class LearnableNu(nn.Module):
    """Positive learnable viscosity nu = softplus(raw).

    The raw parameter is unconstrained (so any optimizer is happy), while the
    exposed value stays strictly positive. `init` is the desired starting nu;
    the raw value is set by inverting softplus so nu begins exactly there.
    """

    def __init__(self, init: float = 0.05):
        super().__init__()
        # Inverse softplus: raw = log(exp(nu) - 1), stable for the small nu here.
        raw0 = math.log(math.expm1(init))
        self.raw = nn.Parameter(torch.tensor(float(raw0)))

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw)

    @property
    def value(self) -> float:
        return F.softplus(self.raw).item()


def inverse_loss(model, nu_module: LearnableNu, meas, coll, w_data=1.0, w_pde=1.0):
    """Data misfit on sparse noisy (u, v) plus the PDE residual.

    `meas` = (x, y, t, u_noisy, v_noisy) sparse measurement points.
    `coll` = (x, y, t) collocation points for the residual.
    No pressure term and no IC/BC: pressure and the interior field are recovered
    from physics, and nu is identified jointly with the network weights.
    """
    nu = nu_module()

    xm, ym, tm, um, vm = meas
    u_pred, v_pred, _ = pde.predict_uvp(model, xm, ym, tm)
    loss_data = F.mse_loss(u_pred, um) + F.mse_loss(v_pred, vm)

    xc, yc, tc = coll
    f_u, f_v = pde.navier_stokes_residual(model, xc, yc, tc, nu)
    loss_pde = f_u.pow(2).mean() + f_v.pow(2).mean()

    total = w_data * loss_data + w_pde * loss_pde
    parts = {
        "total": total.item(),
        "data": loss_data.item(),
        "pde": loss_pde.item(),
        "nu": nu.item(),
    }
    return total, parts


def nu_error(nu_est: float, nu_true: float) -> float:
    """Relative error |nu_est - nu_true| / nu_true."""
    return abs(nu_est - nu_true) / nu_true
