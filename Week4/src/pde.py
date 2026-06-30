"""Autograd derivatives, Navier-Stokes residuals, and stream-function -> (u, v)."""

from __future__ import annotations

import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """First derivative d(outputs)/d(inputs) with the graph kept for higher orders."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def velocity_from_stream(psi: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Recover (u, v) from the stream function: u = psi_y, v = -psi_x.

    Continuity u_x + v_y = psi_yx - psi_xy = 0 then holds exactly by construction.
    """
    psi_x = grad(psi, x)
    psi_y = grad(psi, y)
    u = psi_y
    v = -psi_x
    return u, v


def navier_stokes_residual(model, x, y, t, nu):
    """Momentum residuals (f_u, f_v) of the 2D incompressible NS equations.

    `model(x, y, t)` returns (psi, p). Velocities come from the stream function,
    so continuity is exact and only the two momentum equations are penalised:
        f_u = u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy)
        f_v = v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy)

    `nu` may be a float or a tensor (for the inverse problem).
    """
    psi, p = model(x, y, t)
    u, v = velocity_from_stream(psi, x, y)

    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    p_x = grad(p, x)
    p_y = grad(p, y)

    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return f_u, f_v


def predict_uvp(model, x, y, t):
    """Convenience: return (u, v, p) from the model at the given points."""
    psi, p = model(x, y, t)
    u, v = velocity_from_stream(psi, x, y)
    return u, v, p
