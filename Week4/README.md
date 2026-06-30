# Week 4 — Navier–Stokes Physics-Informed Neural Network

A 2D incompressible Navier–Stokes PINN that performs a forward solve and an inverse
field reconstruction: from sparse, noisy velocity samples it recovers the full
velocity field, the unobserved pressure field, and an unknown viscosity `ν`.

The project is organised in phases:

1. **Forward validation** on the Taylor–Green vortex (exact analytical solution) to
   de-risk the residual implementation.
2. **Inverse reconstruction** — the headline result — recovering `(u, v, p, ν)` from
   sparse noisy measurements using the stream-function formulation.

Layout:

- `src/` — library code (config, data, PDE residuals, model, training, viz).
- `notebooks/` — narrated walkthroughs for the forward and inverse phases.
- `results/` — saved figures and final metrics.
- `archive/` — earlier Newton's-cooling PINN, kept for history.

Results and headline numbers will be filled in here as the phases complete.
