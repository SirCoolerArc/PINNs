"""Phase 2 driver: inverse reconstruction from sparse, noisy velocity data.

From fully scattered, noisy (u, v) samples only -- with pressure hidden -- the
PINN reconstructs the full velocity field, the unobserved pressure field, and an
unknown viscosity nu (a learnable parameter started deliberately off-truth).

Trains with a data-misfit + PDE-residual loss (no IC/BC, no pressure term),
evaluates relative-L2 across time slices, recovers nu, and saves showcase
figures (fields, error maps, vorticity, pressure, training + nu-convergence
curves) plus an "inverse" block in results/metrics.json.

Run from the Week4 directory:  python run_inverse.py
"""

from __future__ import annotations

import json

import torch

from src import config as cfg_mod
from src import data, inverse, pde, train, utils, viz
from src.config import Config


# --- Inverse-problem settings ---
N_MEAS = 4000          # scattered noisy velocity measurements
NOISE = 0.03           # Gaussian noise as a fraction of the signal std
NU_INIT = 0.05         # deliberately wrong start (true nu = 0.01)


def vorticity(model, x, y, t):
    """omega = v_x - u_y from the model at the given (grad-enabled) points."""
    u, v, _ = pde.predict_uvp(model, x, y, t)
    v_x = pde.grad(v, x)
    u_y = pde.grad(u, y)
    return v_x - u_y


def evaluate(model, cfg: Config, device, eval_times, grid_n=100):
    per_time = {}
    saved = {}
    for t in eval_times:
        x, y, tt, shape = data.grid_eval_points(cfg, t=t, n=grid_n, device=device)
        for z in (x, y, tt):
            z.requires_grad_(True)

        u_p, v_p, p_p = pde.predict_uvp(model, x, y, tt)
        u_t, v_t, p_t = data.tgv_fields(x, y, tt, cfg.nu)

        # True vorticity of the TGV: omega = -2 cos(x) cos(y) exp(-2 nu t).
        w_p = vorticity(model, x, y, tt)
        v_x_t = pde.grad(v_t, x)
        u_y_t = pde.grad(u_t, y)
        w_t = v_x_t - u_y_t

        # Pressure defined up to an additive constant -> mean-subtract.
        p_p_c = p_p - p_p.mean()
        p_t_c = p_t - p_t.mean()

        per_time[f"t={t:g}"] = {
            "u": utils.relative_l2(u_p, u_t),
            "v": utils.relative_l2(v_p, v_t),
            "p": utils.relative_l2(p_p_c, p_t_c),
        }
        saved[t] = dict(
            x=x.detach().cpu().numpy(), y=y.detach().cpu().numpy(), shape=shape,
            u_p=u_p.detach().cpu().numpy(), v_p=v_p.detach().cpu().numpy(),
            p_p=p_p_c.detach().cpu().numpy(), w_p=w_p.detach().cpu().numpy(),
            u_t=u_t.detach().cpu().numpy(), v_t=v_t.detach().cpu().numpy(),
            p_t=p_t_c.detach().cpu().numpy(), w_t=w_t.detach().cpu().numpy(),
        )
    return per_time, saved


def main():
    cfg = Config(
        n_layers=6,
        n_units=64,
        n_collocation=15000,
        adam_iters=3000,
        lbfgs_iters=2000,
        log_every=250,
    )
    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.prefer_cuda)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    print(f"device: {device}")

    from src.model import PINN
    model = PINN(cfg).to(device)
    nu_module = inverse.LearnableNu(init=NU_INIT).to(device)
    print(f"nu init={nu_module.value:.5f}  true={cfg.nu}")

    meas, coll = train.make_inverse_data(cfg, N_MEAS, NOISE, device, generator)
    history, nu_traj = train.train_inverse(model, nu_module, cfg, meas, coll)

    nu_est = nu_module.value
    nu_rel_err = inverse.nu_error(nu_est, cfg.nu)
    print(f"\nnu recovered = {nu_est:.5f}  (true {cfg.nu}, error {nu_rel_err:.2%})")

    eval_times = [cfg.t_min, 0.5 * cfg.t_max, cfg.t_max]
    per_time, saved = evaluate(model, cfg, device, eval_times)
    agg = {k: sum(per_time[s][k] for s in per_time) / len(per_time)
           for k in ("u", "v", "p")}
    print("=== Inverse reconstruction relative-L2 ===")
    for s, m in per_time.items():
        print(f"  {s}: u={m['u']:.4%}  v={m['v']:.4%}  p={m['p']:.4%}")
    print(f"  mean: u={agg['u']:.4%}  v={agg['v']:.4%}  p={agg['p']:.4%}")

    # --- Figures at the mid time slice ---
    cfg_mod.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t_show = eval_times[1]
    s = saved[t_show]
    for field, pred, true in (
        ("u", s["u_p"], s["u_t"]),
        ("v", s["v_p"], s["v_t"]),
        ("p (mean-subtracted)", s["p_p"], s["p_t"]),
    ):
        fname = f"inverse_{field.split()[0]}_t{t_show:g}.png"
        viz.plot_field_comparison(
            s["x"], s["y"], pred, true, s["shape"],
            title=f"{field}  (t={t_show:g})",
            path=cfg_mod.FIGURES_DIR / fname,
        )
    viz.plot_vorticity(
        s["x"], s["y"], s["w_p"], s["w_t"], s["shape"],
        cfg_mod.FIGURES_DIR / f"inverse_vorticity_t{t_show:g}.png", t=t_show,
    )
    viz.plot_loss_curve(history, cfg_mod.FIGURES_DIR / "inverse_loss.png")
    viz.plot_param_curve(nu_traj, cfg.nu, cfg_mod.FIGURES_DIR / "inverse_nu.png", name="nu")
    print(f"figures -> {cfg_mod.FIGURES_DIR}")

    # --- Metrics ---
    metrics = {}
    if cfg_mod.METRICS_PATH.exists():
        metrics = json.loads(cfg_mod.METRICS_PATH.read_text())
    metrics["inverse"] = {
        "nu_true": cfg.nu,
        "nu_init": NU_INIT,
        "nu_recovered": nu_est,
        "nu_rel_error": nu_rel_err,
        "n_measurements": N_MEAS,
        "noise": NOISE,
        "rel_l2_mean": agg,
        "rel_l2_per_time": per_time,
        "config": {
            "n_layers": cfg.n_layers, "n_units": cfg.n_units,
            "n_collocation": cfg.n_collocation,
            "adam_iters": cfg.adam_iters, "lbfgs_iters": cfg.lbfgs_iters,
            "seed": cfg.seed,
        },
    }
    cfg_mod.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"metrics -> {cfg_mod.METRICS_PATH}")


if __name__ == "__main__":
    main()
