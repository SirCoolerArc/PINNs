"""Phase 1 driver: train the forward Taylor-Green PINN and validate it.

Trains a physics-driven PINN (IC + periodic BC + PDE residual, no interior
velocity labels), evaluates the relative-L2 error against the analytical fields
across several time slices, saves comparison figures and the training curve, and
writes the forward metrics to results/metrics.json.

Run from the Week4 directory:  python run_forward.py
"""

from __future__ import annotations

import json

import torch

from src import config as cfg_mod
from src import data, pde, train, utils, viz
from src.config import Config


def evaluate(model, cfg: Config, device, eval_times, grid_n=100):
    """Relative-L2 over a dense grid at each eval time; returns metrics + arrays."""
    per_time = {}
    saved = {}
    for t in eval_times:
        x, y, tt, shape = data.grid_eval_points(cfg, t=t, n=grid_n, device=device)
        x.requires_grad_(True)
        y.requires_grad_(True)
        tt.requires_grad_(True)

        u_p, v_p, p_p = pde.predict_uvp(model, x, y, tt)
        u_t, v_t, p_t = data.tgv_fields(x, y, tt, cfg.nu)

        # Pressure is defined only up to an additive constant -> mean-subtract.
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
            p_p=p_p_c.detach().cpu().numpy(),
            u_t=u_t.detach().cpu().numpy(), v_t=v_t.detach().cpu().numpy(),
            p_t=p_t_c.detach().cpu().numpy(),
        )
    return per_time, saved


def main():
    cfg = Config(
        n_layers=6,
        n_units=64,
        n_collocation=15000,
        n_initial=3000,
        n_boundary=1500,
        adam_iters=2000,
        lbfgs_iters=1000,
        log_every=250,
    )
    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.prefer_cuda)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    print(f"device: {device}")

    from src.model import PINN
    model = PINN(cfg).to(device)

    history = train.train(model, cfg, device, generator)

    eval_times = [cfg.t_min, 0.5 * cfg.t_max, cfg.t_max]
    per_time, saved = evaluate(model, cfg, device, eval_times)

    # Aggregate (mean over eval times) for the headline numbers.
    agg = {k: sum(per_time[s][k] for s in per_time) / len(per_time)
           for k in ("u", "v", "p")}
    print("\n=== Forward validation relative-L2 ===")
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
        fname = f"forward_{field.split()[0]}_t{t_show:g}.png"
        viz.plot_field_comparison(
            s["x"], s["y"], pred, true, s["shape"],
            title=f"{field}  (t={t_show:g})",
            path=cfg_mod.FIGURES_DIR / fname,
        )
    viz.plot_loss_curve(history, cfg_mod.FIGURES_DIR / "forward_loss.png")
    print(f"figures -> {cfg_mod.FIGURES_DIR}")

    # --- Metrics ---
    metrics = {}
    if cfg_mod.METRICS_PATH.exists():
        metrics = json.loads(cfg_mod.METRICS_PATH.read_text())
    metrics["forward"] = {
        "nu": cfg.nu,
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
