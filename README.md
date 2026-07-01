# Reimagining Physics Solutions: Integrating Neural Network Techniques

**Seasons of Code 2025 — Project ID 32**
**Author:** Rishabh Kumar

This repository is my Seasons of Code journey from Python fundamentals to building a
research-style **Physics-Informed Neural Network (PINN)** for the 2D incompressible
Navier–Stokes equations. The early weeks build the groundwork (Python, scientific
libraries, neural networks from scratch, deep-learning frameworks); the final week is a
self-contained project that uses all of it.

## The final project (Week 4)

A 2D **Navier–Stokes PINN** that solves both a forward and an inverse problem. From nothing
but **sparse, noisy velocity measurements**, it reconstructs the full velocity field, the
**pressure field it was never given**, and an **unknown viscosity** — the classic "hidden
fluid mechanics" result on the hardest of the classic PDEs.

- **[Week4/README.md](Week4/README.md)** — results, figures, and how to reproduce them.
- **[Week4/THEORY.md](Week4/THEORY.md)** — the mathematics: the equations, the
  stream-function formulation, and the inverse setup.

Headline results (from noisy velocity data only): viscosity recovered to within ~4%, and
velocity/pressure reconstructed to sub-1% / ~1.6% relative L2 error.

## Weekly progression

| Week | Focus | Contents |
| --- | --- | --- |
| [Week 0](Week0/) | Python fundamentals | Basic Python, file I/O, small exercises |
| [Week 1](Week1/) | Scientific Python | NumPy, Pandas, matplotlib |
| [Week 2](Week2/) | Neural networks from scratch | *Neural Networks from Scratch* implementation |
| [Week 3](Week3/) | Deep-learning frameworks | PyTorch, TensorFlow, first look at PINNs |
| [Week 4](Week4/) | **Navier–Stokes PINN** | Forward validation + inverse reconstruction |

Each week has its own README describing what it covers. The final project is where the
threads come together: automatic differentiation and learnable parameters from the
framework weeks are reused to solve a genuine PDE inverse problem.

## Setup

The project code (Week 4) runs on Python 3.11/3.12 with PyTorch. See
[Week4/requirements.txt](Week4/requirements.txt) for pinned dependencies and
[Week4/README.md](Week4/README.md) for reproduction steps.
