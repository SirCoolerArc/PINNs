# Theory — Physics-Informed Neural Networks for Navier–Stokes

This note explains the mathematics behind the project: the equations I am solving, why
a neural network can solve them, the stream-function trick that makes it stable, and how
the inverse problem recovers quantities that were never measured.

## 1. What a PINN is

A Physics-Informed Neural Network represents the solution of a differential equation as a
neural network and trains it so that the equation itself is (approximately) satisfied.

Instead of fitting a network to a table of input–output pairs, I make the network
$\mathcal{N}_\theta$ a candidate solution field and penalise how much it violates the
governing physics. Because the network is differentiable, I can take its derivatives with
**automatic differentiation** — exactly the derivatives the PDE needs — and plug them
straight into the equation. The mismatch (the *residual*) becomes a loss term.

This is the same idea as the earlier Newton's-law-of-cooling PINN, where the network
$T(t)$ was trained so that $\frac{dT}{dt} + k\,(T - T_{\text{env}}) = 0$. Here I scale the
same machinery from a first-order ODE up to the hardest classic PDE in fluid dynamics.

## 2. The governing equations

I solve the **2D incompressible Navier–Stokes equations** (non-dimensional form). For a
velocity field $(u, v)$ and pressure $p$, with kinematic viscosity $\nu$:

**Momentum (x and y):**

$$
u_t + u\,u_x + v\,u_y = -p_x + \nu\,(u_{xx} + u_{yy})
$$

$$
v_t + u\,v_x + v\,v_y = -p_y + \nu\,(v_{xx} + v_{yy})
$$

**Continuity (incompressibility):**

$$
u_x + v_y = 0
$$

Each momentum equation balances inertia (the left-hand side, including the nonlinear
convection terms $u\,u_x$ etc.) against the pressure gradient and viscous diffusion. The
nonlinearity is what makes this hard — it is why fluid dynamics is difficult in general.

## 3. The stream-function trick (exact mass conservation)

The continuity equation $u_x + v_y = 0$ is a hard constraint. Rather than enforce it as an
extra loss term and hope the optimiser balances it, I build it into the architecture.

I have the network output a **stream function** $\psi$ (and pressure $p$), and define the
velocity as its curl:

$$
u = \psi_y, \qquad v = -\psi_x
$$

Then continuity is satisfied *identically*, for any $\psi$ whatsoever:

$$
u_x + v_y = \psi_{yx} - \psi_{xy} \equiv 0
$$

because mixed partial derivatives commute. So the network **cannot** produce a
mass-violating flow — incompressibility holds by construction, not by penalty. In the code
I verify this numerically: the measured divergence $|u_x + v_y|$ sits around $10^{-9}$,
i.e. floating-point round-off. This removes one loss term, which is a real stability win:
the optimiser has one fewer competing objective to balance.

Concretely the network is $\mathcal{N}_\theta:(x, y, t) \mapsto (\psi, p)$, an MLP with
`tanh` activations. All the derivatives above — $\psi_x, \psi_y$, then $u_t, u_x, u_{xx},
\dots$ — are computed with `torch.autograd.grad(..., create_graph=True)`, which keeps the
graph so second derivatives are available.

## 4. Phase 1 — forward validation on the Taylor–Green vortex

Before solving an unknown flow I validate the residual code on one whose exact answer I
know. The **Taylor–Green vortex** is an analytical solution of these equations on
$[0, 2\pi]^2$ with periodic boundaries:

$$
u = -\cos x \sin y \; e^{-2\nu t}, \quad
v = \sin x \cos y \; e^{-2\nu t}, \quad
p = -\tfrac14 (\cos 2x + \cos 2y)\, e^{-4\nu t}
$$

The flow is a grid of counter-rotating vortices whose amplitude decays as $e^{-2\nu t}$ —
viscosity slowly dissipates the energy.

For the forward solve I train on **physics alone**, with three loss terms and *no* interior
velocity labels:

$$
\mathcal{L}_{\text{fwd}}
= \underbrace{\mathcal{L}_{\text{IC}}}_{\text{initial condition at }t=0}
+ \underbrace{\mathcal{L}_{\text{BC}}}_{\text{periodicity on }[0,2\pi]^2}
+ \underbrace{\mathcal{L}_{\text{PDE}}}_{\text{momentum residual}}
$$

where the PDE term is the mean squared momentum residual at scattered *collocation* points:

$$
\mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum \left( f_u^2 + f_v^2 \right),
\quad
f_u = u_t + u u_x + v u_y + p_x - \nu(u_{xx}+u_{yy})
$$

and $f_v$ analogously. If the trained field matches the analytical solution to low relative
$L^2$ error (it does: sub-1% on velocity), the engine is trustworthy. Notice pressure never
appears in the loss — it is inferred purely because it is coupled to the velocity through
the momentum equations.

## 5. Phase 2 — the inverse problem (hidden fluid mechanics)

This is the real point of the project. Now I *pretend I don't know the flow*. I am given
only **sparse, noisy velocity measurements** and must reconstruct everything else.

**Setup.** I sample $N$ scattered points $(x, y, t)$ across space and time, record the exact
velocity $(u, v)$ there, and corrupt it with Gaussian noise (3%). I then throw away three
things: the pressure (never measured), the boundary/initial conditions (not used), and the
value of $\nu$ (treated as unknown).

**Unknowns.** Two things are learned jointly:

1. the network weights $\theta$ (hence the full fields $u, v, p$ everywhere), and
2. the viscosity $\nu$, a single trainable scalar.

To keep $\nu > 0$ I parameterise it through a softplus, $\nu = \operatorname{softplus}(w)$,
so the optimiser sees an unconstrained variable while the physical value stays positive. I
initialise it deliberately *wrong* (5× too large) so that recovering the truth is a genuine
result, not a lucky guess.

**Loss.** Only two terms — a data misfit on the noisy velocity, and the same PDE residual as
before (now with the *learnable* $\nu$ inside it):

$$
\mathcal{L}_{\text{inv}}
= \underbrace{\frac{1}{N_d}\sum \big(|u - u^{\text{meas}}|^2 + |v - v^{\text{meas}}|^2\big)}_{\text{data misfit}}
+ \underbrace{\frac{1}{N_c}\sum \big(f_u^2 + f_v^2\big)}_{\text{PDE residual}}
$$

There is no pressure term and no boundary condition. The data pins down the velocity where
it is measured; the PDE residual then propagates that information everywhere else and, in
doing so, forces a consistent pressure field and a consistent $\nu$ into existence.

**Why $\nu$ is identifiable.** The decay factor $e^{-2\nu t}$ ties the viscosity to *how
fast the flow slows down over time*. If I only had one time slice I could not separate
"weak viscosity" from "strong viscosity, later moment". Sampling across multiple times in
$[0, T]$ is what makes $\nu$ recoverable from velocity alone.

**Why pressure comes for free.** The momentum equations read
$p_x = \nu(u_{xx}+u_{yy}) - (u_t + u u_x + v u_y)$ and similarly for $p_y$. Once the velocity
field is known, these fix the pressure gradient everywhere, so pressure is determined (up to
a constant) by physics. The network never needs a single pressure measurement.

**The pressure gauge.** In incompressible flow only $\nabla p$ appears — pressure is defined
only up to an additive constant. So when I report a pressure error I first subtract the
spatial mean from both the predicted and the true field (per time slice); comparing absolute
pressure would be meaningless.

## 6. Optimisation

I train in two stages, which is a standard and cheap accuracy win over Adam alone:

- **Adam** (learning rate $10^{-3}$, with a plateau scheduler) does the bulk of the work —
  robust, stochastic-friendly, good at escaping bad regions.
- **L-BFGS** then fine-tunes full-batch. It is a quasi-Newton method that uses curvature
  information, so near a good solution it converges fast and precisely. This is also the
  stage where $\nu$ locks tightly onto its true value — visible as a sharp bend in the
  $\nu$-convergence curve.

All random seeds (Python, NumPy, Torch) are fixed for reproducibility, and the final numbers
are written to `results/metrics.json`.

## 7. What the results show

- $\nu$ recovered to within a few percent of ground truth, from noisy data, starting 5× off.
- Velocity reconstructed to sub-1% relative $L^2$.
- Pressure — never measured — reconstructed to a couple of percent.
- Vorticity $\omega = v_x - u_y$, a *derived* quantity involving further derivatives of the
  reconstruction, also comes out clean, which confirms the solution is smooth and physically
  consistent rather than merely correct at the sampled points.

See [README.md](README.md) for the figures and the exact numbers.
