"""MLP with tanh activations (optional Fourier-feature input embedding)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class PINN(nn.Module):
    """Fully-connected tanh network mapping (x, y, t) -> (psi, p).

    Inputs are taken as three separate column tensors so the caller can request
    autograd derivatives w.r.t. each coordinate. They are normalised to [-1, 1]
    using the domain bounds before entering the network, which mitigates
    spectral bias on the [0, 2*pi] domain.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        lows = torch.tensor(cfg.domain_lows, dtype=torch.float32)
        highs = torch.tensor(cfg.domain_highs, dtype=torch.float32)
        # Registered as buffers so they move with .to(device) and are saved.
        self.register_buffer("in_low", lows)
        self.register_buffer("in_high", highs)

        dims = [cfg.n_inputs] + [cfg.n_units] * cfg.n_layers + [cfg.n_outputs]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.linears = nn.ModuleList(layers)
        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _normalize(self, inp: torch.Tensor) -> torch.Tensor:
        return 2.0 * (inp - self.in_low) / (self.in_high - self.in_low) - 1.0

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        h = self._normalize(torch.cat([x, y, t], dim=1))
        for layer in self.linears[:-1]:
            h = self.activation(layer(h))
        out = self.linears[-1](h)
        psi = out[:, 0:1]
        p = out[:, 1:2]
        return psi, p
