"""Neural network models for CBF learning (GPU-optimized)."""

import torch
import torch.nn as nn


class NeuralCBF(nn.Module):
    """Neural Control Barrier Function h_theta(x) -> scalar."""

    def __init__(self, input_dim=2, hidden_dim=256, num_layers=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralController(nn.Module):
    """Neural controller u_phi(x) -> 2D control input."""

    def __init__(self, input_dim=2, hidden_dim=256, num_layers=4, output_dim=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualCBF(nn.Module):
    """h(x) = h_pretrained(x) + r_theta(x)

    Pretrained CBF is frozen, only the residual network is trainable.
    Inspired by KNODE hybrid architecture (Pan et al., 2026).
    """

    def __init__(self, pretrained_cbf, hidden_dim=64, num_layers=3):
        super().__init__()
        self.pretrained = pretrained_cbf
        for p in self.pretrained.parameters():
            p.requires_grad = False

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.residual = nn.Sequential(*layers)
        self.residual_hidden_dim = hidden_dim
        self.residual_num_layers = num_layers

    def forward(self, x):
        return self.pretrained(x) + self.residual(x)

    def trainable_parameters(self):
        """Returns only trainable (residual) parameters."""
        return self.residual.parameters()
