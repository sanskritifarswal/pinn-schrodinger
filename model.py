import torch
import torch.nn as nn


class SchrodingerMLP(nn.Module):
    """MLP that maps (x, t) -> (u, v), the real and imaginary parts of psi."""

    def __init__(self, hidden_layers: int = 5, hidden_dim: int = 64):
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))  # outputs: [u, v]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: spatial coordinate, shape (N, 1)
            t: time coordinate,    shape (N, 1)
        Returns:
            u: real part of psi,      shape (N, 1)
            v: imaginary part of psi, shape (N, 1)
        """
        xt = torch.cat([x, t], dim=1)
        out = self.net(xt)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v
