"""
Custom sine activation layer used in PINN networks.
"""

import torch
from torch import nn, Tensor


class SinActivation(nn.Module):
    """Applies element-wise sine: y = sin(x)."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return torch.sin(x)

