"""
Fully-connected network with sine activations for PINN.
"""

from typing import Sequence

import torch
from torch import nn, Tensor

from .activation import SinActivation


class FCN(nn.Module):
    """[in] → hidden*L (sin) → [out]."""

    def __init__(
        self,
        in_features: int,
        hidden: int | Sequence[int] = 64,
        out_features: int = 1,
    ) -> None:
        super().__init__()

        if isinstance(hidden, int):
            hidden = [hidden] * 4  # default depth = 4

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), SinActivation()]
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.model(x)
