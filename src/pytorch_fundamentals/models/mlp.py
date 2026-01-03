from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 784
    hidden_dims: List[int] = None
    dropout: float = 0.2
    num_classes: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            object.__setattr__(self, "hidden_dims", [256, 128])


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        dims = [cfg.input_dim] + list(cfg.hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=cfg.dropout))
        layers.append(nn.Linear(dims[-1], cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)
