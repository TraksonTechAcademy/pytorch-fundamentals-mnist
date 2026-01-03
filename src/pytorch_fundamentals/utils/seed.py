from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = True


def set_seed(cfg: SeedConfig) -> None:
    """Set random seeds for reproducibility."""
    random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
