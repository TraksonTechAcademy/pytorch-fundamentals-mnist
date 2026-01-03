from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())
