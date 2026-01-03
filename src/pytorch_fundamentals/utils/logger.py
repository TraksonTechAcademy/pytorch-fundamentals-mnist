from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter


@dataclass
class RunLogger:
    run_dir: str
    tb: SummaryWriter
    csv_path: str

    @classmethod
    def create(cls, run_dir: str) -> "RunLogger":
        os.makedirs(run_dir, exist_ok=True)
        tb = SummaryWriter(log_dir=run_dir)
        csv_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "split", "loss", "acc"])
        return cls(run_dir=run_dir, tb=tb, csv_path=csv_path)

    def log_scalars(self, step: int, split: str, loss: float, acc: float) -> None:
        self.tb.add_scalar(f"{split}/loss", loss, step)
        self.tb.add_scalar(f"{split}/acc", acc, step)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, split, f"{loss:.6f}", f"{acc:.6f}"])

    def close(self) -> None:
        self.tb.flush()
        self.tb.close()
