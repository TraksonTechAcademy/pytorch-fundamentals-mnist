from __future__ import annotations

import argparse
import yaml
import torch
import torch.nn as nn

from pytorch_fundamentals.data.mnist import MNISTDataConfig, make_loaders
from pytorch_fundamentals.models.mlp import MLP, MLPConfig
from pytorch_fundamentals.utils.checkpoints import load_checkpoint
from pytorch_fundamentals.utils.metrics import accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    # Build model
    mcfg = cfg["model"]
    model = MLP(
        MLPConfig(
            input_dim=int(mcfg["input_dim"]),
            hidden_dims=list(mcfg["hidden_dims"]),
            dropout=float(mcfg["dropout"]),
            num_classes=int(mcfg["num_classes"]),
        )
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Data
    data_cfg = MNISTDataConfig(
        data_dir=cfg["data"]["data_dir"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )
    _, val_loader = make_loaders(data_cfg)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            n_batches += 1

    print(f"Eval loss: {total_loss/max(1,n_batches):.4f}")
    print(f"Eval acc : {total_acc/max(1,n_batches):.4f}")


if __name__ == "__main__":
    main()
