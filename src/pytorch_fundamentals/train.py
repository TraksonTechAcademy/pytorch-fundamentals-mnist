from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from pytorch_fundamentals.data.mnist import MNISTDataConfig, make_loaders
from pytorch_fundamentals.models.mlp import MLP, MLPConfig
from pytorch_fundamentals.utils.checkpoints import save_checkpoint
from pytorch_fundamentals.utils.logger import RunLogger
from pytorch_fundamentals.utils.metrics import accuracy
from pytorch_fundamentals.utils.seed import SeedConfig, set_seed


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    device: str
    amp: bool
    log_every: int
    save_every_epoch: bool


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a clean MNIST baseline in PyTorch.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run_name", type=str, default=None, help="Optional override for run directory name.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Seed
    set_seed(SeedConfig(seed=int(cfg["seed"]), deterministic=True))

    # Data
    data_cfg = MNISTDataConfig(
        data_dir=cfg["data"]["data_dir"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )
    train_loader, val_loader = make_loaders(data_cfg)

    # Model
    mcfg = cfg["model"]
    model = MLP(
        MLPConfig(
            input_dim=int(mcfg["input_dim"]),
            hidden_dims=list(mcfg["hidden_dims"]),
            dropout=float(mcfg["dropout"]),
            num_classes=int(mcfg["num_classes"]),
        )
    )

    # Device
    device = resolve_device(cfg["train"]["device"])
    model.to(device)

    # Training config
    tcfg = TrainConfig(
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
        device=cfg["train"]["device"],
        amp=bool(cfg["train"]["amp"]),
        log_every=int(cfg["train"]["log_every"]),
        save_every_epoch=bool(cfg["train"]["save_every_epoch"]),
    )

    # Optim / loss
    optimizer = optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.amp and device.type == "cuda"))

    # Run dirs
    run_root = cfg["output"]["run_dir"]
    os.makedirs(run_root, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"mnist-mlp-{ts}"
    run_dir = os.path.join(run_root, run_name)
    logger = RunLogger.create(run_dir)

    ckpt_dir = cfg["output"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    best_val_acc = -1.0

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg.epochs}", leave=False)
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if tcfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            acc = accuracy(logits.detach(), y)
            if global_step % tcfg.log_every == 0:
                logger.log_scalars(global_step, "train", float(loss.item()), float(acc))
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
            global_step += 1

        # Validation per epoch
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.log_scalars(global_step, "val", float(val_loss), float(val_acc))

        # Save checkpoints
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": cfg,
        }

        if tcfg.save_every_epoch:
            save_checkpoint(os.path.join(ckpt_dir, f"{run_name}-epoch{epoch}.pt"), state)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(os.path.join(ckpt_dir, f"{run_name}-best.pt"), state)

        print(f"[epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")

    logger.close()
    print(f"Done. TensorBoard logs in: {run_dir}")


if __name__ == "__main__":
    main()
