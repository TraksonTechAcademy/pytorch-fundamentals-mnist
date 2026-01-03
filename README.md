# PyTorch Fundamentals (MNIST) — Clean, Reproducible Baseline

This repository is a **research-ready PyTorch baseline** that demonstrates the essentials Prof. Kim–style AI labs care about:

- **Clean project structure** (data / model / train / eval / utils)
- **Reproducible experiments** (seeds, configs)
- **Proper logging** (TensorBoard + CSV)
- **Checkpointing** (best + per-epoch)
- **Readable code** with minimal magic

> Goal: provide a small but high-signal codebase proving deep learning readiness and research engineering discipline.

---

## What this project does

We train a simple **MLP classifier** on **MNIST**.

- Input: 28×28 grayscale images (flattened to 784)
- Model: MLP with configurable hidden layers + dropout
- Output: 10-class digit prediction

This is intentionally simple: the focus is on **correct experimental practice**, not model novelty.

---

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2) Train
```bash
python -m pytorch_fundamentals.train --config configs/default.yaml
```

MNIST will auto-download on first run.

### 3) Monitor (TensorBoard)
```bash
tensorboard --logdir runs
```

### 4) Evaluate a checkpoint
```bash
python -m pytorch_fundamentals.eval --checkpoint checkpoints/<RUN_NAME>-best.pt
```

---

## Repository structure

```
pytorch-fundamentals-mnist/
  configs/
    default.yaml              # experiment config (seed, data, model, train)
  src/pytorch_fundamentals/
    data/mnist.py             # dataset + dataloaders
    models/mlp.py             # model definition
    train.py                  # training loop (AMP + grad-clip + logging)
    eval.py                   # evaluation script
    utils/                    # seed, metrics, logging, checkpoints
  scripts/run_train.sh
  tests/test_smoke.py
```

---

## Reproducibility & experiment discipline

- All hyperparameters live in `configs/default.yaml`
- Deterministic seed setup in `utils/seed.py`
- Metrics recorded in:
  - `runs/<run_name>/` (TensorBoard)
  - `runs/<run_name>/metrics.csv`

---

## Extending to Agentic AI / RL

This repo is designed to be extended into **agentic decision systems**:

- Replace MNIST with an environment `(state → action → reward)`
- Use the same structure:
  - `data/` → environment wrappers / replay buffers
  - `models/` → policy / value networks
  - `train.py` → rollout + update loop + evaluation
  - `utils/` → logging + checkpointing + metrics

---

## Notes for reviewers (Prof. Kim-style)
If you skim only three things:
1. `configs/default.yaml` (controls experiments)
2. `train.py` (clean training loop with AMP/clip/logging)
3. `utils/logger.py` (TensorBoard + CSV)

---

## License
MIT
