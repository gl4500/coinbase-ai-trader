"""
Portable CNN training script for cloud GPU runs (Colab/Kaggle).

Self-contained: avoids importing backend.cnn_agent (which pulls FastAPI,
aiosqlite, ollama clients, etc. into the import graph).

Loads a per-product dataset cache produced by backend.cnn_agent
(`cnn_dataset_cache.pt`) and trains the same SignalCNN architecture using
the same hyperparameters as production. Saves three artifacts that drop
into the backend filesystem unchanged:

  - cnn_model.pt              — {arch, n_channels, state_dict} (matches CNNAgent.save_model)
  - cnn_best_loss.txt         — single float as string (matches _BEST_LOSS_PATH)
  - train_cloud_progress.json — {status, epoch_log, best_val_loss, n_train, n_val}

Usage (Colab):
    !python tools/train_cloud.py \
        --cache backend/cnn_dataset_cache.pt \
        --out-model backend/cnn_model.pt \
        --out-best-loss backend/cnn_best_loss.txt \
        --out-progress backend/train_cloud_progress.json \
        --epochs 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# ── Constants — keep in lockstep with backend/agents/cnn_agent.py ────────────
N_CHANNELS = 27
SEQ_LEN = 60
_LABEL_SMOOTH = 0.05
_FORWARD_HOURS = 4

# Mirrors backend.cnn_agent._TRAINING_CONSTANT_CHANNELS (cnn_agent.py:284).
# Channels that are constant-zero during training (cache build skips them);
# inference must zero the same channels to avoid train/serve skew. Ch 20
# (funding rate) is masked here too — fapi.binance is geo-blocked from the
# US (HTTP 451), so the cache's funding column is silently zero (see #80).
DEFAULT_MASK = frozenset({10, 11, 20, 24, 25, 26})

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model — mirrors backend.cnn_agent SignalCNN exactly (arch="glu2") ────────
class GatedConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.conv_main = nn.Conv1d(in_ch, out_ch, kernel, padding=padding)
        self.conv_gate = nn.Conv1d(in_ch, out_ch, kernel, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = self.conv_main(x) * torch.sigmoid(self.conv_gate(x))
        return self.bn(out)


class SignalCNN(nn.Module):
    arch = "glu2"

    def __init__(self, n_ch: int = N_CHANNELS, lstm_drop: float = 0.2, fc_drop: float = 0.3):
        super().__init__()
        self.c1 = GatedConv1d(n_ch, 32)
        self.c2 = GatedConv1d(32, 64)
        self.p2 = nn.MaxPool1d(2)
        self.c3 = GatedConv1d(64, 128)
        self.p3 = nn.MaxPool1d(2)
        self.c4 = GatedConv1d(128, 128)
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=lstm_drop)
        self.drop = nn.Dropout(fc_drop)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x); x = self.p2(x)
        x = self.c3(x); x = self.p3(x)
        x = self.c4(x)
        x = x.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.drop(x)
        return self.fc(x)


# ── Helpers ──────────────────────────────────────────────────────────────────
def apply_mask(X: torch.Tensor, mask_channels: frozenset) -> torch.Tensor:
    """Zero out channels listed in mask_channels. X shape: (N, n_ch, seq_len)."""
    if not mask_channels:
        return X
    X = X.clone()
    for idx in mask_channels:
        X[:, idx, :] = 0.0
    return X


def _smooth(y: torch.Tensor, eps: float) -> torch.Tensor:
    return y * (1.0 - 2.0 * eps) + eps


def save_prod_model(model: nn.Module, path: str, n_channels: int = N_CHANNELS) -> None:
    """Write {arch, n_channels, state_dict} dict — matches CNNAgent.save_model
    in backend/agents/cnn_agent.py:1420 so the artifact loads unchanged on the
    backend side."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "arch": getattr(model, "arch", "glu2"),
            "n_channels": int(n_channels),
            "state_dict": model.state_dict(),
        },
        path,
    )


def write_best_loss(path: str, value: float) -> None:
    """Write single float as string — matches _BEST_LOSS_PATH format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(str(float(value)))


def write_progress_json(
    path: str,
    *,
    status: str,
    epoch_log: List[Dict],
    best_val_loss: Optional[float] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Write training progress in the format the backend hot-reload endpoint
    will consume (#69). status ∈ {running, completed, failed}."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict = {"status": status, "epoch_log": list(epoch_log)}
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    if n_train is not None:
        payload["n_train"] = int(n_train)
    if n_val is not None:
        payload["n_val"] = int(n_val)
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _uniqueness_from_indices(indices: List[int], forward_hours: int, n_candles: int) -> List[float]:
    """López de Prado ch. 4 concurrency-based weighting. Mirrors
    backend.cnn_agent._compute_uniqueness."""
    if not indices:
        return []
    horizon = n_candles + forward_hours + 1
    concurrency = [0] * horizon
    for idx in indices:
        for t in range(idx, min(idx + forward_hours + 1, horizon)):
            concurrency[t] += 1
    weights = []
    for idx in indices:
        win = concurrency[idx: min(idx + forward_hours + 1, horizon)]
        win = [c for c in win if c > 0]
        weights.append(sum(1.0 / c for c in win) / len(win) if win else 1.0)
    return weights


# ── Dataset loader ───────────────────────────────────────────────────────────
def load_dataset(cache_path: str) -> Tuple[List[torch.Tensor], List[float], List[float]]:
    """Load per-product cache and flatten to (X_list, y_list, w_list).
    Cache schema must match backend cnn_agent _DATASET_CACHE_VERSION ≥ 6."""
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    products = blob.get("products") if isinstance(blob, dict) else None
    if not isinstance(products, dict):
        raise RuntimeError(f"unrecognised cache format at {cache_path}")
    X_list, y_list, w_list = [], [], []
    for pid, entry in products.items():
        X = entry.get("X", [])
        y = entry.get("y", [])
        if not X or not y:
            continue
        indices = entry.get("indices", [])
        n_last = int(entry.get("last_n", 0))
        w = _uniqueness_from_indices(indices, _FORWARD_HOURS, n_last) if indices else [1.0] * len(X)
        for xi, yi, wi in zip(X, y, w):
            X_list.append(xi if hasattr(xi, "shape") else torch.tensor(xi, dtype=torch.float32))
            y_list.append(float(yi.item()) if hasattr(yi, "item") else float(yi))
            w_list.append(float(wi))
    return X_list, y_list, w_list


# ── Training ─────────────────────────────────────────────────────────────────
def run_training(
    cache_path: str,
    out_model: str,
    out_best_loss: str,
    out_progress: str,
    *,
    mask: frozenset = DEFAULT_MASK,
    epochs: int = 16,
    batch_anchor: int = 64,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop: int = 8,
    seed: int = 42,
    val_frac: float = 0.2,
) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    write_progress_json(out_progress, status="running", epoch_log=[])

    t0 = time.time()
    X_list, y_list, w_list = load_dataset(cache_path)
    n = len(X_list)
    if n < 100:
        write_progress_json(out_progress, status="failed", epoch_log=[],
                            extra={"error": f"too few samples: {n}"})
        return {"error": f"too few samples: {n}"}

    X_all = apply_mask(torch.stack(X_list), mask).to(_DEVICE)
    y_all = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)
    w_all = torch.tensor(w_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)

    # Naive 80/20 split (preserves chronological order if cache iteration is stable)
    s = max(1, int(n * (1.0 - val_frac)))
    X_tr, X_val = X_all[:s], X_all[s:]
    y_tr, y_val = y_all[:s], y_all[s:]
    w_tr, w_val = w_all[:s], w_all[s:]

    n_pos = int(y_tr.sum().item()); n_neg = len(y_tr) - n_pos
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_w = torch.tensor([pos_weight_val], dtype=torch.float32, device=_DEVICE)

    n_train = len(X_tr)
    if   n_train >= 200_000: BATCH = 256
    elif n_train >= 50_000:  BATCH = 128
    else:                    BATCH = 64
    lr = base_lr * (BATCH / batch_anchor) ** 0.5

    model = SignalCNN(n_ch=N_CHANNELS).to(_DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    epoch_log: List[Dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=_DEVICE)
        losses = []
        for start in range(0, n_train, BATCH):
            idx = perm[start: start + BATCH]
            opt.zero_grad()
            per_sample = F.binary_cross_entropy_with_logits(
                model(X_tr[idx]),
                _smooth(y_tr[idx], _LABEL_SMOOTH),
                pos_weight=pos_w, reduction="none",
            )
            wi = w_tr[idx]
            loss = (per_sample * wi).sum() / wi.sum().clamp(min=1e-9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
        tl = sum(losses) / max(1, len(losses))

        model.eval()
        with torch.no_grad():
            per = F.binary_cross_entropy_with_logits(
                model(X_val), y_val, pos_weight=pos_w, reduction="none",
            )
            vl = float((per * w_val).sum() / w_val.sum().clamp(min=1e-9))
        sched.step(vl)
        epoch_log.append({"epoch": ep, "train_loss": round(tl, 4),
                          "val_loss": round(vl, 4),
                          "lr": opt.param_groups[0]["lr"]})

        write_progress_json(out_progress, status="running",
                            epoch_log=epoch_log, n_train=n_train, n_val=len(X_val))

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_prod_model(model, out_model, n_channels=N_CHANNELS)
    write_best_loss(out_best_loss, best_val)
    write_progress_json(
        out_progress, status="completed", epoch_log=epoch_log,
        best_val_loss=best_val, n_train=n_train, n_val=len(X_val),
        extra={"duration_secs": round(time.time() - t0, 1)},
    )
    return {
        "n_train": n_train, "n_val": len(X_val),
        "best_val_loss": round(best_val, 4),
        "stopped_epoch": epoch_log[-1]["epoch"] if epoch_log else 0,
        "duration_secs": round(time.time() - t0, 1),
    }


def _resolve_default(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


if __name__ == "__main__":
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",         default=_resolve_default("backend/cnn_dataset_cache.pt", repo_root))
    ap.add_argument("--out-model",     default=_resolve_default("backend/cnn_model.pt",        repo_root))
    ap.add_argument("--out-best-loss", default=_resolve_default("backend/cnn_best_loss.txt",   repo_root))
    ap.add_argument("--out-progress",  default=_resolve_default("backend/train_cloud_progress.json", repo_root))
    ap.add_argument("--epochs",        type=int,   default=16)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--val-frac",      type=float, default=0.2)
    args = ap.parse_args()

    print(f"device={_DEVICE} cache={args.cache}")
    result = run_training(
        cache_path=args.cache,
        out_model=args.out_model,
        out_best_loss=args.out_best_loss,
        out_progress=args.out_progress,
        epochs=args.epochs,
        seed=args.seed,
        val_frac=args.val_frac,
    )
    print(json.dumps(result, indent=2))
