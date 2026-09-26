"""Permutation feature importance for the trained CNN-LSTM signal model.

For a trained CNN, this script measures how much each of the 27 input channels
contributes to validation-loss reduction by:
  1. Computing baseline BCE on the held-out (last 20% chronological) split.
  2. For each channel c, replacing channel c with a random permutation of itself
     across two axes — cross-sample and within-sample-time — running inference
     again, and recording the loss delta.
  3. Repeating with K random seeds to estimate variance.

Channels with delta near zero (relative to the per-channel std across seeds)
are not being used by the model and can be dropped to free gradient signal for
the channels that matter.

Run:
    cd C:\\Users\\gl450\\polymarket_app\\backend
    .\\.venv\\Scripts\\python.exe -m tools.permutation_importance

Reads CNN_ARCH from env to pick the checkpoint. Output CSV lands at
backend/cnn_perm_importance_<arch>.csv. Inference only — does not touch
the live backend, the dataset cache, or any model checkpoint.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

# When invoked as a script (python -m tools.permutation_importance from inside
# backend/), backend/ is automatically on sys.path. When invoked as a module
# from a deeper cwd, ensure imports resolve.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ── Pure permutation primitives (tested in tests/test_permutation_importance.py)


def permute_channel_cross_sample(
    x: torch.Tensor, ch: int, generator: torch.Generator
) -> torch.Tensor:
    """Return a copy of x with channel `ch` permuted across the sample (N) axis.

    Args:
      x:         tensor of shape [N, C, T]
      ch:        channel index to permute
      generator: torch.Generator for reproducible shuffling

    Other channels are untouched. Input is not mutated.
    """
    out = x.clone()
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)
    out[:, ch, :] = x[perm, ch, :]
    return out


def permute_channel_within_sample_time(
    x: torch.Tensor, ch: int, generator: torch.Generator
) -> torch.Tensor:
    """Return a copy of x with channel `ch` shuffled along the time (T) axis
    independently per sample. Other channels untouched. Input not mutated.
    """
    out = x.clone()
    n, _, t = x.shape
    for i in range(n):
        perm = torch.randperm(t, generator=generator)
        out[i, ch, :] = x[i, ch, perm]
    return out


def weighted_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> float:
    """Compute uniqueness-weighted BCE-with-logits, matching _train_arch.

    Mirrors the validation reduction at agents/cnn_agent.py:2581-2587 so the
    baseline number printed here is directly comparable to per-epoch val_loss
    in the training logs.
    """
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    denom = weights.sum().clamp(min=1e-9)
    return float((per_sample * weights).sum() / denom)


# ── Dataset reconstruction (mirrors _build_dataset's assembly, no-recompute)


def _load_val_split() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the per-product cache from disk and reproduce the same chronological
    80/20 split that _train_arch uses (X_all[:split] = train, X_all[split:] = val).

    Returns: (X_val, y_val, w_val) on the configured device.
    """
    from agents.cnn_agent import (
        _DATASET_CACHE_PATH,
        _DEVICE,
        _TRAINING_CONSTANT_CHANNELS,
        N_CHANNELS,
        SEQ_LEN,
        _compute_uniqueness,
        _dataset_schema,
        _load_pp_cache,
    )

    _FORWARD_HOURS = 4
    _LABEL_THRESH = 0.003

    schema = _dataset_schema(SEQ_LEN, _FORWARD_HOURS, _LABEL_THRESH, N_CHANNELS)
    cache = _load_pp_cache(_DATASET_CACHE_PATH, schema)
    if not cache:
        raise RuntimeError(
            f"No matching dataset cache at {_DATASET_CACHE_PATH}. "
            f"Cache must be at version {schema['version']} (current schema)."
        )

    # Cache stores entry["X"] as list-of-tensors (each [C,T]); _build_dataset
    # uses torch.stack on the same items, so we mirror that.
    X_list, y_list, w_list = [], [], []
    for pid in sorted(cache.keys()):
        entry = cache[pid]
        if not entry:
            continue
        X_list.extend(entry.get("X", []))
        y_list.extend(entry.get("y", []))
        w_list.extend(
            _compute_uniqueness(entry.get("indices", []), _FORWARD_HOURS, int(entry["last_n"]))
        )

    if len(X_list) < 4:
        raise RuntimeError(f"Only {len(X_list)} samples in cache; need ≥4")
    if len(w_list) != len(X_list):
        w_list = [1.0] * len(X_list)

    X_all = torch.stack(
        [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in X_list]
    ).to(_DEVICE)
    # Apply mask AFTER stacking — currently _TRAINING_CONSTANT_CHANNELS is
    # frozenset() (empty) post-#86, so this is a no-op; future re-masks Just Work.
    for ch in _TRAINING_CONSTANT_CHANNELS:
        X_all[:, ch, :] = 0.0
    y_all = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)
    w_all = torch.tensor(w_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)

    split = max(1, int(len(X_list) * 0.8))
    return X_all[split:], y_all[split:], w_all[split:]


def _load_model():
    """Load checkpoint matching the active CNN_ARCH; return (model, arch)."""
    from agents.cnn_agent import (
        _ARCH_REGISTRY,
        _DEVICE,
        N_CHANNELS,
        _active_arch,
        _model_path_for,
    )

    arch = _active_arch()
    if arch not in _ARCH_REGISTRY:
        raise RuntimeError(f"Unknown CNN_ARCH={arch!r}; registry={list(_ARCH_REGISTRY)}")

    ckpt_path = _model_path_for(arch)
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"No checkpoint at {ckpt_path} for arch={arch!r}")

    model = _ARCH_REGISTRY[arch](n_ch=N_CHANNELS).to(_DEVICE)
    blob = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
    state = blob.get("model_state") or blob.get("state_dict") or blob
    model.load_state_dict(state)
    model.eval()
    return model, arch


# Channel name table — extracted from cnn_agent.py:1164-1370 comments. Kept
# here (not in cnn_agent) so this debug tool doesn't widen the agent's API.
CHANNEL_NAMES: Tuple[str, ...] = (
    "norm_close",
    "log_volume",
    "hl_range",
    "body",
    "rsi14",
    "macd_hist",
    "ema9_dist",
    "ema21_dist",
    "bb_pos",
    "ret_1",
    "bid",
    "ask",
    "mfi14",
    "obv_slope",
    "stoch_rsi_k",
    "adx14",
    "vwap_dist",
    "fast_rsi_1h",
    "velocity_1h",
    "vol_zscore",
    "funding_rate",
    "btc_corr_20",
    "hour_sin",
    "hour_cos",
    "ivrv_20",
    "ivrv_60",
    "vol_sentiment",
)


@torch.no_grad()
def _val_bce(model, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, batch: int = 256) -> float:
    """Batched val-loss compute to avoid OOM on big val tensors."""
    losses, weights_acc = 0.0, 0.0
    n = X.shape[0]
    for s in range(0, n, batch):
        e = min(s + batch, n)
        logits = model(X[s:e])
        per = F.binary_cross_entropy_with_logits(logits, y[s:e], reduction="none")
        losses += float((per * w[s:e]).sum())
        weights_acc += float(w[s:e].sum())
    return losses / max(weights_acc, 1e-9)


def main(k_seeds: int = 5, out_dir: str | None = None) -> str:
    """Run permutation importance and write CSV. Returns the output path."""
    out_dir = out_dir or _BACKEND_DIR

    print("[perm] loading model + val split...", flush=True)
    model, arch = _load_model()
    X_val, y_val, w_val = _load_val_split()
    n_val = X_val.shape[0]
    print(f"[perm] arch={arch} val_samples={n_val:,} channels={X_val.shape[1]}", flush=True)

    baseline = _val_bce(model, X_val, y_val, w_val)
    print(f"[perm] baseline_val_bce = {baseline:.6f}", flush=True)

    rows: List[dict] = []
    n_ch = X_val.shape[1]
    t0 = time.time()
    for ch in range(n_ch):
        for axis_name, fn in (
            ("sample", permute_channel_cross_sample),
            ("time", permute_channel_within_sample_time),
        ):
            seed_losses = []
            for seed in range(k_seeds):
                gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + ch)
                # Permute on CPU then move target channel back — cheaper than
                # generating on the device tensor when N is large.
                X_cpu = X_val.cpu()
                X_perm = fn(X_cpu, ch, gen).to(X_val.device)
                seed_losses.append(_val_bce(model, X_perm, y_val, w_val))
                del X_perm, X_cpu

            mean_loss = sum(seed_losses) / k_seeds
            std_loss = (sum((x - mean_loss) ** 2 for x in seed_losses) / k_seeds) ** 0.5
            delta = mean_loss - baseline
            rows.append(
                {
                    "ch_idx": ch,
                    "ch_name": CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}",
                    "axis": axis_name,
                    "baseline": round(baseline, 6),
                    "shuffled_mean": round(mean_loss, 6),
                    "shuffled_std": round(std_loss, 6),
                    "delta": round(delta, 6),
                    "k_seeds": k_seeds,
                }
            )
        elapsed = time.time() - t0
        print(
            f"[perm] ch {ch:2d} ({CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else '?':<14s}) "
            f"sample_d={rows[-2]['delta']:+.4f}  time_d={rows[-1]['delta']:+.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    out_path = os.path.join(out_dir, f"cnn_perm_importance_{arch}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[perm] wrote {out_path}", flush=True)

    sample_rows = [r for r in rows if r["axis"] == "sample"]
    sample_rows.sort(key=lambda r: -r["delta"])
    print("\n  rank ch  name              base    shuf     delta    flag")
    for rk, r in enumerate(sample_rows, 1):
        sigma = max(r["shuffled_std"], 1e-9)
        z = r["delta"] / sigma if sigma > 0 else 0.0
        flag = (
            "  HEAVY"
            if r["delta"] > 0.005
            else "  used"
            if r["delta"] > 0.001
            else "  marginal"
            if r["delta"] > 0.0
            else "  NOISE"
        )
        print(
            f"  {rk:>3}  {r['ch_idx']:>2}  {r['ch_name']:<14s} "
            f"{r['baseline']:.4f}  {r['shuffled_mean']:.4f}  {r['delta']:+.4f}  z={z:+.2f}{flag}"
        )
    return out_path


if __name__ == "__main__":
    main()
