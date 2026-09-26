"""Dry-run SignalCNNGluM on CPU to confirm forward+backward complete.

Usage:  cd backend && python -m tools.dryrun_glum
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, ".")

from agents.cnn_agent import N_CHANNELS, SEQ_LEN, SignalCNNGluM


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = SignalCNNGluM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dryrun] SignalCNNGluM params: {n_params:,}")

    for batch in (1, 8, 64, 256):
        x = torch.randn(batch, N_CHANNELS, SEQ_LEN, device=device)
        y = torch.randint(0, 2, (batch, 1), device=device, dtype=torch.float32)

        t0 = time.perf_counter()
        model.train()
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        model.zero_grad()
        elapsed = time.perf_counter() - t0
        print(
            f"[dryrun] batch={batch:>4}  out={tuple(logits.shape)}  "
            f"loss={loss.item():.4f}  fwd+bwd={elapsed * 1000:.1f}ms"
        )

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, N_CHANNELS, SEQ_LEN, device=device)
        prob = float(torch.sigmoid(model(x)).item())
    print(f"[dryrun] eval prob (single sample): {prob:.4f}")
    print("[dryrun] OK — no hang, no shape mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
