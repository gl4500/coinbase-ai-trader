"""
Standalone CNN training worker.
Runs in a separate process so training survives backend restarts.
Progress is written atomically to cnn_train_progress.json.

Usage: python train_worker.py --epochs 50
"""
import sys
import os
import json
import asyncio
import argparse
import time
import logging

# ── Path bootstrap (mirrors main.py) ─────────────────────────────────────────
_root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_site = os.path.join(_root, ".venv", "Lib", "site-packages")
if os.path.isdir(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ───────────────────────────────────────────────────────────────────
_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            os.path.join(_log_dir, "cnn_training.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train_worker")

# ── Progress file ─────────────────────────────────────────────────────────────
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "cnn_train_progress.json")


def _write(data: dict) -> None:
    """Atomic write — never leaves a partial file on disk."""
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, PROGRESS_FILE)


# ── Main training coroutine ───────────────────────────────────────────────────
async def _run(epochs: int, started_at: float) -> None:
    import database
    from agents.cnn_agent import CoinbaseCNNAgent

    await database.init_db()

    agent = CoinbaseCNNAgent(ws_subscriber=None)
    result = await agent.train_on_history(epochs=epochs)

    _write({
        "status":       "failed" if "error" in result else "completed",
        "elapsed_secs": result.get("duration_secs", int(time.time() - started_at)),
        "result":       result,
        "finished_at":  time.time(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    started_at = time.time()
    _write({
        "status":     "running",
        "pid":        os.getpid(),
        "started_at": started_at,
        "result":     None,
    })

    logger.info(f"CNN training worker started — PID={os.getpid()} epochs={args.epochs}")
    try:
        asyncio.run(_run(args.epochs, started_at))
    except Exception as exc:
        logger.error(f"CNN training worker failed: {exc}", exc_info=True)
        _write({
            "status":       "failed",
            "elapsed_secs": int(time.time() - started_at),
            "result":       {"error": str(exc)},
            "finished_at":  time.time(),
        })
        sys.exit(1)

    logger.info("CNN training worker finished successfully")
