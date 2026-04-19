"""
CNN Retrain Script — Coinbase AI Trader
=========================================
Stops the backend, retrains the CNN model offline against the live SQLite DB,
clears the stale LightGBM filter, then restarts the backend.

Usage:
    python retrain.py                  # stop → train → restart
    python retrain.py --no-restart     # train + clear LGBM only (keep backend stopped)
    python retrain.py --schedule       # register daily 03:00 via Windows Task Scheduler
    python retrain.py --unschedule     # remove the scheduled task

Why stop/restart instead of hitting /api/cnn/train?
  The backend holds the LightGBM model in memory.  Restarting ensures the fresh
  model and pass-through LGBM state both load cleanly with no in-memory residue.
"""

import asyncio
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "backend"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LGBM_PKL    = BACKEND_DIR / "data" / "lgbm_filter.pkl"
DB_PATH     = BACKEND_DIR / "coinbase.db"
BACKEND_URL = "http://localhost:8001/api/status"
TASK_NAME   = "CoinbaseAITrader_Retrain"

NO_RESTART = "--no-restart" in sys.argv
SCHEDULE   = "--schedule"   in sys.argv
UNSCHEDULE = "--unschedule" in sys.argv


# ── Helpers ────────────────────────────────────────────────────────────────────

def _python() -> str:
    return str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def _banner(msg: str) -> None:
    print(f"\n  {'─' * 50}")
    print(f"  {msg}")
    print(f"  {'─' * 50}")


def _stop_backend() -> None:
    """Kill whatever process is listening on port 8001."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        for line in out.splitlines():
            if ":8001" in line and ("LISTENING" in line or "LISTEN" in line):
                pid = line.split()[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    print(f"  [STOP]  Backend stopped (PID {pid})")
                    time.sleep(1.5)   # let the process fully release the port
                    return
        print("  [STOP]  Backend not running — nothing to stop")
    except Exception as e:
        print(f"  [WARN]  Port check failed: {e}")


async def _retrain_lgbm() -> None:
    """
    Retrain LightGBM on historical closed CNN trades using the already-loaded DB.
    This re-calibrates LGBM to the new CNN's output distribution without wiping
    accumulated trade history.  Agnostic to CNN channel count — LGBM only sees
    the scalar cnn_prob output plus market features (rsi, adx, etc.).
    """
    import database
    from data.lgbm_filter import LGBMFilter

    rows = await database.get_lgbm_training_rows()
    n = len(rows)
    print(f"  [LGBM]  {n} closed CNN trades available for LGBM training")

    lgbm = LGBMFilter()
    if LGBM_PKL.exists():
        lgbm.load(str(LGBM_PKL))   # load existing weights as starting point

    metrics = lgbm.train(rows)
    if metrics:
        lgbm.save(str(LGBM_PKL))
        print(f"  [LGBM]  Retrained: n={metrics['n_samples']}  "
              f"win={metrics['win_rate']}%  auc={metrics['auc']}")
    else:
        print(f"  [LGBM]  Not enough data yet ({n} rows, need 50 with both win/loss classes)")
        print(f"          Existing model kept — backend will use pass-through until data accumulates")


def _start_backend() -> bool:
    """Start the backend in the background; wait up to 60 s for it to be ready."""
    python = _python()
    print(f"  [START] Launching backend...")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(
        [python, "main.py"],
        cwd=str(BACKEND_DIR),
        creationflags=creationflags,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BACKEND_URL, timeout=2)
            print(f"  [START] Backend ready at {BACKEND_URL.replace('/api/status','')}")
            return True
        except Exception:
            time.sleep(0.5)
    print(f"  [WARN]  Backend did not respond within 60 s — check logs/backend.log")
    return False


def _schedule_task() -> None:
    """Register a daily 03:00 Windows Task Scheduler job."""
    script = str(Path(__file__).resolve())
    python = _python()
    cmd = [
        "schtasks", "/Create", "/F",
        "/TN", TASK_NAME,
        "/TR", f'"{python}" "{script}"',
        "/SC", "DAILY",
        "/ST", "03:00",
        "/RL", "HIGHEST",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [SCHED] Task '{TASK_NAME}' registered — runs daily at 03:00")
        print(f"          View in: Task Scheduler > Task Scheduler Library")
    else:
        print(f"  [FAIL]  Scheduling failed: {result.stderr.strip()}")
        print(f"          Try running this script as Administrator")


def _unschedule_task() -> None:
    result = subprocess.run(
        ["schtasks", "/Delete", "/F", "/TN", TASK_NAME],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  [SCHED] Task '{TASK_NAME}' removed from Task Scheduler")
    else:
        print(f"  [WARN]  Task not found or already removed: {result.stderr.strip()}")


# ── Training ───────────────────────────────────────────────────────────────────

async def _train() -> bool:
    """
    Instantiate the CNN agent directly (no backend running) and run
    train_on_history().  Reads from the live SQLite DB + parquet history.
    """
    # Add backend to path so all imports resolve
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    # Load .env so config picks up DATABASE_URL, credentials, etc.
    env_file = ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # Override DB path to the known live database
    os.environ["DATABASE_URL"] = str(DB_PATH)

    import database
    import importlib, config as _cfg
    importlib.reload(_cfg)          # re-read config with updated env
    database.DB_PATH = str(DB_PATH)

    await database.init_db()

    from agents.cnn_agent import CoinbaseCNNAgent
    agent = CoinbaseCNNAgent()

    products = await database.get_products()
    candle_counts = []
    for p in products[:5]:
        pid = p["product_id"]
        from services.history_backfill import load_history
        hist = load_history(pid)
        live = await database.get_candles(pid, limit=200)
        candle_counts.append((pid, len(hist or []), len(live)))
    print("  [DATA]  Sample candle availability (parquet + live):")
    for pid, ph, lv in candle_counts:
        print(f"            {pid:16s}  parquet={ph:5d}  live={lv:3d}")

    print(f"\n  [TRAIN] Starting CNN training (epochs=50)...")
    t0 = time.time()
    result = await agent.train_on_history(epochs=50)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"  [FAIL]  Training failed: {result['error']}")
        return False

    print(f"  [TRAIN] Done in {elapsed:.0f}s")
    print(f"          Samples    : {result['train_samples']} train / {result['val_samples']} val")
    print(f"          Loss       : {result['initial_loss']:.4f} → {result['final_train_loss']:.4f} (train)")
    print(f"          Val loss   : {result['final_val_loss']:.4f}")
    print(f"          Fit status : {result['fit_status']}")
    if result['fit_status'] != 'OK':
        print(f"          Advice     : {result['fit_advice']}")
    print(f"          Model saved: backend/cnn_model.pt")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("Coinbase AI Trader — CNN Retrain")

    if SCHEDULE:
        _schedule_task()
        return

    if UNSCHEDULE:
        _unschedule_task()
        return

    # 1. Stop the running backend
    print("\n  Step 1/4  Stop backend")
    _stop_backend()

    # 2. Train
    print("\n  Step 2/4  Train CNN model")
    success = asyncio.run(_train())
    if not success:
        print("\n  [ABORT] Training failed — backend not restarted.")
        print("          Check that the database has enough candle history.")
        sys.exit(1)

    # 3. Retrain LightGBM on historical trades (re-calibrate to new CNN weights)
    print("\n  Step 3/4  Retrain LightGBM filter")
    asyncio.run(_retrain_lgbm())

    # 4. Restart
    if NO_RESTART:
        print("\n  Step 4/4  Skipped (--no-restart)")
        print("\n  CNN + LGBM retrained. Start the backend manually when ready.")
    else:
        print("\n  Step 4/4  Restart backend")
        _start_backend()
        print("\n  All done. CNN running with fresh weights, LGBM recalibrated to new CNN outputs.")


if __name__ == "__main__":
    main()
