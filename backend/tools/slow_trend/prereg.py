"""Frozen preregistration for the BTC/ETH SMA100 falsification screen (2026-10-03).

Every value here was fixed BEFORE any rule ran on any data. See
docs/superpowers/plans/2026-10-03-slow-trend-screen.md#preregistration. Changing a value is a
new preregistration (new commit), never a revision of an existing verdict.
"""

from dataclasses import dataclass

PRODUCTS = ("BTC-USD", "ETH-USD")
SMA_LEN = 100
INITIAL_USD = 1000.0
SLEEVE_USD = INITIAL_USD / len(PRODUCTS)

TAKER_FEE = 0.009  # verified Intro tier, transaction_summary 2026-10-03
MAKER_FEE = 0.005

FETCH_FROM = "2015-01-01"  # request bound only; actual first day is recorded
DEV_END = "2025-04-13"
BLOCK_START = "2025-04-14"
BLOCK_END = "2026-10-02"  # last complete UTC day at preregistration
MAX_MISSING_DAYS = 3

DCA_TRANCHES = 52
G2_DRAWDOWN_RATIO = 2 / 3  # arbitrary provisional utility gate, not statistical materiality
MIN_BLOCK_ROUND_TRIPS = 4  # arbitrary administrative threshold, not an evidence threshold

BOOT_BLOCK_WEEKS = 8
BOOT_SENS_WEEKS = (4, 13)
BOOT_RESAMPLES = 10_000
BOOT_SEED = 20261003

FREEZE_PATHS = ("backend/tools/slow_trend", "backend/clients/coinbase_client.py")


@dataclass(frozen=True)
class Scenario:
    name: str
    entry_fee: float
    exit_fee: float
    slip: float  # adverse price adjustment per leg, fraction
    delay: int  # extra days between decision close and execution open


SCENARIOS = (
    Scenario("P0", TAKER_FEE, TAKER_FEE, 0.0, 0),
    Scenario("S10", TAKER_FEE, TAKER_FEE, 0.0010, 0),
    Scenario("S25", TAKER_FEE, TAKER_FEE, 0.0025, 0),
    Scenario("D1", TAKER_FEE, TAKER_FEE, 0.0, 1),
    Scenario("SM", MAKER_FEE, TAKER_FEE, 0.0, 0),  # optimistic; never consulted by gates
)
GATING_SENSITIVITIES = ("S10", "S25", "D1")
