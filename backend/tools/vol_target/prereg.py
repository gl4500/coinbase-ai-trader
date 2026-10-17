"""Frozen preregistration for the BTC/ETH volatility-target screen (2026-10-03).

Every value was fixed BEFORE any rule ran on any data, in the Claude/Codex debate archived at
C:\\Users\\gl450\\analysis_archive\\vol_target_prereg_2026-10-03 (c4c6827b + amendments 52c7e2ee).
Shared values are imported from the frozen slow_trend preregistration so the two screens cannot
drift apart. Changing a value is a new preregistration, never a revision of a verdict.
"""

from tools.slow_trend import prereg as S

PRODUCTS = S.PRODUCTS
INITIAL_USD = S.INITIAL_USD
SLEEVE_USD = S.SLEEVE_USD
SCENARIOS = S.SCENARIOS
GATING_SENSITIVITIES = S.GATING_SENSITIVITIES
G2_DRAWDOWN_RATIO = S.G2_DRAWDOWN_RATIO
DCA_TRANCHES = S.DCA_TRANCHES
MAX_MISSING_DAYS = S.MAX_MISSING_DAYS
BOOT_BLOCK_WEEKS, BOOT_SENS_WEEKS = S.BOOT_BLOCK_WEEKS, S.BOOT_SENS_WEEKS
BOOT_RESAMPLES, BOOT_SEED = S.BOOT_RESAMPLES, S.BOOT_SEED

DEV_START = "2016-08-30"  # the SMA screen's evaluated dev start, frozen for comparability
DEV_END, BLOCK_START, BLOCK_END = S.DEV_END, S.BLOCK_START, S.BLOCK_END

VOL_RETURNS = 20  # log returns -> 21 consecutive valid daily closes
ANNUALISATION = 365
SIGMA_TARGET = 0.50  # per asset sleeve; discretionary risk budget, not an optimum
DEADBAND = 0.10  # absolute weight; trade only when |target - held| > DEADBAND (strict)
DECISION_WEEKDAY = 6  # Sunday-labelled completed UTC daily bar
FIXED_DIAG_WEIGHT = 0.5  # NON-GATING diagnostic only
MIN_BLOCK_VALID_DECISIONS = 4  # per sleeve, P0, block; administrative coverage, not power

SOURCE_SNAPSHOT_LOCK = "sha256:5215b520ba2d25b78897cbe0b5c3024547f10b353454e3cc3a89f09f0b71320c"
FREEZE_PATHS = (
    "backend/tools/vol_target",
    "backend/tools/slow_trend",
    "backend/clients/coinbase_client.py",
)
