"""
Central configuration for the backtesting engine.
All tunable parameters live here — no magic numbers in library code.
"""

# ──────────────────────────────────────────────
# Asset Universe
# ──────────────────────────────────────────────
ASSETS = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000
    "EFA",  # International Developed
    "TLT",  # 20+ Year Treasury
    "IEF",  # 7-10 Year Treasury
    "GLD",  # Gold
    "SLV",  # Silver
    "USO",  # Oil
    "BTC-USD",  # Bitcoin
    "DBA",  # Agriculture
    "VNQ",  # Real Estate (REITs)
]

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
CACHE_DIR = "data/cache"

# ──────────────────────────────────────────────
# Backtest Engine
# ──────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000  # $1M starting portfolio
REBALANCE_FREQ = "monthly"  # 'daily' or 'monthly'
TRANSACTION_COST_BPS = 10  # 10 basis points round-trip
SLIPPAGE_BPS = 5  # 5 basis points slippage

# ──────────────────────────────────────────────
# Almgren-Chriss Transaction Cost Model
# ──────────────────────────────────────────────
COST_MODEL = "almgren_chriss"       # "proportional" or "almgren_chriss"
SPREAD_BPS_DEFAULT = 5.0           # Half-spread in basis points (default)
IMPACT_ETA = 0.1                   # Temporary impact coefficient
IMPACT_GAMMA = 0.1                 # Permanent impact coefficient
SLIPPAGE_FACTOR = 0.5              # Volatility-proportional slippage multiplier
ADV_WINDOW = 20                    # Rolling average daily volume window
MAX_TURNOVER = 0.30                # Max one-way turnover per rebalance (None = unconstrained)

# ──────────────────────────────────────────────
# Optimizer
# ──────────────────────────────────────────────
RISK_FREE_RATE = 0.04  # Annualized
KELLY_FRACTION = 0.5  # Half-Kelly (0.5), Full Kelly (1.0)
MAX_POSITION_WEIGHT = 0.40  # No single asset > 40%
MAX_LEVERAGE = 1.5  # Total gross exposure cap
LOOKBACK_WINDOW = 252  # Trading days for return estimation
COV_METHOD = "ledoit_wolf"  # 'ledoit_wolf', 'ledoit_wolf_nonlinear', 'sample', 'ewma', 'rmt_denoised', 'dcc_garch'
EWMA_LAMBDA = 0.94  # For EWMA covariance
RMT_DENOISE_METHOD = "constant_residual"  # 'constant_residual' or 'targeted_shrinkage'
DCC_GARCH_WINDOW = 500  # Lookback window for DCC-GARCH fitting

# ──────────────────────────────────────────────
# Volatility Targeting
# ──────────────────────────────────────────────
TARGET_VOL = 0.10  # Annualized target volatility (10%)
VOL_TARGET_MAX_LEVERAGE = 2.0  # Maximum leverage from vol targeting

# ──────────────────────────────────────────────
# Covariance Comparison
# ──────────────────────────────────────────────
ENABLE_COVARIANCE_COMPARISON = False
COV_COMPARISON_METHODS = [
    "sample", "ledoit_wolf", "ledoit_wolf_nonlinear",
    "ewma", "rmt_denoised", "dcc_garch",
]

# ──────────────────────────────────────────────
# Strategy Comparison
# ──────────────────────────────────────────────
STRATEGIES_TO_RUN = [
    "equal_weight",
    "half_kelly",
    "full_kelly",
    "bl_kelly",
    "hrp",
    "risk_parity",
    "vol_targeted_bl",
]

# ──────────────────────────────────────────────
# Monte Carlo
# ──────────────────────────────────────────────
MC_NUM_PATHS = 50_000
MC_HORIZON_DAYS = 252  # 1 year forward simulation
MC_SEED = 42

# ──────────────────────────────────────────────
# Risk
# ──────────────────────────────────────────────
VAR_CONFIDENCE = 0.95
ROLLING_SHARPE_WINDOW = 252
REGIME_VOL_THRESHOLDS = {
    "bull": 0.15,  # < 15% annualized vol
    "sideways": 0.25,  # 15-25%
    # "bear" = everything above 25%
}

# ──────────────────────────────────────────────
# HMM Regime Detection
# ──────────────────────────────────────────────
REGIME_DETECTOR = "hmm"           # "hmm" or "threshold" (legacy)
REGIME_BENCHMARK = "SPY"          # Benchmark ticker for regime classification
HMM_N_STATES = None               # None = auto-select 2 vs 3 by BIC; or int
HMM_FEATURE_WINDOW = 20           # Rolling vol window for feature engineering
HMM_RANDOM_STATE = 42             # Seed for HMM fitting

# ──────────────────────────────────────────────
# Black-Litterman
# ──────────────────────────────────────────────
# Approximate relative market-cap weights for the 12-asset universe.
# Used as the equilibrium prior in the BL model.  Weights sum to 1.0.
# Sources: rough relative market-cap proxies (not live data).
MARKET_CAP_WEIGHTS = {
    "SPY": 0.40,  # S&P 500 — US large-cap equity dominates
    "QQQ": 0.15,  # Nasdaq 100
    "IWM": 0.05,  # Russell 2000 small-cap
    "EFA": 0.10,  # International developed markets
    "TLT": 0.08,  # Long-duration US Treasuries
    "IEF": 0.05,  # Intermediate US Treasuries
    "GLD": 0.05,  # Gold
    "SLV": 0.01,  # Silver
    "USO": 0.02,  # Oil
    "BTC-USD": 0.03,  # Bitcoin
    "DBA": 0.01,  # Agriculture commodities
    "VNQ": 0.05,  # US REITs
}

BL_RISK_AVERSION = 2.5  # delta — standard market equilibrium assumption
BL_TAU = 0.05  # prior uncertainty scaling (typical range: 0.01–0.10)
BL_VIEW_LOOKBACK = 252  # days for momentum view (1 year)

# ──────────────────────────────────────────────
# Statistical Validation
# ──────────────────────────────────────────────
N_STRATEGY_TRIALS = 1  # Number of strategy variations tested (for DSR)
ENABLE_CPCV = True  # Run combinatorial purged cross-validation
CPCV_N_GROUPS = 10  # Chronological blocks for CPCV
CPCV_N_TEST_GROUPS = 2  # Test groups per CPCV split
CPCV_PURGE_WINDOW = 5  # Trading days purged at train/test boundaries
CPCV_EMBARGO_PCT = 0.01  # Fraction of data embargoed after test groups

# ──────────────────────────────────────────────
# Factor Attribution
# ──────────────────────────────────────────────
ENABLE_FACTOR_ATTRIBUTION = True
FACTOR_CACHE_DIR = "data/cache/factors"
FACTOR_ROLLING_WINDOW = 252  # Trading days for rolling attribution

# ──────────────────────────────────────────────
# Parameter Stability
# ──────────────────────────────────────────────
ENABLE_PARAMETER_STABILITY = False
PARAM_STABILITY_WINDOW_YEARS = 3
PARAM_STABILITY_STEP_MONTHS = 3

# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────
CHART_DIR = "output/charts"
REPORT_PATH = "output/report.txt"
CHART_DPI = 300
