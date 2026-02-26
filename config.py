"""
Central configuration for the backtesting engine.
All tunable parameters live here — no magic numbers in library code.
"""

# ──────────────────────────────────────────────
# Asset Universe
# ──────────────────────────────────────────────
ASSETS = [
    "SPY",      # S&P 500
    "QQQ",      # Nasdaq 100
    "IWM",      # Russell 2000
    "EFA",      # International Developed
    "TLT",      # 20+ Year Treasury
    "IEF",      # 7-10 Year Treasury
    "GLD",      # Gold
    "SLV",      # Silver
    "USO",      # Oil
    "BTC-USD",  # Bitcoin
    "DBA",      # Agriculture
    "VNQ",      # Real Estate (REITs)
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
REBALANCE_FREQ = "monthly"   # 'daily' or 'monthly'
TRANSACTION_COST_BPS = 10    # 10 basis points round-trip
SLIPPAGE_BPS = 5             # 5 basis points slippage

# ──────────────────────────────────────────────
# Optimizer
# ──────────────────────────────────────────────
RISK_FREE_RATE = 0.04        # Annualized
KELLY_FRACTION = 0.5         # Half-Kelly (0.5), Full Kelly (1.0)
MAX_POSITION_WEIGHT = 0.40   # No single asset > 40%
MAX_LEVERAGE = 1.5           # Total gross exposure cap
LOOKBACK_WINDOW = 252        # Trading days for return estimation
COV_METHOD = "ledoit_wolf"   # 'ledoit_wolf', 'sample', 'ewma'
EWMA_LAMBDA = 0.94           # For EWMA covariance

# ──────────────────────────────────────────────
# Monte Carlo
# ──────────────────────────────────────────────
MC_NUM_PATHS = 50_000
MC_HORIZON_DAYS = 252        # 1 year forward simulation
MC_SEED = 42

# ──────────────────────────────────────────────
# Risk
# ──────────────────────────────────────────────
VAR_CONFIDENCE = 0.95
ROLLING_SHARPE_WINDOW = 252
REGIME_VOL_THRESHOLDS = {
    "bull": 0.15,            # < 15% annualized vol
    "sideways": 0.25,        # 15-25%
    # "bear" = everything above 25%
}

# ──────────────────────────────────────────────
# Black-Litterman
# ──────────────────────────────────────────────
# Approximate relative market-cap weights for the 12-asset universe.
# Used as the equilibrium prior in the BL model.  Weights sum to 1.0.
# Sources: rough relative market-cap proxies (not live data).
MARKET_CAP_WEIGHTS = {
    "SPY":     0.40,   # S&P 500 — US large-cap equity dominates
    "QQQ":     0.15,   # Nasdaq 100
    "IWM":     0.05,   # Russell 2000 small-cap
    "EFA":     0.10,   # International developed markets
    "TLT":     0.08,   # Long-duration US Treasuries
    "IEF":     0.05,   # Intermediate US Treasuries
    "GLD":     0.05,   # Gold
    "SLV":     0.01,   # Silver
    "USO":     0.02,   # Oil
    "BTC-USD": 0.03,   # Bitcoin
    "DBA":     0.01,   # Agriculture commodities
    "VNQ":     0.05,   # US REITs
}

BL_RISK_AVERSION = 2.5   # delta — standard market equilibrium assumption
BL_TAU = 0.05             # prior uncertainty scaling (typical range: 0.01–0.10)
BL_VIEW_LOOKBACK = 252    # days for momentum view (1 year)

# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────
CHART_DIR = "output/charts"
REPORT_PATH = "output/report.txt"
CHART_DPI = 300
