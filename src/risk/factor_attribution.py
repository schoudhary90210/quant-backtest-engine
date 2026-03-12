"""
Factor attribution using Fama-French 5 factors + Momentum.

Decomposes strategy returns into exposure to known risk factors to determine
whether alpha is real or just factor loading.
"""

from __future__ import annotations

import io
import logging
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config

logger = logging.getLogger(__name__)

_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


class FactorAttribution:
    """Fetch Fama-French factor data, run OLS regressions, and compute rolling attribution."""

    def __init__(self, factors_df: pd.DataFrame | None = None) -> None:
        self._cache_dir = Path(config.FACTOR_CACHE_DIR)
        self._factors: pd.DataFrame | None = factors_df

    # ── Public API ────────────────────────────────────────────────

    def fetch_factors(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch FF5 + Momentum daily factors. Returns date-sliced copy."""
        if self._factors is not None:
            return self._factors.loc[start_date:end_date]

        cache_path = self._cache_dir / "ff5_momentum_daily.parquet"

        try:
            ff5 = self._download_and_parse(_FF5_URL)
            mom = self._download_and_parse(_MOM_URL)

            # Rename momentum column (strip whitespace from column names)
            mom.columns = [c.strip() for c in mom.columns]
            if "Mom" not in mom.columns:
                # Try common alternatives
                for col in mom.columns:
                    if "mom" in col.lower():
                        mom = mom.rename(columns={col: "Mom"})
                        break

            merged = ff5.join(mom[["Mom"]], how="inner")
            self._factors = merged

            # Cache full history
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(cache_path)
            logger.info("Factor data cached → %s (%d rows)", cache_path, len(merged))

        except Exception as exc:
            if cache_path.exists():
                logger.warning(
                    "Factor download failed (%s) — loading from cache.", exc
                )
                self._factors = pd.read_parquet(cache_path)
            else:
                raise RuntimeError(
                    f"Failed to fetch factor data and no cache available: {exc}"
                ) from exc

        return self._factors.loc[start_date:end_date]

    def attribute(self, strategy_returns: pd.Series) -> dict:
        """Full-sample OLS factor attribution with HAC standard errors."""
        if self._factors is None:
            raise ValueError(
                "No factor data loaded; call fetch_factors() or provide factors_df"
            )

        # Align on common dates
        common = strategy_returns.index.intersection(self._factors.index)
        n = len(common)
        if n < 60:
            raise ValueError(
                f"Only {n} overlapping observations between strategy returns "
                f"and factor data (need >= 60)"
            )

        ret = strategy_returns.loc[common]
        factors = self._factors.loc[common]

        excess = ret - factors["RF"]
        X = sm.add_constant(factors[FACTOR_COLS])
        model = sm.OLS(excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        alpha_daily = float(model.params["const"])
        unexplained = alpha_daily + model.resid

        return {
            "alpha_annual": alpha_daily * 252,
            "alpha_tstat": float(model.tvalues["const"]),
            "alpha_pvalue": float(model.pvalues["const"]),
            "betas": {f: float(model.params[f]) for f in FACTOR_COLS},
            "beta_tstats": {f: float(model.tvalues[f]) for f in FACTOR_COLS},
            "beta_pvalues": {f: float(model.pvalues[f]) for f in FACTOR_COLS},
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "factor_attributed_returns": {
                f: float(model.params[f]) * float(factors[f].mean()) * 252
                for f in FACTOR_COLS
            },
            "residual_alpha": alpha_daily * 252,
            "alpha_sharpe": (
                float(unexplained.mean() / unexplained.std() * np.sqrt(252))
                if unexplained.std() > 0
                else 0.0
            ),
            "information_ratio": (
                float(alpha_daily / model.resid.std() * np.sqrt(252))
                if model.resid.std() > 0
                else 0.0
            ),
            "n_observations": int(model.nobs),
        }

    def rolling_attribution(
        self, strategy_returns: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """Rolling OLS factor attribution (no HAC for speed)."""
        if self._factors is None:
            raise ValueError(
                "No factor data loaded; call fetch_factors() or provide factors_df"
            )

        common = strategy_returns.index.intersection(self._factors.index)
        n = len(common)
        if n < window + 10:
            raise ValueError(
                f"Only {n} overlapping observations (need >= {window + 10} "
                f"for window={window})"
            )

        ret = strategy_returns.loc[common]
        factors = self._factors.loc[common]
        excess = ret - factors["RF"]

        records = []
        for end in range(window, len(common) + 1):
            start = end - window
            y = excess.iloc[start:end]
            X = sm.add_constant(factors[FACTOR_COLS].iloc[start:end])
            model = sm.OLS(y, X).fit()

            row = {
                "alpha_annual": float(model.params["const"]) * 252,
                "alpha_se_annual": float(model.bse["const"]) * 252,
                "r_squared": float(model.rsquared),
            }
            for f in FACTOR_COLS:
                row[f] = float(model.params[f])
            records.append((common[end - 1], row))

        dates, rows = zip(*records)
        return pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    def _download_and_parse(url: str) -> pd.DataFrame:
        """Download a Ken French zip, extract the CSV, parse into DataFrame."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        resp = urllib.request.urlopen(url, timeout=30, context=ctx)  # noqa: S310
        data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = None
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    csv_name = name
                    break
            if csv_name is None:
                raise RuntimeError(f"No CSV found in {url}")

            raw_lines = zf.read(csv_name).decode("utf-8", errors="replace").splitlines()

        # Find first data line (starts with a digit)
        skip = 0
        for i, line in enumerate(raw_lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                skip = i
                break

        df = pd.read_csv(
            io.StringIO("\n".join(raw_lines[skip - 1:])),  # include header row
            index_col=0,
        )

        # If header wasn't captured, re-read with the proper skip count
        if df.empty or df.columns[0].strip().isdigit():
            # Fallback: read from the line before data, which should be the header
            header_line = skip - 1
            if header_line < 0:
                header_line = 0
            df = pd.read_csv(
                io.StringIO("\n".join(raw_lines)),
                skiprows=header_line,
                index_col=0,
            )

        # Normalize index: convert to string and strip whitespace
        df.index = df.index.astype(str).str.strip()

        # Keep only daily data rows (YYYYMMDD = 8 chars)
        mask = df.index.str.len() == 8
        df = df.loc[mask]

        # Parse dates
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = None

        # Strip column name whitespace
        df.columns = [c.strip() for c in df.columns]

        # Convert from percentage to decimal
        df = df.astype(float) / 100.0

        return df
