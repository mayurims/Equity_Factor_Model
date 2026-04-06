"""
factors.py — Compute cross-sectional factor signals for the equity universe.

Factors implemented (Fama-French inspired):
  1. MOMENTUM  — 12-month return skipping the most recent month (12-1)
  2. VALUE     — Price-to-Book ratio (inverted: low P/B = cheap = high value score)
  3. SIZE      — Log market capitalization (inverted: small = high score)
  4. LOW VOL   — Realized volatility over trailing 60 trading days (inverted)

Each factor is:
  - Cross-sectionally z-scored each month (subtract mean, divide by std)
  - Winsorized at ±3σ to limit outlier influence
  - Adaptively signed: trailing IC determines whether the factor is currently
    working in its textbook direction or inverted
  - IC-weighted into a composite: factors with stronger recent predictive
    power get more weight

Output:
  A DataFrame of composite z-scores and individual factor z-scores per month,
  plus quintile assignments for portfolio construction.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore, spearmanr

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
MOMENTUM_LOOKBACK = 252       # ~12 months of trading days
MOMENTUM_SKIP = 21            # skip most recent month (avoid short-term reversal)
VOLATILITY_WINDOW = 60        # ~3 months of trading days
WINSORIZE_SIGMA = 3.0         # cap z-scores at ±3
QUINTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# Adaptive weighting parameters
IC_LOOKBACK_MONTHS = 12       # trailing months to compute IC
MIN_IC_MONTHS = 6             # minimum months before we can compute IC
IC_SHRINKAGE = 0.5            # blend IC weights with equal weights (0=pure equal, 1=pure IC)


def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum = (Price_t-21 / Price_t-252) - 1
    
    This is the classic 12-1 momentum signal: the return over the past 12 months,
    excluding the most recent month. We skip the recent month because very 
    short-term returns tend to reverse (the "short-term reversal" anomaly).
    
    High momentum stocks have gone up a lot → we expect them to continue going up.
    """
    lagged = prices.shift(MOMENTUM_SKIP)
    far_lagged = prices.shift(MOMENTUM_LOOKBACK)
    momentum = (lagged / far_lagged) - 1
    return momentum


def compute_value(fundamentals: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Value = -1 × Price-to-Book ratio
    
    Low P/B means the stock is "cheap" relative to its book value.
    We negate it so that high value score = cheap stock = what we want to buy.
    """
    pb = fundamentals.reindex(tickers)["priceToBook"]
    value = -pb
    return value


def compute_size(fundamentals: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Size = -1 × log(Market Cap)
    
    Small-cap stocks have historically outperformed large-caps.
    Negated so small stocks get a high score.
    """
    mcap = fundamentals.reindex(tickers)["marketCap"]
    size = -np.log(mcap.replace(0, np.nan))
    return size


def compute_low_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Low Volatility = -1 × realized volatility (std of daily returns, 60-day window)
    
    Low-vol stocks tend to deliver higher risk-adjusted returns.
    Annualized by multiplying by sqrt(252). Negated so low vol = high score.
    """
    daily_returns = prices.pct_change()
    rolling_vol = daily_returns.rolling(window=VOLATILITY_WINDOW, min_periods=40).std() * np.sqrt(252)
    low_vol = -rolling_vol
    return low_vol


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score across all stocks at a single point in time.
    z = (x - mean) / std
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - mean) / std


def winsorize(series: pd.Series, sigma: float = WINSORIZE_SIGMA) -> pd.Series:
    """Cap extreme values at ±sigma standard deviations."""
    return series.clip(lower=-sigma, upper=sigma)


def compute_trailing_ic(ic_history: list[dict], factor_name: str,
                        lookback: int = IC_LOOKBACK_MONTHS) -> float:
    """
    Compute the trailing Information Coefficient (IC) for a factor.
    
    IC = Spearman rank correlation between the factor signal at time t
         and the realized cross-sectional return at time t+1.
    
    A positive IC means the factor is working as intended (high signal → high return).
    A negative IC means the factor is inverted (high signal → low return).
    
    We average the monthly IC over the trailing `lookback` months.
    
    This is computed using ONLY past data — no look-ahead bias.
    """
    recent = ic_history[-lookback:]
    
    if len(recent) < MIN_IC_MONTHS:
        return 0.0  # not enough history, default to no opinion
    
    ics = []
    for record in recent:
        ic_val = record.get(factor_name)
        if ic_val is not None and not np.isnan(ic_val):
            ics.append(ic_val)
    
    if len(ics) < MIN_IC_MONTHS:
        return 0.0
    
    return np.mean(ics)


def compute_adaptive_weights(ic_dict: dict) -> dict:
    """
    Compute factor weights based on trailing IC magnitudes.
    
    Process:
      1. Take absolute value of each factor's IC (magnitude of predictive power)
      2. Normalize so weights sum to 1
      3. Shrink toward equal weights to avoid overfitting to recent IC
      4. Use the SIGN of the IC to flip factors working in reverse
    
    Returns dict of {factor_name: signed_weight}.
    """
    factor_names = list(ic_dict.keys())
    n_factors = len(factor_names)
    equal_weight = 1.0 / n_factors
    
    # Absolute ICs for magnitude weighting
    abs_ics = {k: abs(v) for k, v in ic_dict.items()}
    total_abs_ic = sum(abs_ics.values())
    
    weights = {}
    for f in factor_names:
        if total_abs_ic > 0:
            ic_weight = abs_ics[f] / total_abs_ic
        else:
            ic_weight = equal_weight
        
        # Shrink toward equal weights
        blended_weight = IC_SHRINKAGE * ic_weight + (1 - IC_SHRINKAGE) * equal_weight
        
        # Apply sign: if IC is negative, flip the factor direction
        sign = np.sign(ic_dict[f]) if ic_dict[f] != 0 else 1.0
        weights[f] = sign * blended_weight
    
    return weights


def compute_all_factors(prices: pd.DataFrame, fundamentals: pd.DataFrame,
                        rebalance_dates: pd.DatetimeIndex,
                        monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all four factor signals at each rebalance date with adaptive weighting.
    
    ADAPTIVE PROCESS for each month t:
      1. Compute raw signal for each factor using data up to t
      2. Cross-sectionally z-score and winsorize
      3. Compute trailing IC using months [t-12, t-1] — how well did each
         factor predict REALIZED returns in the recent past?
      4. If IC is negative, FLIP the factor (multiply z-score by -1)
      5. Weight factors by |IC| magnitude (factors working better get more weight)
      6. Composite = weighted sum of (signed) z-scores
      7. Quintile assignment based on composite
    
    This is fully walk-forward: IC at month t uses only data from months before t.
    """
    tickers = prices.columns.tolist()
    
    # Pre-compute time-series factors on daily data
    print("[factors] Computing momentum signals...")
    momentum_ts = compute_momentum(prices)
    
    print("[factors] Computing low-volatility signals...")
    lowvol_ts = compute_low_volatility(prices)
    
    # Static factors
    print("[factors] Computing value and size signals...")
    value_raw = compute_value(fundamentals, tickers)
    size_raw = compute_size(fundamentals, tickers)
    
    all_records = []
    ic_history = []      # list of dicts: {momentum: IC, value: IC, ...} per month
    prev_signals = None   # signals from previous month (to compute IC with this month's returns)
    factor_names = ["momentum", "value", "size", "lowvol"]
    
    print("[factors] Computing adaptive composite with trailing IC weighting...")
    
    for i, date in enumerate(rebalance_dates):
        # Get the closest available date in our price data
        available_dates = prices.index[prices.index <= date]
        if len(available_dates) == 0:
            continue
        closest_date = available_dates[-1]
        
        # --- Raw signals at this date ---
        mom_signal = momentum_ts.loc[closest_date] if closest_date in momentum_ts.index else pd.Series(dtype=float)
        vol_signal = lowvol_ts.loc[closest_date] if closest_date in lowvol_ts.index else pd.Series(dtype=float)
        val_signal = value_raw
        siz_signal = size_raw
        
        # Combine into a single DataFrame for this month
        month_df = pd.DataFrame({
            "momentum": mom_signal,
            "value": val_signal,
            "size": siz_signal,
            "lowvol": vol_signal,
        }, index=tickers)
        
        month_df = month_df.dropna()
        
        if len(month_df) < 20:
            continue
        
        # --- Cross-sectional z-score + winsorize each factor ---
        for col in factor_names:
            month_df[f"{col}_z"] = winsorize(cross_sectional_zscore(month_df[col]))
        
        # --- Compute IC for PREVIOUS month's signals vs THIS month's returns ---
        # This is the walk-forward IC: did last month's signal predict this month's return?
        if prev_signals is not None and date in monthly_returns.index:
            realized = monthly_returns.loc[date]
            month_ic = {}
            for f in factor_names:
                sig = prev_signals.get(f)
                if sig is None:
                    month_ic[f] = np.nan
                    continue
                aligned = pd.DataFrame({"signal": sig, "ret": realized}).dropna()
                if len(aligned) < 20:
                    month_ic[f] = np.nan
                    continue
                corr, _ = spearmanr(aligned["signal"], aligned["ret"])
                month_ic[f] = corr if not np.isnan(corr) else 0.0
            ic_history.append(month_ic)
        
        # --- Compute trailing IC for each factor (using only past data) ---
        ic_dict = {}
        for f in factor_names:
            ic_dict[f] = compute_trailing_ic(ic_history, f)
        
        # --- Compute adaptive weights ---
        weights = compute_adaptive_weights(ic_dict)
        
        # --- Composite score: IC-weighted, adaptively signed ---
        composite = pd.Series(0.0, index=month_df.index)
        for f in factor_names:
            composite += weights[f] * month_df[f"{f}_z"]
        
        month_df["composite_z"] = composite
        
        # Store weights for diagnostics
        for f in factor_names:
            month_df[f"{f}_weight"] = weights[f]
        
        # --- Quintile assignment based on composite score ---
        try:
            month_df["quintile"] = pd.qcut(
                month_df["composite_z"], q=5, labels=QUINTILE_LABELS
            )
        except ValueError:
            # If too many tied values, use rank-based assignment
            ranks = month_df["composite_z"].rank(method="first")
            month_df["quintile"] = pd.qcut(
                ranks, q=5, labels=QUINTILE_LABELS
            )
        
        month_df["date"] = date
        month_df["ticker"] = month_df.index
        all_records.append(month_df)
        
        # --- Store current signals for next month's IC computation ---
        prev_signals = {}
        for f in factor_names:
            prev_signals[f] = month_df[f"{f}_z"].copy()
        
        # Log adaptive weights periodically
        if (i + 1) % 12 == 0 or i == len(rebalance_dates) - 1:
            w_str = ", ".join([f"{f}={weights[f]:+.3f}" for f in factor_names])
            ic_str = ", ".join([f"{f}={ic_dict[f]:+.3f}" for f in factor_names])
            print(f"[factors]   Month {i+1}: IC=[{ic_str}]")
            print(f"[factors]            Wt=[{w_str}]")
    
    if not all_records:
        raise ValueError("No valid factor data computed. Check your input data.")
    
    result = pd.concat(all_records, ignore_index=True)
    result = result.set_index(["date", "ticker"])
    
    print(f"[factors] Computed adaptive factors for {len(rebalance_dates)} months, "
          f"avg {result.groupby('date').size().mean():.0f} stocks/month")
    
    return result


def get_signal_matrix(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the factor data into a (date × ticker) signal matrix.
    Useful for quick analysis and visualization.
    """
    return factor_data["composite_z"].unstack("ticker")


def run(prices: pd.DataFrame, fundamentals: pd.DataFrame,
        monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Compute all factors at each month-end rebalance date.
    """
    rebalance_dates = monthly_returns.index
    factor_data = compute_all_factors(prices, fundamentals, rebalance_dates,
                                      monthly_returns)
    
    # Save to disk
    factor_data.to_csv("data/factor_signals.csv")
    print("[factors] ✓ Factor signals saved to data/factor_signals.csv")
    
    return factor_data


if __name__ == "__main__":
    import data
    prices, volumes, monthly_returns, fundamentals = data.run()
    run(prices, fundamentals, monthly_returns)