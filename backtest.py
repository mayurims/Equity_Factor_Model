"""
backtest.py — Walk-forward backtest engine with full performance analytics.

This module:
  1. Constructs quintile portfolios (Q1=short, Q5=long) each month
  2. Runs a walk-forward backtest: expanding training window, never looks forward
  3. Computes the long-short (Q5-Q1) portfolio return each month
  4. Calculates performance metrics: Sharpe, max drawdown, Calmar, IR vs SPY
  5. Runs transaction cost sensitivity analysis (5-25 bps sweep)
  6. Generates all required plots

WALK-FORWARD METHODOLOGY:
  - At each month t, we use data from [start, t] to compute factor signals
  - Signals are used to rank stocks and form quintile portfolios
  - We then observe the NEXT month's (t+1) returns — never looked at before
  - This simulates a real trading process: decide today, observe tomorrow
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "results"
QUINTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
TC_BPS_RANGE = range(5, 30, 5)  # 5, 10, 15, 20, 25 bps
RISK_FREE_RATE = 0.0            # simplified; use 0 for excess return calc


def construct_portfolios(factor_data: pd.DataFrame,
                         monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build equal-weighted quintile portfolio returns each month.
    
    Process:
      1. At each rebalance date, group stocks by their quintile assignment
      2. Equal-weight all stocks within each quintile
      3. The return for month t is the average return of stocks in that quintile
         during month t+1 (the NEXT month — this is the walk-forward part)
    
    Returns:
        DataFrame with columns Q1..Q5 and 'long_short' (Q5 - Q1)
    """
    dates = factor_data.index.get_level_values("date").unique().sort_values()
    
    portfolio_returns = []
    turnover_records = []
    prev_holdings = {q: set() for q in QUINTILE_LABELS}
    
    for i, date in enumerate(dates[:-1]):  # skip last month (no forward return)
        next_date = dates[i + 1]
        
        # Get quintile assignments at this date
        month_data = factor_data.loc[date]
        
        # Get next month's returns
        if next_date not in monthly_returns.index:
            continue
        next_returns = monthly_returns.loc[next_date]
        
        row = {"date": next_date}
        month_turnover = {}
        
        for q in QUINTILE_LABELS:
            stocks_in_q = month_data[month_data["quintile"] == q].index.tolist()
            
            # Filter to stocks that have return data
            valid_stocks = [s for s in stocks_in_q if s in next_returns.index
                          and not np.isnan(next_returns[s])]
            
            if len(valid_stocks) == 0:
                row[q] = 0.0
            else:
                # Equal-weighted portfolio return
                row[q] = next_returns[valid_stocks].mean()
            
            # Track turnover
            current_set = set(valid_stocks)
            if prev_holdings[q]:
                turnover = 1 - len(current_set & prev_holdings[q]) / max(
                    len(current_set | prev_holdings[q]), 1)
            else:
                turnover = 1.0
            month_turnover[q] = turnover
            prev_holdings[q] = current_set
        
        row["long_short"] = row["Q5"] - row["Q1"]
        portfolio_returns.append(row)
        
        turnover_records.append({
            "date": next_date,
            "Q1_turnover": month_turnover.get("Q1", 0),
            "Q5_turnover": month_turnover.get("Q5", 0),
            "avg_turnover": np.mean([month_turnover.get(q, 0) for q in QUINTILE_LABELS]),
        })
    
    returns_df = pd.DataFrame(portfolio_returns).set_index("date")
    turnover_df = pd.DataFrame(turnover_records).set_index("date")
    
    print(f"[backtest] Portfolio returns: {len(returns_df)} months")
    print(f"[backtest] Avg monthly turnover: {turnover_df['avg_turnover'].mean():.1%}")
    
    return returns_df, turnover_df


def download_spy_returns(returns_df: pd.DataFrame) -> pd.Series:
    """Download SPY monthly returns aligned to our backtest dates."""
    import yfinance as yf
    
    start = returns_df.index.min() - pd.DateOffset(months=1)
    end = returns_df.index.max() + pd.DateOffset(months=1)
    
    spy = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    spy_monthly = spy["Close"].resample("ME").last().pct_change().dropna()
    
    # Handle MultiIndex columns from yfinance
    if isinstance(spy_monthly, pd.DataFrame):
        spy_monthly = spy_monthly.iloc[:, 0]
    
    # Align to our dates
    spy_aligned = spy_monthly.reindex(returns_df.index, method="nearest")
    return spy_aligned


def compute_metrics(returns: pd.Series, spy_returns: pd.Series = None,
                    name: str = "") -> dict:
    """
    Compute comprehensive performance metrics for a return series.
    
    Metrics:
      - Annualized Return: geometric mean of monthly returns, annualized
      - Annualized Volatility: std of monthly returns × sqrt(12)
      - Sharpe Ratio: ann. return / ann. vol (assuming Rf=0)
      - Max Drawdown: largest peak-to-trough decline
      - Calmar Ratio: ann. return / |max drawdown|
      - Information Ratio: (strategy - benchmark) / tracking error, vs SPY
    """
    # Remove NaN
    returns = returns.dropna()
    
    # Cumulative returns
    cum = (1 + returns).cumprod()
    
    # Annualized return (geometric)
    total_return = cum.iloc[-1] - 1
    n_years = len(returns) / 12
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(12)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Max drawdown
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # Information Ratio (vs SPY)
    ir = np.nan
    if spy_returns is not None:
        active = returns - spy_returns.reindex(returns.index).fillna(0)
        tracking_error = active.std() * np.sqrt(12)
        ir = active.mean() * 12 / tracking_error if tracking_error > 0 else 0
    
    metrics = {
        "name": name,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "information_ratio": ir,
        "n_months": len(returns),
    }
    
    return metrics


def transaction_cost_analysis(returns_df: pd.DataFrame,
                              turnover_df: pd.DataFrame,
                              spy_returns: pd.Series) -> pd.DataFrame:
    """
    Sweep transaction costs from 5 to 25 bps one-way.
    
    Transaction cost per month = turnover × cost_per_trade (one-way) × 2 (buy + sell)
    
    Shows at what cost level the strategy breaks even vs SPY.
    """
    results = []
    ls_returns = returns_df["long_short"]
    avg_turnover = turnover_df[["Q1_turnover", "Q5_turnover"]].mean(axis=1)
    
    for bps in TC_BPS_RANGE:
        cost_rate = bps / 10000  # convert bps to decimal
        # Two-way cost: turnover × cost × 2
        monthly_cost = avg_turnover * cost_rate * 2
        net_returns = ls_returns - monthly_cost.reindex(ls_returns.index).fillna(0)
        
        metrics = compute_metrics(net_returns, spy_returns, name=f"{bps} bps")
        metrics["tc_bps"] = bps
        results.append(metrics)
    
    tc_df = pd.DataFrame(results)
    
    # Find breakeven
    breakeven = tc_df[tc_df["ann_return"] <= 0]
    if not breakeven.empty:
        print(f"[backtest] Strategy breaks even at ~{breakeven.iloc[0]['tc_bps']} bps")
    else:
        print(f"[backtest] Strategy is profitable even at {max(TC_BPS_RANGE)} bps")
    
    return tc_df


def factor_return_decomposition(factor_data: pd.DataFrame,
                                monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the return of each individual factor's long-short portfolio.
    Useful to see which factor contributed most to overall performance.
    """
    dates = factor_data.index.get_level_values("date").unique().sort_values()
    factor_names = ["momentum", "value", "size", "lowvol"]
    
    records = []
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        if next_date not in monthly_returns.index:
            continue
        
        month_data = factor_data.loc[date]
        next_returns = monthly_returns.loc[next_date]
        
        row = {"date": next_date}
        for factor in factor_names:
            z_col = f"{factor}_z"
            if z_col not in month_data.columns:
                continue
            
            # Top quintile (Q5) and bottom quintile (Q1) by this factor alone
            scores = month_data[z_col].dropna()
            q_labels = pd.qcut(scores, q=5, labels=QUINTILE_LABELS)
            
            q5_stocks = q_labels[q_labels == "Q5"].index
            q1_stocks = q_labels[q_labels == "Q1"].index
            
            q5_valid = [s for s in q5_stocks if s in next_returns.index and not np.isnan(next_returns[s])]
            q1_valid = [s for s in q1_stocks if s in next_returns.index and not np.isnan(next_returns[s])]
            
            q5_ret = next_returns[q5_valid].mean() if q5_valid else 0
            q1_ret = next_returns[q1_valid].mean() if q1_valid else 0
            row[factor] = q5_ret - q1_ret
        
        records.append(row)
    
    return pd.DataFrame(records).set_index("date")


# ─────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────

def set_plot_style():
    """Professional plot styling."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })


def plot_cumulative_returns(returns_df: pd.DataFrame, spy_returns: pd.Series,
                            save_path: str):
    """Plot cumulative returns: Q5, Q1, Long-Short, and SPY."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col, label, ls in [("Q5", "Q5 (Long)", "-"),
                           ("Q1", "Q1 (Short)", "--"),
                           ("long_short", "Long-Short (Q5−Q1)", "-")]:
        cum = (1 + returns_df[col]).cumprod()
        ax.plot(cum.index, cum.values, label=label, linestyle=ls, linewidth=2)
    
    spy_cum = (1 + spy_returns).cumprod()
    ax.plot(spy_cum.index, spy_cum.values, label="SPY", linestyle=":", 
            linewidth=2, color="gray")
    
    ax.set_title("Cumulative Returns: Factor Portfolio vs SPY")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[backtest] Saved: {save_path}")


def plot_drawdown_waterfall(returns_df: pd.DataFrame, save_path: str):
    """Plot drawdown waterfall for the long-short portfolio."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    cum = (1 + returns_df["long_short"]).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    
    ax.fill_between(drawdown.index, drawdown.values, 0, 
                    color="crimson", alpha=0.4, label="Drawdown")
    ax.plot(drawdown.index, drawdown.values, color="crimson", linewidth=0.8)
    
    ax.set_title("Long-Short Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[backtest] Saved: {save_path}")


def plot_factor_returns(factor_returns: pd.DataFrame, save_path: str):
    """Bar chart of annualized return per factor."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ann_returns = factor_returns.mean() * 12
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    
    bars = ax.bar(ann_returns.index, ann_returns.values, color=colors, edgecolor="white")
    
    for bar, val in zip(bars, ann_returns.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold")
    
    ax.set_title("Annualized Long-Short Return by Factor")
    ax.set_ylabel("Annualized Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axhline(y=0, color="black", linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[backtest] Saved: {save_path}")


def plot_turnover(turnover_df: pd.DataFrame, save_path: str):
    """Plot monthly turnover for Q1 and Q5 portfolios."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.bar(turnover_df.index, turnover_df["Q5_turnover"], 
           alpha=0.6, label="Q5 (Long)", color="#2196F3", width=20)
    ax.bar(turnover_df.index, turnover_df["Q1_turnover"], 
           alpha=0.6, label="Q1 (Short)", color="#FF5722", width=20)
    
    ax.set_title("Monthly Portfolio Turnover")
    ax.set_ylabel("Turnover Fraction")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[backtest] Saved: {save_path}")


def plot_tc_sensitivity(tc_df: pd.DataFrame, save_path: str):
    """Plot Sharpe ratio as a function of transaction costs."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(tc_df["tc_bps"], tc_df["sharpe"], "o-", linewidth=2, 
             color="#2196F3", markersize=8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Transaction Cost (bps, one-way)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe Ratio vs Transaction Costs")
    
    ax2.plot(tc_df["tc_bps"], tc_df["ann_return"] * 100, "o-", linewidth=2,
             color="#4CAF50", markersize=8)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Transaction Cost (bps, one-way)")
    ax2.set_ylabel("Annualized Return (%)")
    ax2.set_title("Annualized Return vs Transaction Costs")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[backtest] Saved: {save_path}")


def generate_results_table(returns_df: pd.DataFrame, spy_returns: pd.Series,
                           factor_returns: pd.DataFrame,
                           tc_df: pd.DataFrame) -> str:
    """Generate a text summary table of all results."""
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST RESULTS SUMMARY")
    lines.append("=" * 70)
    
    # Overall metrics
    for col, label in [("long_short", "Long-Short (Q5−Q1)"),
                       ("Q5", "Q5 (Long Only)"),
                       ("Q1", "Q1 (Short Only)")]:
        m = compute_metrics(returns_df[col], spy_returns, name=label)
        lines.append(f"\n--- {label} ---")
        lines.append(f"  Total Return:       {m['total_return']:>8.1%}")
        lines.append(f"  Annualized Return:  {m['ann_return']:>8.1%}")
        lines.append(f"  Annualized Vol:     {m['ann_volatility']:>8.1%}")
        lines.append(f"  Sharpe Ratio:       {m['sharpe']:>8.2f}")
        lines.append(f"  Max Drawdown:       {m['max_drawdown']:>8.1%}")
        lines.append(f"  Calmar Ratio:       {m['calmar']:>8.2f}")
        lines.append(f"  Info Ratio vs SPY:  {m['information_ratio']:>8.2f}")
    
    # SPY benchmark
    m_spy = compute_metrics(spy_returns, name="SPY")
    lines.append(f"\n--- SPY Benchmark ---")
    lines.append(f"  Annualized Return:  {m_spy['ann_return']:>8.1%}")
    lines.append(f"  Sharpe Ratio:       {m_spy['sharpe']:>8.2f}")
    lines.append(f"  Max Drawdown:       {m_spy['max_drawdown']:>8.1%}")
    
    # Factor decomposition
    lines.append(f"\n--- Individual Factor Returns (annualized) ---")
    for col in factor_returns.columns:
        ann = factor_returns[col].mean() * 12
        lines.append(f"  {col:>12s}:  {ann:>8.1%}")
    
    # Transaction cost sensitivity
    lines.append(f"\n--- Transaction Cost Sensitivity ---")
    lines.append(f"  {'BPS':>5s}  {'Ann Return':>12s}  {'Sharpe':>8s}")
    for _, row in tc_df.iterrows():
        lines.append(f"  {row['tc_bps']:>5.0f}  {row['ann_return']:>12.1%}  {row['sharpe']:>8.2f}")
    
    lines.append("\n" + "=" * 70)
    
    result = "\n".join(lines)
    print(result)
    return result


def run(factor_data: pd.DataFrame, monthly_returns: pd.DataFrame,
        prices: pd.DataFrame = None):
    """
    Main entry point. Run the full backtest and generate all outputs.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Construct portfolios
    print("\n[backtest] === Constructing portfolios ===")
    returns_df, turnover_df = construct_portfolios(factor_data, monthly_returns)
    
    # 2. Download SPY for benchmark comparison
    print("\n[backtest] === Downloading SPY benchmark ===")
    spy_returns = download_spy_returns(returns_df)
    
    # 3. Factor return decomposition
    print("\n[backtest] === Factor return decomposition ===")
    factor_returns = factor_return_decomposition(factor_data, monthly_returns)
    
    # 4. Transaction cost analysis
    print("\n[backtest] === Transaction cost sensitivity ===")
    tc_df = transaction_cost_analysis(returns_df, turnover_df, spy_returns)
    
    # 5. Generate all plots
    print("\n[backtest] === Generating plots ===")
    plot_cumulative_returns(returns_df, spy_returns,
                          os.path.join(OUTPUT_DIR, "cumulative_returns.png"))
    plot_drawdown_waterfall(returns_df,
                          os.path.join(OUTPUT_DIR, "drawdown.png"))
    plot_factor_returns(factor_returns,
                       os.path.join(OUTPUT_DIR, "factor_returns.png"))
    plot_turnover(turnover_df,
                  os.path.join(OUTPUT_DIR, "turnover.png"))
    plot_tc_sensitivity(tc_df,
                        os.path.join(OUTPUT_DIR, "tc_sensitivity.png"))
    
    # 6. Results table
    print("\n[backtest] === Results Summary ===")
    summary = generate_results_table(returns_df, spy_returns, factor_returns, tc_df)
    
    with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w") as f:
        f.write(summary)
    
    # 7. Save returns data
    returns_df.to_csv(os.path.join(OUTPUT_DIR, "portfolio_returns.csv"))
    turnover_df.to_csv(os.path.join(OUTPUT_DIR, "turnover.csv"))
    tc_df.to_csv(os.path.join(OUTPUT_DIR, "tc_sensitivity.csv"), index=False)
    factor_returns.to_csv(os.path.join(OUTPUT_DIR, "factor_returns.csv"))
    
    print(f"\n[backtest] ✓ All results saved to {OUTPUT_DIR}/")
    
    # Return key metrics for README
    ls_metrics = compute_metrics(returns_df["long_short"], spy_returns)
    return ls_metrics, tc_df


if __name__ == "__main__":
    import data
    import factors
    
    # Load data
    prices, volumes, monthly_returns, fundamentals = data.run()
    
    # Compute factors
    factor_data = factors.run(prices, fundamentals, monthly_returns)
    
    # Run backtest
    run(factor_data, monthly_returns, prices)
