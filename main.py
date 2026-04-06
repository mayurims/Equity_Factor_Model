"""
main.py — Run the entire factor model pipeline end-to-end.

Usage:
    python main.py                  # run full pipeline (uses cached data if available)
    python main.py --force-download # re-download all data from scratch
    python main.py --backtest-only  # skip data download, just re-run backtest
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Sectional Equity Factor Model + Walk-Forward Backtest"
    )
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download of all data")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Skip data download, just re-run backtest")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # ── Step 1: Data Pipeline ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1/3: DATA PIPELINE")
    print("=" * 60)
    
    import data
    prices, volumes, monthly_returns, fundamentals = data.run(
        force_download=args.force_download
    )
    
    # ── Step 2: Factor Construction ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2/3: FACTOR CONSTRUCTION")
    print("=" * 60)
    
    import factors
    if args.backtest_only:
        import pandas as pd
        factor_data = pd.read_csv("data/factor_signals.csv", index_col=[0, 1], 
                                   parse_dates=[0])
        print("[factors] Loaded cached factor signals")
    else:
        factor_data = factors.run(prices, fundamentals, monthly_returns)
    
    # ── Step 3: Backtest + Analytics ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3/3: BACKTEST + ANALYTICS")
    print("=" * 60)
    
    import backtest
    metrics, tc_df = backtest.run(factor_data, monthly_returns, prices)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed:.0f} seconds")
    print(f"{'=' * 60}")
    print(f"\nKey Results:")
    print(f"  Sharpe Ratio:       {metrics['sharpe']:.2f}")
    print(f"  Annualized Return:  {metrics['ann_return']:.1%}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:.1%}")
    print(f"  Information Ratio:  {metrics['information_ratio']:.2f}")
    print(f"\nOutputs saved to: results/")
    print(f"  - cumulative_returns.png")
    print(f"  - drawdown.png")
    print(f"  - factor_returns.png")
    print(f"  - turnover.png")
    print(f"  - tc_sensitivity.png")
    print(f"  - results_summary.txt")


if __name__ == "__main__":
    main()
