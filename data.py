"""
data.py — Download, clean, and align S&P 500 price + fundamental data.

This module handles:
  1. Fetching current S&P 500 constituent tickers from Wikipedia
  2. Downloading 5 years of daily OHLCV data via yfinance
  3. Downloading fundamental data (P/B ratio, market cap) via yfinance
  4. Cleaning: forward-fill missing prices, drop tickers with >20% missing data
  5. Saving aligned price and fundamental DataFrames to CSV

SURVIVORSHIP BIAS NOTE:
  We use the *current* S&P 500 membership list, which means stocks that were
  removed (due to bankruptcy, acquisition, or delisting) are excluded. This
  creates a look-ahead bias that inflates backtested returns. In a production
  setting, you would use a point-in-time constituent list from CRSP or Compustat.
  We acknowledge this limitation explicitly.
"""

import os
import datetime
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import warnings
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
YEARS_OF_DATA = 5
MIN_HISTORY_FRACTION = 0.80  # drop tickers with <80% of trading days


# def get_sp500_tickers() -> list[str]:
#     """
#     Scrape current S&P 500 tickers from Wikipedia.
#     Returns a sorted list of ticker symbols with '.' replaced by '-'
#     (Yahoo Finance convention, e.g., BRK.B -> BRK-B).
#     """
#     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#     tables = pd.read_html(url)
#     df = tables[0]
#     tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
#     print(f"[data] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
#     return sorted(tickers)

def get_sp500_tickers() -> list[str]:
    """
    Scrape current S&P 500 tickers from Wikipedia with a custom User-Agent.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Define a header to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Fetch the page content first
    response = requests.get(url, headers=headers)
    
    # Pass the HTML text to pandas
    tables = pd.read_html(response.text)
    df = tables[0]
    
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"[data] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
    return sorted(tickers)


def download_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted close prices and volume for all tickers.
    Uses yfinance's bulk download for efficiency.

    Returns:
        prices: DataFrame with columns = tickers, index = dates (daily adj close)
        volumes: DataFrame with columns = tickers, index = dates (daily volume)
    """
    print(f"[data] Downloading price data from {start} to {end}...")
    
    # Download in batches to avoid timeouts
    batch_size = 50
    all_adj_close = []
    all_volume = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)
        print(f"[data]   Batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1} "
              f"({len(batch)} tickers)...")
        
        try:
            raw = yf.download(batch_str, start=start, end=end,
                              auto_adjust=True, threads=True, progress=False)
            
            if raw.empty:
                continue
            
            # yfinance returns MultiIndex columns: (field, ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                adj_close = raw["Close"]
                volume = raw["Volume"]
            else:
                # Single ticker case
                adj_close = raw[["Close"]].rename(columns={"Close": batch[0]})
                volume = raw[["Volume"]].rename(columns={"Volume": batch[0]})
            
            all_adj_close.append(adj_close)
            all_volume.append(volume)
        except Exception as e:
            print(f"[data]   Warning: batch failed with {e}")
        
        time.sleep(1)  # be polite to Yahoo's servers
    
    prices = pd.concat(all_adj_close, axis=1) if all_adj_close else pd.DataFrame()
    volumes = pd.concat(all_volume, axis=1) if all_volume else pd.DataFrame()
    
    print(f"[data] Downloaded prices for {prices.shape[1]} tickers, "
          f"{prices.shape[0]} trading days")
    return prices, volumes


def download_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Download fundamental data for each ticker:
      - marketCap (for Size factor)
      - priceToBook (for Value factor)
    
    Returns a DataFrame indexed by ticker with columns [marketCap, priceToBook].
    
    Note: yfinance returns the LATEST fundamentals, not historical. For a proper
    backtest you'd need quarterly fundamental snapshots from a provider like
    Compustat. We use trailing values as an approximation.
    """
    print(f"[data] Downloading fundamentals for {len(tickers)} tickers...")
    
    records = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"[data]   Progress: {i + 1}/{len(tickers)}")
            time.sleep(1)
        
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                "marketCap": info.get("marketCap", np.nan),
                "priceToBook": info.get("priceToBook", np.nan),
            })
        except Exception:
            records.append({
                "ticker": ticker,
                "marketCap": np.nan,
                "priceToBook": np.nan,
            })
    
    df = pd.DataFrame(records).set_index("ticker")
    valid = df.dropna().shape[0]
    print(f"[data] Got fundamentals for {valid}/{len(tickers)} tickers")
    return df


def clean_price_data(prices: pd.DataFrame, volumes: pd.DataFrame,
                     min_frac: float = MIN_HISTORY_FRACTION):
    """
    Clean price data:
      1. Drop tickers with too many missing values
      2. Forward-fill remaining gaps (weekends/holidays already excluded)
      3. Drop any remaining NaN rows at the start
    """
    n_days = len(prices)
    min_days = int(n_days * min_frac)
    
    # Count valid (non-NaN) days per ticker
    valid_counts = prices.notna().sum()
    keep = valid_counts[valid_counts >= min_days].index.tolist()
    dropped = prices.shape[1] - len(keep)
    
    print(f"[data] Cleaning: keeping {len(keep)} tickers, "
          f"dropping {dropped} with <{min_frac * 100:.0f}% history")
    
    prices = prices[keep].ffill().bfill()
    volumes = volumes[[c for c in keep if c in volumes.columns]].ffill().fillna(0)
    
    return prices, volumes


def build_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end and compute monthly returns.
    Returns a DataFrame of monthly returns (index = month-end dates).
    """
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    print(f"[data] Built monthly returns: {monthly_returns.shape[0]} months, "
          f"{monthly_returns.shape[1]} tickers")
    return monthly_returns


def run(force_download: bool = False):
    """
    Main entry point. Downloads and saves all data to the data/ directory.
    If data already exists, loads from disk unless force_download=True.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    prices_path = os.path.join(DATA_DIR, "prices.csv")
    volumes_path = os.path.join(DATA_DIR, "volumes.csv")
    returns_path = os.path.join(DATA_DIR, "monthly_returns.csv")
    fundamentals_path = os.path.join(DATA_DIR, "fundamentals.csv")
    tickers_path = os.path.join(DATA_DIR, "tickers.csv")
    
    if not force_download and os.path.exists(prices_path):
        print("[data] Loading cached data from disk...")
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        volumes = pd.read_csv(volumes_path, index_col=0, parse_dates=True)
        monthly_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        fundamentals = pd.read_csv(fundamentals_path, index_col=0)
        print(f"[data] Loaded: {prices.shape[1]} tickers, {prices.shape[0]} days")
        return prices, volumes, monthly_returns, fundamentals
    
    # Step 1: Get tickers
    tickers = get_sp500_tickers()
    
    # Step 2: Set date range
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=365 * YEARS_OF_DATA)).strftime("%Y-%m-%d")
    
    # Step 3: Download prices
    prices, volumes = download_price_data(tickers, start_date, end_date)
    
    # Step 4: Clean
    prices, volumes = clean_price_data(prices, volumes)
    
    # Step 5: Monthly returns
    monthly_returns = build_monthly_returns(prices)
    
    # Step 6: Fundamentals
    fundamentals = download_fundamentals(prices.columns.tolist())
    
    # Step 7: Save everything
    prices.to_csv(prices_path)
    volumes.to_csv(volumes_path)
    monthly_returns.to_csv(returns_path)
    fundamentals.to_csv(fundamentals_path)
    pd.DataFrame({"ticker": prices.columns.tolist()}).to_csv(tickers_path, index=False)
    
    print(f"\n[data] ✓ All data saved to {DATA_DIR}/")
    print(f"  prices.csv          : {prices.shape}")
    print(f"  volumes.csv         : {volumes.shape}")
    print(f"  monthly_returns.csv : {monthly_returns.shape}")
    print(f"  fundamentals.csv    : {fundamentals.shape}")
    
    return prices, volumes, monthly_returns, fundamentals


if __name__ == "__main__":
    run(force_download=True)
