import os
import pandas as pd
from datetime import datetime

from core.fetch_data import fetch_prices
from core.tradeable_pairs import find_tradeable_pairs
from core.backtester import backtest_pairs

# === CONFIGURATION ===
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "SPY", "IVV"]
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
DATA_DIR = "cache"
RESULTS_DIR = "results"
WINDOW_SIZE = 100
MAX_HOLDING = 20

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=== 1. Fetching Data ===")
    price_data = fetch_prices(TICKERS, START_DATE, END_DATE)
    print(f"Fetched data for {len(price_data)} tickers.")

    print("\n=== 2. Finding Tradeable Pairs ===")
    tradeable_pairs = find_tradeable_pairs(price_data)
    print(f"Tradeable pairs found: {tradeable_pairs}")

    if not tradeable_pairs:
        print("No tradeable pairs found. Exiting.")
        return

    print("\n=== 3. Running Backtest ===")
    results = backtest_pairs(price_data, tradeable_pairs, window_size=WINDOW_SIZE, max_holding=MAX_HOLDING)

    print("\n=== 4. Trade Summary ===")
    if results.empty:
        print("No trades were executed.")
    else:
        print(results)
        pnl = results['pnl'].sum()
        print(f"\nTotal PnL: {pnl:.2f} spreads")
        results.to_csv(os.path.join(RESULTS_DIR, "trades_summary.csv"), index=False)

if __name__ == "__main__":
    main()
