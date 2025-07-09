import yfinance as yf
import pandas as pd
import os
import pickle

def fetch_prices(tickers, start_date, end_date, use_cache=True, cache_dir="cache"):
    """
    Fetch adjusted close prices for a list of tickers between start_date and end_date.
    Caches results locally to avoid re-fetching.

    Args:
        tickers (list): List of stock tickers.
        start_date (str): Format 'YYYY-MM-DD'.
        end_date (str): Format 'YYYY-MM-DD'.
        use_cache (bool): If True, load from cache if available.
        cache_dir (str): Directory to store cache files.

    Returns:
        dict: {ticker: Series of adjusted close prices}
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    prices = {}
    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker}_{start_date}_{end_date}.pkl")

        if use_cache and os.path.exists(cache_path):
            prices[ticker] = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            print(f"Fetching {ticker} from Yahoo Finance...")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if "Close" in data.columns:
                series = data["Close"].dropna().ffill()
                series.to_csv(cache_path)
                prices[ticker] = series
            else:
                print(f"Warning: No 'Close' column found for {ticker}. Skipping.")

    return prices



# Optional: Test block
if __name__ == "__main__":
    test_tickers = ['AAPL', 'MSFT']
    data = fetch_prices(test_tickers, '2023-01-01', '2023-03-01')
    for ticker in test_tickers:
        if ticker in data:
            print(f"\n{ticker} prices:")
            print(data[ticker].head())
        else:
            print(f"{ticker} data not found.")
