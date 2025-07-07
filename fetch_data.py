import yfinance as yf

"""
    Fetches daily adjusted closing prices for two tickers from Yahoo Finance.

    Args:
        ticker1 (str): First stock symbol (e.g., "AAPL")
        ticker2 (str): Second stock symbol (e.g., "MSFT")
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame with two columns of adjusted close prices
"""
def fetch_data(ticker1, ticker2, start, end):
    data = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=False)

    try:
        adj_close = data['Adj Close']
    except KeyError:
        print("Could not find 'Adj Close' in downloaded data. Here's what was returned:")
        print(data.columns)
        raise

    return adj_close[[ticker1, ticker2]].dropna()



# --- Testing Methods ---
def test_fetch_data_valid():
    df = fetch_data("AAPL", "MSFT", "2022-01-01", "2022-03-01")
    assert not df.empty, "Test failed: DataFrame is empty."
    assert "AAPL" in df.columns and "MSFT" in df.columns, "Test failed: Missing expected columns."
    print("test_fetch_data_valid passed.")


# test_fetch_data_valid()
