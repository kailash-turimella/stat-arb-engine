import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd

"""
    Performs Engle-Granger cointegration test between two time series to get the p-value.
    Calculates hedge ratio (beta) from linear regression: x = alpha + beta * y to get the beta.
    
    Args:
        x (pd.Series): Price series of stock A
        y (pd.Series): Price series of stock B

    Returns:
        float: p-value of the cointegration test (p < 0.05 = cointegrated)
        float: beta coefficient
"""
def test_cointegration(x, y):
    score, pvalue, _ = coint(x, y)
    model = sm.OLS(x, sm.add_constant(y)).fit()
    beta = model.params.iloc[1]
    return pvalue, beta




# --- Test code ---
def test_test_cointegration():
    print("Running cointegration module test...")
    import fetch_data

    stocks_a = ["AAPL", "XOM", "SPY"]
    stocks_b = ["MSFT", "CVX", "IVV"]
    start = "2018-01-01"
    end = "2023-01-01"

    for stock_a, stock_b in zip(stocks_a, stocks_b):
        df = fetch_data.fetch_data(stock_a, stock_b, start, end)
        x = df[stock_a]
        y = df[stock_b]

        pvalue, beta = test_cointegration(x, y)

        if pvalue < 0.05:
            print(f"{stock_a} and {stock_b} are cointegrated; p-value: {pvalue:.4f}, beta: {beta:.4f}")
        else:
            print(f"{stock_a} and {stock_b} are NOT cointegrated; p-value: {pvalue:.4f}, beta: {beta:.4f}")


test_test_cointegration()