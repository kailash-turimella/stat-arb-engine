import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(price_dict, pval_threshold=0.05):
    """
    Test all pairs for cointegration using the Engle-Granger test.

    Args:
        price_dict (dict): {ticker: price_series}
        pval_threshold (float): significance level for cointegration

    Returns:
        list of tuples: [(ticker1, ticker2), ...] where p-value < threshold
    """
    tickers = list(price_dict.keys())
    selected_pairs = []

    for i in range(len(tickers) - 1) :
        for j in range(i + 1, len(tickers)):
            t1 = tickers[i]
            t2 = tickers[j]
            s1 = price_dict[t1]
            s2 = price_dict[t2]

            # Align on common dates
            df = pd.concat([s1, s2], axis=1).dropna()
            if len(df) < 100:
                continue  # skip if not enough data

            score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
            # print(f"p-value: {pvalue} for the stocks {t1}  {t2}")
            if pvalue < pval_threshold:
                selected_pairs.append((t1, t2))

    return selected_pairs



# if __name__ == "__main__":
#     from fetch_data import fetch_prices
#     prices = fetch_prices(['AAPL', 'GOOG', 'MSFT', 'SPY', 'AMZN', 'IVV'], '2023-01-01', '2023-06-01')
#     pairs = find_cointegrated_pairs(prices)
#     print("Cointegrated pairs:", pairs)
