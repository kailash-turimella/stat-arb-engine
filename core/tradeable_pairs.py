import pandas as pd
import statsmodels.api as sm
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

    for i in range(len(tickers) - 1):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            s1, s2 = price_dict[t1], price_dict[t2]
            df = pd.concat([s1, s2], axis=1).dropna()
            if len(df) < 100:
                continue  # skip if not enough data

            score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
            print(f"p-value: {pvalue} for the stocks {t1}  {t2}")
            if pvalue < pval_threshold:
                selected_pairs.append((t1, t2))

    return selected_pairs


def filter_tradeable_pairs(price_dict, cointegrated_pairs, min_corr=0.85, min_spread_std=0.5):
    """
    Apply correlation, spread volatility, and z-score stability filters.

    Args:
        price_dict (dict): {ticker: price_series}
        cointegrated_pairs (list): from find_cointegrated_pairs
        min_corr (float): correlation threshold
        min_spread_std (float): minimum spread std deviation

    Returns:
        list of tradeable pairs
    """
    selected_pairs = []

    for t1, t2 in cointegrated_pairs:
        s1, s2 = price_dict[t1], price_dict[t2]
        df = pd.concat([s1, s2], axis=1).dropna()
        if len(df) < 100:
            continue

        # Filter out pairs with weak short-term correlation
        x, y = df.iloc[:, 0], df.iloc[:, 1]
        corr = x.corr(y)
        if corr < min_corr:
            continue

        # Filter out pairs whose spread doesn't fluctuate enough to be tradable
        y_with_const = sm.add_constant(y)
        model = sm.OLS(x, y_with_const).fit()
        hedge_ratio = model.params.iloc[1]
        spread = x - hedge_ratio * y
        spread_std = spread.std()
        if spread_std < min_spread_std:
            continue

        # Filter out pairs where the spread doesn't stay close to the mean consistently
        zscore = (spread - spread.mean()) / spread_std
        stability_score = (zscore.abs() < 1).mean()
        if stability_score < 0.5:
            continue

        selected_pairs.append((t1, t2))

    return selected_pairs


def find_tradeable_pairs(price_dict, pval_threshold=0.05, min_corr=0.85, min_spread_std=0.5):
    """
    Pipeline: cointegration test + filters for tradability.

    Args:
        price_dict (dict): {ticker: price_series}
        pval_threshold (float): significance level
        min_corr (float): correlation threshold
        min_spread_std (float): spread std threshold

    Returns:
        list of filtered cointegrated pairs
    """
    raw_pairs = find_cointegrated_pairs(price_dict, pval_threshold)
    print("Cointegrated pairs:", raw_pairs)
    tradeable_pairs = filter_tradeable_pairs(price_dict, raw_pairs, min_corr, min_spread_std)
    return tradeable_pairs


# if __name__ == "__main__":
#     from fetch_data import fetch_prices

#     tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "SPY", "IVV"]
#     start = "2023-01-01"
#     end = "2023-06-01"

#     # Step 2: Fetch adjusted close prices
#     price_data = fetch_prices(tickers, start, end)

#     # Step 3: Run tradeable pair detection
#     tradeable = find_tradeable_pairs(price_data)

#     # Step 4: Print final result
#     print("\nFiltered Tradeable Pairs:")
#     for p in tradeable:
#         print(p)
