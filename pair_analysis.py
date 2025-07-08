import pandas as pd
import numpy as np

"""
    Calculates the spread between two price series given a hedge ratio, and then calculates the z-score of the spread.

    spread = x - beta * y
    z = (spread - mean) / std

    Args:
        x (pd.Series): Price series for stock A
        y (pd.Series): Price series for stock B
        beta (float): Hedge ratio from OLS regression
        window (int, optional): Rolling window for dynamic z-score. Defaults to entire series.

    Returns:
        pd.Series: Spread series (Difference between the two price series)
        pd.Series: z-score series (How far the current spread is from its average, in units of standard deviation)
"""
def get_spread_and_zscore(x, y, beta, window: int = None) :
    spread = x - beta * y

    if window:
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
    else:
        mean = spread.mean()
        std = spread.std()

    zscore = (spread - mean) / std

    return spread, zscore



# --- Test example ---
def test_strategy_module():
    print("Running strategy module test...")
    import fetch_data
    import cointegration

    x, y = "SPY", "IVV"
    data = fetch_data.fetch_data(x, y, "2018-01-01", "2023-01-01")

    pval, beta = cointegration.test_cointegration(data[x], data[y])
    print(f"p-value: {pval:.4f}, beta: {beta:.4f}")

    spread, zscore = get_spread_and_zscore(data[x], data[y], beta)

    print("Spread sample:")
    print(spread.head())

    print("\nZ-score sample:")
    print(zscore.head())



# test_strategy_module()