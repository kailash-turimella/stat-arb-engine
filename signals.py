import pandas as pd

"""
    Generates trade signals based on z-score thresholds.

    Args:
        zscore (pd.Series): Series of z-scores
        entry_threshold (float): Threshold for opening trades
        exit_threshold (float): Threshold for closing trades

    Returns:
        pd.Series: Signal series with values: "LONG", "SHORT", "CLOSE", "HOLD"
"""
def get_trade_signals(z_scores, entry_threshold: float = 1.0, exit_threshold: float = 0.1):
    signals = []

    position = None

    for z in z_scores:
        if position is None:
            if z > entry_threshold:
                signals.append("SHORT")
                position = "SHORT"
            elif z < -entry_threshold:
                signals.append("LONG")
                position = "LONG"
            else:
                signals.append("HOLD")
        elif position == "LONG" or position == "SHORT":
            if abs(z) < exit_threshold:
                signals.append("CLOSE")
                position = None
            else:
                signals.append("HOLD")

    return pd.Series(signals, index=z_scores.index)




def test_signals():
    print("Running signals module test...")
    import fetch_data
    import cointegration
    import pair_analysis

    x, y = "SPY", "IVV"
    df = fetch_data.fetch_data(x, y, "2018-01-01", "2023-01-01")
    pval, beta = cointegration.test_cointegration(df[x], df[y])
    
    if pval >= 0.05:
        print("Pair not cointegrated — skipping.")
        return

    _, zscore = pair_analysis.get_spread_and_zscore(df[x], df[y], beta, window=30)
    signals = get_trade_signals(zscore)

    print(signals.value_counts())
    print(signals.head(10))
    print(signals.tail(10))



# test_signals()