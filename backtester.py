import pandas as pd

"""
    Backtests the trading signals based on the spread.

    Args:
        spread (pd.Series): Spread series
        signals (pd.Series): Series of trade signals ("LONG", "SHORT", "CLOSE", "HOLD")

    Returns:
        pd.DataFrame: DataFrame with daily PnL, cumulative PnL, and position
"""
def backtest_signals(spread, signals):
    position = 0
    position_history = []
    pnl = [0]  # First day = no PnL

    for i in range(1, len(spread)):
        signal = signals.iloc[i]

        # Update position
        if signal == "LONG":
            position = 1
        elif signal == "SHORT":
            position = -1
        elif signal == "CLOSE":
            position = 0
        # HOLD → no change

        # Calculate daily PnL
        daily_pnl = position * (spread.iloc[i] - spread.iloc[i - 1])
        pnl.append(daily_pnl)
        position_history.append(position)

    # Warn if a position remains open at the end
    if position != 0:
        print(f"Backtest ended with an open {position} position; PnL is unrealized.")
        unrealized_pnl = position * (spread.iloc[-1] - spread.iloc[-2])
        print(f"Unrealized PnL: {unrealized_pnl:.2f}")

    # Pad position history for first row
    position_history = [0] + position_history

    result = pd.DataFrame({
        "Spread": spread,
        "Signal": signals,
        "Position": position_history,
        "PnL": pnl
    })

    result["Cumulative PnL"] = result["PnL"].cumsum()
    return result





def test_backtester():
    print("Running backtester test...")
    import fetch_data
    import cointegration
    import pair_analysis
    import signals

    x, y = "SPY", "IVV"
    entry = 1
    exit = 0.1
    window = 60

    df = fetch_data.fetch_data(x, y, "2018-01-01", "2023-01-01")
    pval, beta = cointegration.test_cointegration(df[x], df[y])
    if pval >= 0.05:
        print("Pair not cointegrated.")
        return

    spread, zscore = pair_analysis.get_spread_and_zscore(df[x], df[y], beta, window)
    print("Z-score range:", zscore.min(), zscore.max())
    signal_series = signals.get_trade_signals(zscore, entry, exit)
    bt_df = backtest_signals(spread, signal_series)

    # print(bt_df.tail())
    print(f"Total return: {bt_df['Cumulative PnL'].iloc[-1]:.2f}")


# test_backtester()