import fetch_data
import cointegration
import pair_analysis
import signals
import backtester
import plotter


def main():
    # sample run
    x, y = "SPY", "IVV"
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    entry_threshold = 1.0
    exit_threshold = 0.1
    window = 60

    # 1. Fetch data
    print("Fetching data...")
    df = fetch_data.fetch_data(x, y, start_date, end_date)

    # 2. Test cointegration
    print("Testing cointegration...")
    pval, beta = cointegration.test_cointegration(df[x], df[y])
    print(f"p-value: {pval:.4f}, beta: {beta:.4f}")

    if pval >= 0.05:
        print(f"{x} and {y} are NOT cointegrated. Exiting.")
        return

    # 3. Compute spread and z-score
    print("Calculating spread and z-score...")
    spread, zscore = pair_analysis.get_spread_and_zscore(df[x], df[y], beta, window)
    print("Z-score range:", zscore.min(), zscore.max())

    # 4. Generate signals
    print("Generating trade signals...")
    signal_series = signals.get_trade_signals(zscore, entry_threshold, exit_threshold)

    # 5. Backtest strategy
    print("Running backtest...")
    bt_df = backtester.backtest_signals(spread, signal_series)
    print(f"Total return: {bt_df['Cumulative PnL'].iloc[-1]:.2f}")

    # 6. Plot results
    print("Plotting results...")
    plotter.plot_spread_with_signals(bt_df)
    plotter.plot_zscore(zscore, entry_threshold, exit_threshold)


if __name__ == "__main__":
    main()