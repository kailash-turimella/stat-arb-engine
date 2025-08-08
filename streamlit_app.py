import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import fetch_data
import cointegration
import pair_analysis
import signals
import backtester


def main():
    st.title("Pairs Trading Strategy Dashboard")

    # Sidebar inputs
    st.sidebar.header("Configuration")
    x = st.sidebar.text_input("Ticker X", value="SPY")
    y = st.sidebar.text_input("Ticker Y", value="IVV")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    entry_threshold = st.sidebar.number_input("Entry Threshold (z-score)", value=1.0)
    exit_threshold = st.sidebar.number_input("Exit Threshold (z-score)", value=0.1)
    window = st.sidebar.number_input("Rolling Window", value=60)
    notional_per_trade = st.sidebar.number_input("Notional per Leg ($)", value=10000)

    if st.sidebar.button("Run Strategy"):
        st.write("## 1. Fetching Data...")
        df = fetch_data.fetch_data(x, y, start_date, end_date)
        st.write(df.tail())

        st.write("## 2. Testing Cointegration...")
        pval, beta = cointegration.test_cointegration(df[x], df[y])
        st.write(f"**p-value:** {pval:.4f}, **beta:** {beta:.4f}")
        if pval >= 0.05:
            st.warning(f"{x} and {y} are NOT cointegrated. Try different tickers.")
            return

        st.write("## 3. Calculating Spread & Z-score...")
        spread, zscore = pair_analysis.get_spread_and_zscore(df[x], df[y], beta, window)
        st.line_chart(zscore.dropna())

        st.write("## 4. Generating Trade Signals...")
        signal_series = signals.get_trade_signals(zscore, entry_threshold, exit_threshold)
        
        # Map string signals to numeric values for plotting
        signal_numeric = signal_series.replace({"BUY": 1, "SELL": -1, "HOLD": 0})
        st.line_chart(signal_numeric.dropna())

        st.write("## 5. Running Backtest...")
        bt_df = backtester.backtest_signals(spread, signal_series)

        # Convert PnL in spread units to dollar terms
        bt_df["Dollar PnL"] = bt_df["PnL"] * (notional_per_trade / df[x])  # Assumes notional in x
        bt_df["Cumulative Dollar PnL"] = bt_df["Dollar PnL"].cumsum()

        total_dollar_pnl = bt_df["Cumulative Dollar PnL"].iloc[-1]
        st.success(f"**Total Cumulative PnL: ${total_dollar_pnl:,.2f}**")

        st.write("## 6. Spread with Buy/Sell Signals")
        fig, ax = plt.subplots()
        ax.plot(spread.index, spread, label="Spread")
        ax.plot(bt_df[bt_df["Position"] == 1].index, spread[bt_df["Position"] == 1], "^", color="g", label="Buy", markersize=8)
        ax.plot(bt_df[bt_df["Position"] == -1].index, spread[bt_df["Position"] == -1], "v", color="r", label="Sell", markersize=8)
        ax.set_title("Spread and Trade Signals")
        ax.legend()
        st.pyplot(fig)

        st.write("## 7. Cumulative PnL")
        st.line_chart(bt_df["Cumulative Dollar PnL"])


if __name__ == "__main__":
    main()