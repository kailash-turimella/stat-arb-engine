import pandas as pd
from core.signals import generate_signal


def backtest_pairs(price_data: dict, pairs: list, window_size: int = 200, max_holding: int = 20):
    """
    Backtest a list of pairs using ML-enhanced signal generation and simple spread trading logic.

    Args:
        price_data (dict): Dictionary of {ticker: price_series}
        pairs (list): List of tuples [(ticker1, ticker2), ...]
        window_size (int): Number of days for historical window to compute signals
        max_holding (int): Max holding period in days

    Returns:
        pd.DataFrame: Trade log containing entry/exit details and PnL
    """
    trades = []

    for t1, t2 in pairs:
        x = price_data[t1]
        y = price_data[t2]
        df = pd.concat([x, y], axis=1).dropna()
        df.columns = ['x', 'y']

        current_trade = None  # Track if a trade is open

        for i in range(window_size, len(df) - 1):
            window_df = df.iloc[i - window_size:i]
            x_window = window_df['x']
            y_window = window_df['y']

            signal = generate_signal(x_window, y_window)
            date = df.index[i]

            if current_trade:
                holding_days = (date - current_trade['entry_date']).days
                spread = df.iloc[i]['x'] - current_trade['hedge_ratio'] * df.iloc[i]['y']

                # Check exit
                if abs(spread) < 0.1 or holding_days >= max_holding:
                    pnl = (spread - current_trade['entry_spread']) * (1 if current_trade['direction'] == 'long' else -1)
                    trades.append({
                        **current_trade,
                        'exit_date': date,
                        'exit_spread': spread,
                        'pnl': pnl,
                        'holding_days': holding_days
                    })
                    current_trade = None
                continue

            # Entry condition
            if signal['signal'] in ['long', 'short']:
                spread = df.iloc[i]['x'] - signal['hedge_ratio'] * df.iloc[i]['y']
                current_trade = {
                    'pair': (t1, t2),
                    'entry_date': date,
                    'entry_spread': spread,
                    'direction': signal['signal'],
                    'hedge_ratio': signal['hedge_ratio'],
                    'confidence': signal['confidence']
                }

    return pd.DataFrame(trades)
