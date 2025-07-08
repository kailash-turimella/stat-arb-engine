import matplotlib.pyplot as plt

"""
    Plots the spread and overlays LONG/SHORT/CLOSE trade signals.

    Args:
        df (pd.DataFrame): Must contain 'Spread', 'Signal' columns with datetime index
"""
def plot_spread_with_signals(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Spread'], label='Spread', color='black')

    # Plot signals
    for signal, color, marker in [("LONG", "green", "^"), ("SHORT", "red", "v"), ("CLOSE", "blue", "o")]:
        signal_idx = df[df["Signal"] == signal]
        plt.scatter(signal_idx.index, signal_idx['Spread'], label=signal, color=color, marker=marker, s=70)

    plt.title("Spread with Trade Signals")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



"""
    Plots the z-score over time with threshold lines.

    Args:
        zscore (pd.Series): Time series of z-score
        entry_threshold (float)
        exit_threshold (float)
"""
def plot_zscore(zscore, entry_threshold=1.0, exit_threshold=0.1):
    plt.figure(figsize=(14, 4))
    plt.plot(zscore.index, zscore, label="Z-score", color='black')

    plt.axhline(entry_threshold, color='red', linestyle='--', label='Entry Threshold')
    plt.axhline(-entry_threshold, color='red', linestyle='--')
    plt.axhline(exit_threshold, color='blue', linestyle='--', label='Exit Threshold')
    plt.axhline(-exit_threshold, color='blue', linestyle='--')
    plt.axhline(0, color='gray', linestyle='-')

    plt.title("Z-score with Entry/Exit Thresholds")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
