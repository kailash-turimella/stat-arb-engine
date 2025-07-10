import pandas as pd
import statsmodels.api as sm
from core.machine_learning import (
    build_training_data,
    train_logistic_regression_model,
    predict
)


def compute_hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    """
    Compute the hedge ratio between two price series using OLS regression.

    Args:
        x (pd.Series): Price series of asset X.
        y (pd.Series): Price series of asset Y.

    Returns:
        float: Hedge ratio from OLS regression.
    """
    y_with_const = sm.add_constant(y)
    model = sm.OLS(x, y_with_const).fit()
    return model.params.iloc[1]  # Slope is the hedge ratio


def generate_signal(x: pd.Series, y: pd.Series, z_entry: float = 1.0, z_exit: float = 0.1, confidence_threshold: float = 0.6) -> dict:
    """
    Generate a trade signal (long/short/none) for a pair using ML prediction.

    Args:
        x (pd.Series): Price series of asset X.
        y (pd.Series): Price series of asset Y.
        z_entry (float): Threshold for z-score entry.
        z_exit (float): Threshold to consider mean-reversion.

    Returns:
        dict: Trade signal with type, confidence, and hedge ratio.
    """
    hedge_ratio = compute_hedge_ratio(x, y)

    # Adaptive window expansion for training data
    for window_multiplier in [1, 2]:
        try:
            X, y_labels = build_training_data(x[-10 * window_multiplier:], y[-10 * window_multiplier:], hedge_ratio)
            if y_labels.nunique() >= 2:
                break
        except Exception:
            continue
    else:
        # Fallback to basic logic if model training is not possible
        print("failed to train model, using fallback logic")
        spread = x - hedge_ratio * y
        zscore = (spread - spread.rolling(10).mean()) / spread.rolling(10).std()
        current_z = zscore.iloc[-1]

        if abs(current_z) > z_entry:
            direction = 'long' if current_z < 0 else 'short'
            return {'signal': direction, 'confidence': 0.5, 'hedge_ratio': hedge_ratio}
        else:
            return {'signal': 'none', 'confidence': 0.0, 'hedge_ratio': hedge_ratio}

    # Train model and make prediction
    model = train_logistic_regression_model(X, y_labels)
    print("MODEL TRAINED")
    pred_label, confidence = predict(model, X)

    spread = x - hedge_ratio * y
    zscore = (spread - spread.rolling(10).mean()) / spread.rolling(10).std()
    current_z = zscore.iloc[-1]

    if abs(current_z) < z_entry or confidence < confidence_threshold or pred_label == 0:
        return {'signal': 'none', 'confidence': confidence, 'hedge_ratio': hedge_ratio}

    direction = 'long' if current_z < 0 else 'short'
    return {'signal': direction, 'confidence': confidence, 'hedge_ratio': hedge_ratio}
