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

    # Build features and labels (train on historical data)
    X, y_labels = build_training_data(x, y, hedge_ratio)
    model = train_logistic_regression_model(X, y_labels)

    # Predict on the latest feature vector
    pred_label, confidence = predict(model, X)

    # Get current z-score to decide direction
    spread = x - hedge_ratio * y
    zscore = (spread - spread.rolling(10).mean()) / spread.rolling(10).std()
    current_z = zscore.iloc[-1]

    # Determine signal type based on prediction and z-score
    if abs(current_z) < z_entry or confidence < confidence_threshold or pred_label == 0:
        return {'signal': 'none', 'confidence': confidence, 'hedge_ratio': hedge_ratio}

    direction = 'long' if current_z < 0 else 'short'
    return {'signal': direction, 'confidence': confidence, 'hedge_ratio': hedge_ratio}
