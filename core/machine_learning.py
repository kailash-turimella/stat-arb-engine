import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.api as sm


def generate_features(x, y, hedge_ratio, window=10):
    """
    Generate input features for machine learning from spread behavior.

    Args:
        x (pd.Series): Price series of asset X.
        y (pd.Series): Price series of asset Y.
        hedge_ratio (float): Ratio to hedge Y against X (from OLS regression).
        window (int): Rolling window size for stats like z-score and std.

    Returns:
        pd.DataFrame: A feature matrix with columns like z-score, volatility ratio, etc.
    """
    spread = x - hedge_ratio * y
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    df = pd.DataFrame({
        'zscore': zscore,
        'zscore_lag1': zscore.shift(1),
        'zscore_lag2': zscore.shift(2),
        'spread': spread,
        'spread_std': spread.rolling(window).std(),
        'spread_mean': spread.rolling(window).mean(),
    })

    df['spread_velocity'] = spread.diff(window)
    df['volatility_ratio'] = x.rolling(window).std() / y.rolling(window).std()
    return df.dropna()




def generate_labels(x, y, hedge_ratio, z_entry=1.0, z_exit=0.1, max_holding=20):
    """
    Generate binary labels based on whether the spread reverts to the mean within a max holding period after exceeding a z-score threshold.

    Args:
        x (pd.Series): Price series of asset X.
        y (pd.Series): Price series of asset Y.
        hedge_ratio (float): Hedge ratio for calculating spread.
        z_entry (float): Z-score threshold to consider as trade entry.
        z_exit (float): Z-score value considered close to mean.
        max_holding (int): Maximum days to wait for mean reversion.

    Returns:
        pd.Series: Binary labels (1 if mean reversion happens within max_holding, else 0).
    """
    spread = x - hedge_ratio * y
    rolling_mean = spread.rolling(10).mean()
    rolling_std = spread.rolling(10).std()
    zscore = (spread - rolling_mean) / rolling_std

    labels = pd.Series(0, index=zscore.index)

    for i in range(len(zscore) - max_holding):
        z = zscore.iloc[i]
        if abs(z) < z_entry:
            continue  # no signal

        # Search for mean reversion in next max_holding steps
        future = zscore.iloc[i+1:i+max_holding+1]
        reverted = future.abs() < z_exit

        if reverted.any():
            labels.iloc[i] = 1

    return labels




def build_training_data(x, y, hedge_ratio, window=10):
    """
    Combine features and labels to prepare a machine learning training dataset.

    Args:
        x (pd.Series): Price series of asset X.
        y (pd.Series): Price series of asset Y.
        hedge_ratio (float): Hedge ratio used to calculate the spread.
        window (int): Rolling window for feature calculations.
        hold_period (int): Lookahead period for labeling.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - Feature DataFrame (X) for training.
            - Label Series (y) with binary outcomes.
    """
    features = generate_features(x, y, hedge_ratio, window)
    labels = generate_labels(x, y, hedge_ratio)
    aligned = features.join(labels.rename('label')).dropna()
    return aligned.drop(columns=['spread', 'spread_mean']), aligned['label']



def train_logistic_regression_model(X, y, model_type='logistic'):
    """
    Train a machine learning classification model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        model_type (str): Type of classifier to train. Currently only 'logistic' is supported.

    Returns:
        sklearn.base.BaseEstimator: Trained classifier object.
    """
    if model_type == 'logistic':
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X, y)
    return model



def predict(model, features_df):
    """
    Predict trade direction using the latest row of features.

    Args:
        model (sklearn.base.BaseEstimator): Trained classifier.
        features_df (pd.DataFrame): Feature DataFrame including the latest row.

    Returns:
        Tuple[int, float]: 
            - Predicted label (0 or 1) for the latest row.
            - Probability of the positive class (i.e., confidence score).
    """
    latest = features_df.iloc[[-1]]
    return model.predict(latest)[0], model.predict_proba(latest)[0][1]
