"""Feature engineering pipeline for ML strategies.

Handles feature selection, target labeling, train/test splitting,
and normalization for preparing trading ML data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from app.data.indicators import compute_features

logger = structlog.get_logger()

# Columns that are not features (raw OHLCV + timestamp)
NON_FEATURE_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

# Features excluded due to instability / heavy drift across market regimes.
# macd_cross is a discrete -1/0/1 signal that walk-forward consistently flags
# as drifting on out-of-sample data.
EXCLUDED_FEATURES = {"macd_cross"}


@dataclass
class DataSplit:
    """A time-based train/validation/test split."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]


@dataclass
class FeaturePipelineConfig:
    """Configuration for the feature pipeline."""
    # Target labeling
    forward_periods: int = 5  # Look ahead N bars for return (aligned with train_model.py)
    buy_threshold: float = 0.008  # +0.8% = BUY (class 2)
    sell_threshold: float = -0.008  # -0.8% = SELL (class 0)
    # Binary mode: BUY vs NOT_BUY (ignores sell_threshold)
    binary_mode: bool = True
    # Triple-barrier labelling — label by whether TP is hit before SL within the
    # holding window, matching production exit geometry. When True, overrides the
    # close-to-close binary target. tp/sl are fractions (0.045 = 4.5%).
    use_triple_barrier: bool = False
    tb_take_profit_pct: float = 0.045
    tb_stop_loss_pct: float = 0.03
    # Feature selection
    correlation_threshold: float = 0.85  # Remove features above this correlation
    min_variance: float = 1e-8  # Remove near-constant features
    # Split ratios (time-based)
    train_pct: float = 0.70
    val_pct: float = 0.15
    # test_pct is implicit: 1 - train - val


def create_target(
    df: pd.DataFrame,
    forward_periods: int = 10,
    buy_threshold: float = 0.015,
    sell_threshold: float = -0.015,
) -> pd.Series:
    """Create a 3-class target based on forward returns.

    0 = SELL (negative return exceeding threshold)
    1 = HOLD (return within thresholds)
    2 = BUY  (positive return exceeding threshold)
    """
    forward_return = df["close"].shift(-forward_periods) / df["close"] - 1

    target = pd.Series(1, index=df.index, name="target")  # Default HOLD
    target[forward_return > buy_threshold] = 2   # BUY
    target[forward_return < sell_threshold] = 0  # SELL

    return target


def create_binary_target(
    df: pd.DataFrame,
    forward_periods: int = 5,
    buy_threshold: float = 0.015,
) -> pd.Series:
    """Create a binary target: 1=BUY (profitable entry), 0=NOT_BUY.

    Uses close-to-close forward return to match live trading reality.
    Previous version used max forward high which created look-ahead bias —
    the model learned to buy when intraday spikes occurred, but live trading
    can't exit at the exact high.
    """
    future_close = df["close"].shift(-forward_periods)
    forward_return = future_close / df["close"] - 1
    target = (forward_return > buy_threshold).astype(int)
    target.name = "target"
    return target


def create_triple_barrier_target(
    df: pd.DataFrame,
    forward_periods: int = 10,
    take_profit_pct: float = 0.045,
    stop_loss_pct: float = 0.03,
) -> pd.Series:
    """Label 1=BUY if the take-profit barrier is hit before the stop barrier.

    This matches what the system actually trades: a position is opened, a stop
    and a take-profit are set, and the trade resolves at whichever barrier the
    price touches first within the holding window. Training on close-to-close
    forward return (create_binary_target) teaches the model direction but not
    whether a trade *survives its stop* — which is where the live edge leaked.

    For each bar, scans the next `forward_periods` bars:
      - hits TP (high >= entry*(1+tp)) before SL (low <= entry*(1-sl)) → 1
      - hits SL first, or neither and closes below entry → 0
    The final `forward_periods` rows get NaN (insufficient lookahead) and are
    dropped downstream.
    """
    entry = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - forward_periods):
        e = entry[i]
        if e <= 0:
            continue
        tp_level = e * (1 + take_profit_pct)
        sl_level = e * (1 - stop_loss_pct)
        outcome = 0
        for j in range(i + 1, i + 1 + forward_periods):
            hit_tp = high[j] >= tp_level
            hit_sl = low[j] <= sl_level
            if hit_tp and hit_sl:
                # Both barriers touched in the same bar — assume the stop fills
                # first (conservative; avoids over-optimistic labels).
                outcome = 0
                break
            if hit_tp:
                outcome = 1
                break
            if hit_sl:
                outcome = 0
                break
        else:
            # Neither barrier hit — label by whether it closed above entry
            outcome = int(close[i + forward_periods] > e)
        labels[i] = outcome

    target = pd.Series(labels, index=df.index, name="target")
    return target


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select all computed feature columns (exclude raw OHLCV, target, and unstable features)."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c not in EXCLUDED_FEATURES and c != "target"
    ]


def remove_low_variance(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_variance: float = 1e-8,
) -> list[str]:
    """Remove features with near-zero variance."""
    variances = df[feature_cols].var()
    kept = variances[variances > min_variance].index.tolist()
    removed = set(feature_cols) - set(kept)
    if removed:
        logger.debug("feature_pipeline.low_variance_removed", count=len(removed))
    return kept


def remove_highly_correlated(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Remove one of each pair of highly correlated features."""
    if len(feature_cols) < 2:
        return feature_cols

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.add(col)

    kept = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        logger.debug(
            "feature_pipeline.correlated_removed",
            count=len(to_drop),
            dropped=list(to_drop)[:5],
        )
    return kept


def normalize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Z-score normalize features using train set statistics.

    Returns normalized datasets and the normalization stats (mean, std).
    """
    means = X_train.mean()
    stds = X_train.std().replace(0, 1)  # Avoid division by zero

    X_train_norm = (X_train - means) / stds
    X_val_norm = (X_val - means) / stds
    X_test_norm = (X_test - means) / stds

    stats = {"mean": means.to_dict(), "std": stds.to_dict()}
    return X_train_norm, X_val_norm, X_test_norm, stats


def time_based_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    embargo: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically (no shuffling - critical for time series).

    Args:
        embargo: Number of rows to drop from the END of each preceding block
            (train, val) so their forward-looking targets cannot peek into the
            next block. With a forward-return label of N periods, the last N
            rows of train have targets computed from the first N rows of val —
            classic look-ahead leakage that inflates validation/test accuracy.
            Set embargo >= forward_periods to purge it.
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    embargo = max(0, int(embargo))
    train = df.iloc[: max(0, train_end - embargo)]
    val = df.iloc[train_end : max(train_end, val_end - embargo)]
    test = df.iloc[val_end:]

    return train, val, test


def balance_classes(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample minority classes to the count of the majority class.

    Preserves all training data (no information loss) by duplicating
    random samples from underrepresented classes.
    """
    class_counts = y.value_counts()
    max_count = class_counts.max()
    balanced_indices: list = []
    rng = np.random.default_rng(random_state)
    for cls in class_counts.index:
        cls_indices = y[y == cls].index.tolist()
        balanced_indices.extend(cls_indices)
        if len(cls_indices) < max_count:
            extra_needed = max_count - len(cls_indices)
            oversampled = rng.choice(cls_indices, size=extra_needed, replace=True).tolist()
            balanced_indices.extend(oversampled)
    balanced_indices.sort()  # preserve chronological order
    return X.loc[balanced_indices], y.loc[balanced_indices]


def prepare_ml_data(
    raw_df: pd.DataFrame,
    config: FeaturePipelineConfig | None = None,
    normalize: bool = True,
    features_precomputed: bool = False,
) -> DataSplit:
    """Full pipeline: features → target → selection → split → normalize.

    Args:
        raw_df: DataFrame with OHLCV columns (timestamp, open, high, low, close, volume).
            When features_precomputed=True, expects features + "target" column already present.
        config: Pipeline configuration. Uses defaults if None.
        normalize: Whether to z-score normalize features.
        features_precomputed: If True, skip compute_features() and create_target().
            Used when features were computed per-symbol before combining.

    Returns:
        DataSplit with train/val/test data ready for ML training.
    """
    if config is None:
        config = FeaturePipelineConfig()

    if features_precomputed:
        features_df = raw_df.copy()
    else:
        # 1. Compute technical indicators
        features_df = compute_features(raw_df)

        # 2. Create target
        if config.use_triple_barrier:
            features_df["target"] = create_triple_barrier_target(
                features_df,
                forward_periods=config.forward_periods,
                take_profit_pct=config.tb_take_profit_pct,
                stop_loss_pct=config.tb_stop_loss_pct,
            )
        elif config.binary_mode:
            features_df["target"] = create_binary_target(
                features_df,
                forward_periods=config.forward_periods,
                buy_threshold=config.buy_threshold,
            )
        else:
            features_df["target"] = create_target(
                features_df,
                forward_periods=config.forward_periods,
                buy_threshold=config.buy_threshold,
                sell_threshold=config.sell_threshold,
            )

    # 3. Drop rows with NaN (from indicators warmup + forward labeling)
    features_df = features_df.dropna()

    if len(features_df) < 100:
        raise ValueError(
            f"Insufficient data after feature computation: {len(features_df)} rows "
            "(need at least 100)"
        )

    # 4. Select and filter features
    feature_cols = select_feature_columns(features_df)
    feature_cols = remove_low_variance(features_df, feature_cols, config.min_variance)
    feature_cols = remove_highly_correlated(
        features_df, feature_cols, config.correlation_threshold
    )

    logger.info("feature_pipeline.features_selected", count=len(feature_cols))

    # 5. Time-based split with an embargo equal to the label horizon so the
    #    forward-return targets near each boundary can't leak into the next set.
    train_df, val_df, test_df = time_based_split(
        features_df, config.train_pct, config.val_pct,
        embargo=config.forward_periods,
    )

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    # 5b. Balance training classes
    # In binary mode, skip oversampling — use XGBoost scale_pos_weight instead
    if not config.binary_mode:
        X_train, y_train = balance_classes(X_train, y_train)

    # 6. Normalize
    if normalize:
        X_train, X_val, X_test, _ = normalize_features(X_train, X_val, X_test)

    logger.info(
        "feature_pipeline.prepared",
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
        features=len(feature_cols),
        class_dist=y_train.value_counts().to_dict(),
    )

    return DataSplit(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_columns=feature_cols,
    )


def get_feature_importance(
    model,
    feature_columns: list[str],
    top_n: int = 15,
) -> dict[str, float]:
    """Extract feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = (
            np.abs(model.coef_).mean(axis=0)
            if model.coef_.ndim > 1
            else np.abs(model.coef_)
        )
    else:
        return {}

    importance_dict = dict(zip(feature_columns, importances))
    sorted_importance = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )
    return sorted_importance
