import numpy as np
import pandas as pd
import pytest

from app.strategy.feature_pipeline import (
    FeaturePipelineConfig,
    balance_classes,
    create_target,
    get_feature_importance,
    normalize_features,
    prepare_ml_data,
    remove_highly_correlated,
    remove_low_variance,
    select_feature_columns,
    time_based_split,
)


def make_ohlcv(n=300):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": close + rng.normal(0, 0.5, n),
        "high": close + abs(rng.normal(0, 1, n)),
        "low": close - abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": rng.integers(1000, 100000, n),
    })


# ── create_target tests ──────────────────────────────────────


def test_target_has_three_classes():
    df = make_ohlcv(300)
    target = create_target(df, forward_periods=5, buy_threshold=0.005, sell_threshold=-0.005)
    unique = set(target.dropna().unique())
    # Should have at least 2 of {0, 1, 2}
    assert len(unique) >= 2
    assert unique.issubset({0, 1, 2})


def test_target_length_matches_input():
    df = make_ohlcv(200)
    target = create_target(df)
    assert len(target) == len(df)


def test_target_buy_for_positive_return():
    df = pd.DataFrame({
        "close": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0],
        "open": [100.0] * 6,
        "high": [110.0] * 6,
        "low": [90.0] * 6,
        "volume": [1000] * 6,
    })
    target = create_target(df, forward_periods=5, buy_threshold=0.005)
    # First row: close=100, close[5]=110 → 10% return → BUY (2)
    assert target.iloc[0] == 2


# ── select_feature_columns tests ─────────────────────────────


def test_select_excludes_ohlcv():
    from app.data.indicators import compute_features
    df = make_ohlcv(100)
    features = compute_features(df)
    cols = select_feature_columns(features)
    assert "close" not in cols
    assert "open" not in cols
    assert "timestamp" not in cols
    assert "sma_10" in cols
    assert "rsi_14" in cols


# ── remove_low_variance tests ────────────────────────────────


def test_low_variance_removes_constant():
    df = pd.DataFrame({
        "a": [1.0] * 100,  # constant
        "b": np.random.randn(100),  # normal
    })
    kept = remove_low_variance(df, ["a", "b"], min_variance=1e-8)
    assert "a" not in kept
    assert "b" in kept


# ── remove_highly_correlated tests ───────────────────────────


def test_correlated_removes_one():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({
        "a": x,
        "b": x + rng.normal(0, 0.01, 100),  # Nearly identical to a
        "c": rng.normal(0, 1, 100),  # Independent
    })
    kept = remove_highly_correlated(df, ["a", "b", "c"], threshold=0.95)
    # One of a/b should be removed
    assert len(kept) == 2
    assert "c" in kept


# ── normalize_features tests ─────────────────────────────────


def test_normalize_zero_mean():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"a": rng.normal(5, 2, 100), "b": rng.normal(10, 3, 100)})
    val = pd.DataFrame({"a": rng.normal(5, 2, 30), "b": rng.normal(10, 3, 30)})
    test = pd.DataFrame({"a": rng.normal(5, 2, 20), "b": rng.normal(10, 3, 20)})

    train_n, val_n, test_n, stats = normalize_features(train, val, test)

    # Training set should be approximately zero mean
    assert abs(train_n["a"].mean()) < 0.2
    assert abs(train_n["b"].mean()) < 0.2
    assert "mean" in stats
    assert "std" in stats


# ── time_based_split tests ───────────────────────────────────


def test_split_preserves_order():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    train, val, test = time_based_split(df, 0.70, 0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Order preserved
    assert train.iloc[-1]["x"] < val.iloc[0]["x"]
    assert val.iloc[-1]["x"] < test.iloc[0]["x"]


def test_split_no_overlap():
    df = pd.DataFrame({"x": range(200)})
    train, val, test = time_based_split(df, 0.60, 0.20)
    all_indices = set(train.index) | set(val.index) | set(test.index)
    assert len(all_indices) == len(df)


# ── prepare_ml_data tests ────────────────────────────────────


def test_prepare_ml_data_full_pipeline():
    df = make_ohlcv(500)
    config = FeaturePipelineConfig(
        forward_periods=5,
        train_pct=0.70,
        val_pct=0.15,
    )
    data = prepare_ml_data(df, config, normalize=True)

    assert len(data.X_train) > 0
    assert len(data.X_val) > 0
    assert len(data.X_test) > 0
    assert len(data.feature_columns) > 5
    assert set(data.y_train.unique()).issubset({0, 1, 2})


def test_prepare_ml_data_insufficient_data():
    df = make_ohlcv(50)  # Too few rows
    with pytest.raises(ValueError, match="Insufficient data"):
        prepare_ml_data(df)


# ── get_feature_importance tests ─────────────────────────────


# ── balance_classes tests ────────────────────────────────────


def test_balance_classes():
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0] * 10 + [1] * 60 + [2] * 30)  # imbalanced: 10/60/30
    X_bal, y_bal = balance_classes(X, y)
    # Should oversample to majority count (60)
    assert len(y_bal) == 180  # 60 per class
    counts = y_bal.value_counts()
    assert counts[0] == 60
    assert counts[1] == 60
    assert counts[2] == 60


def test_balance_classes_preserves_order():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0] * 20 + [1] * 50 + [2] * 30)
    X_bal, y_bal = balance_classes(X, y)
    # Indices should be in ascending (chronological) order
    indices = X_bal.index.tolist()
    assert indices == sorted(indices)


# ── get_feature_importance tests ─────────────────────────────


def test_feature_importance_with_mock_model():
    class MockModel:
        feature_importances_ = np.array([0.3, 0.1, 0.5, 0.1])

    importance = get_feature_importance(MockModel(), ["a", "b", "c", "d"], top_n=3)
    assert len(importance) == 3
    assert list(importance.keys())[0] == "c"  # Highest importance
