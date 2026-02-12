from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction
from app.strategy.ml_strategy import MLStrategy


def make_ohlcv(n=200):
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


def make_snapshot(symbols_data: dict[str, pd.DataFrame]) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        prices={s: df["close"].iloc[-1] for s, df in symbols_data.items()},
        ohlcv=symbols_data,
        features={},
    )


# ── Init tests ────────────────────────────────────────────────


def test_ml_strategy_init_no_model():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    assert strategy._model is None
    assert strategy.name == "ml_xgboost"


def test_ml_strategy_confidence_threshold():
    strategy = MLStrategy(confidence_threshold=0.7, model_path="/nonexistent/path.pkl")
    assert strategy._confidence_threshold == 0.7


# ── Signal generation tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_no_signals_without_model():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    snapshot = make_snapshot({"AAPL": make_ohlcv()})
    signals = await strategy.generate_signals(snapshot)
    assert signals == []


@pytest.mark.asyncio
async def test_signals_with_mock_model():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")

    # Create mock model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.05, 0.1, 0.85]])  # BUY
    strategy._model = mock_model

    # Set feature columns to match what compute_features produces
    from app.data.indicators import compute_features
    df = make_ohlcv(200)
    features_df = compute_features(df)
    feature_cols = [
        c for c in features_df.columns
        if c not in {"timestamp", "open", "high", "low", "close", "volume"}
    ]
    strategy._feature_columns = feature_cols

    snapshot = make_snapshot({"AAPL": df})
    signals = await strategy.generate_signals(snapshot)

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].action == SignalAction.BUY
    assert signals[0].confidence > 0.75


@pytest.mark.asyncio
async def test_low_confidence_becomes_hold():
    strategy = MLStrategy(confidence_threshold=0.8, model_path="/nonexistent/path.pkl")

    mock_model = MagicMock()
    # Probabilities below threshold
    mock_model.predict_proba.return_value = np.array([[0.3, 0.4, 0.3]])
    strategy._model = mock_model

    from app.data.indicators import compute_features
    df = make_ohlcv(200)
    features_df = compute_features(df)
    feature_cols = [
        c for c in features_df.columns
        if c not in {"timestamp", "open", "high", "low", "close", "volume"}
    ]
    strategy._feature_columns = feature_cols

    snapshot = make_snapshot({"AAPL": df})
    signals = await strategy.generate_signals(snapshot)

    assert len(signals) == 1
    assert signals[0].action == SignalAction.HOLD


@pytest.mark.asyncio
async def test_skip_insufficient_data():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    strategy._model = MagicMock()
    strategy._feature_columns = ["sma_10"]

    tiny_df = make_ohlcv(10)
    snapshot = make_snapshot({"AAPL": tiny_df})
    signals = await strategy.generate_signals(snapshot)
    assert signals == []


# ── get_model_info tests ─────────────────────────────────────


def test_model_info_no_model():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    info = strategy.get_model_info()
    assert info["model_loaded"] is False
    assert info["name"] == "ml_xgboost"


def test_model_info_with_metadata():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    strategy._model = MagicMock()
    strategy._model_metadata = {
        "val_accuracy": 0.65,
        "trained_at": "2026-01-01",
    }
    info = strategy.get_model_info()
    assert info["model_loaded"] is True
    assert info["val_accuracy"] == 0.65


# ── get_confidence tests ─────────────────────────────────────


def test_confidence_default():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    assert strategy.get_confidence() == 0.5


def test_confidence_from_metadata():
    strategy = MLStrategy(model_path="/nonexistent/path.pkl")
    strategy._model_metadata = {"val_accuracy": 0.72}
    assert strategy.get_confidence() == 0.72
