"""Tests for walk-forward validation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from app.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
    WindowResult,
)
from app.strategy.base import TrainResult


def _make_data(days: int = 365) -> pd.DataFrame:
    """Create synthetic OHLCV data for walk-forward testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=days * 8, freq="h")  # 8 bars/day
    n = len(dates)
    close = 100 + np.cumsum(np.random.normal(0.01, 0.5, n))
    close = np.maximum(close, 10)

    return pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + np.random.normal(0, 0.002, n)),
        "high": close * (1 + abs(np.random.normal(0, 0.005, n))),
        "low": close * (1 - abs(np.random.normal(0, 0.005, n))),
        "close": close,
        "volume": np.random.randint(100000, 500000, n),
    })


class TestWalkForwardConfig:
    def test_defaults(self):
        config = WalkForwardConfig()
        assert config.train_days == 180
        assert config.test_days == 30
        assert config.step_days == 30
        assert config.initial_capital == 5000.0


class TestWalkForwardResult:
    def test_to_dict(self):
        result = WalkForwardResult(
            windows=[
                WindowResult(
                    window_num=1,
                    train_start="2024-01-01",
                    train_end="2024-07-01",
                    test_start="2024-07-01",
                    test_end="2024-08-01",
                    train_metrics={"total_return_pct": 5.0},
                    test_metrics={"total_return_pct": 3.0},
                    train_trades=25,
                    test_trades=5,
                )
            ],
            aggregate_metrics={"total_return_pct": 3.0},
            degradation_pct=40.0,
            robustness_score=0.6,
        )
        d = result.to_dict()
        assert d["num_windows"] == 1
        assert d["degradation_pct"] == 40.0
        assert d["robustness_score"] == 0.6
        assert len(d["windows"]) == 1
        assert d["windows"][0]["train_trades"] == 25

    def test_empty_result(self):
        result = WalkForwardResult()
        d = result.to_dict()
        assert d["num_windows"] == 0
        assert d["degradation_pct"] == 0.0


class TestWalkForwardEngine:
    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        engine = WalkForwardEngine()
        config = WalkForwardConfig(train_days=180, test_days=30)
        # Only 100 days of data → insufficient
        data = _make_data(days=50)
        strategy = MagicMock()

        with pytest.raises(ValueError, match="Insufficient data"):
            await engine.run(config, strategy, data, "SPY")

    @pytest.mark.asyncio
    async def test_missing_timestamp_column(self):
        engine = WalkForwardEngine()
        config = WalkForwardConfig()
        data = pd.DataFrame({"close": [100, 101, 102]})
        strategy = MagicMock()

        with pytest.raises(ValueError, match="timestamp"):
            await engine.run(config, strategy, data, "SPY")

    @pytest.mark.asyncio
    async def test_robustness_score_range(self):
        """Robustness score should be between 0 and 1."""
        result = WalkForwardResult(
            windows=[
                WindowResult(
                    window_num=1,
                    train_start="2024-01-01",
                    train_end="2024-07-01",
                    test_start="2024-07-01",
                    test_end="2024-08-01",
                    train_metrics={"total_return_pct": 5.0},
                    test_metrics={"total_return_pct": 3.0},
                    train_trades=10,
                    test_trades=5,
                ),
            ],
            robustness_score=0.6,
        )
        assert 0 <= result.robustness_score <= 1
