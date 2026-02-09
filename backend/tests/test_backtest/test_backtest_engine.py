from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import BacktestConfig, BacktestEngine, BacktestTrade
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal


# ── Test strategy implementations ────────────────────────────


class AlwaysBuyStrategy(Strategy):
    """Strategy that always generates a BUY signal."""

    @property
    def name(self) -> str:
        return "always_buy"

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []
        for symbol, price in market_data.prices.items():
            signals.append(TradingSignal(
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=0.8,
                strategy_name=self.name,
            ))
        return signals


class AlwaysSellStrategy(Strategy):
    """Strategy that always generates a SELL signal."""

    @property
    def name(self) -> str:
        return "always_sell"

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []
        for symbol, price in market_data.prices.items():
            signals.append(TradingSignal(
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=0.8,
                strategy_name=self.name,
            ))
        return signals


class AlternatingStrategy(Strategy):
    """Strategy that alternates between BUY and SELL."""

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "alternating"

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        self._call_count += 1
        action = SignalAction.BUY if self._call_count % 2 == 1 else SignalAction.SELL
        signals = []
        for symbol in market_data.prices:
            signals.append(TradingSignal(
                symbol=symbol,
                action=action,
                confidence=0.7,
                strategy_name=self.name,
            ))
        return signals


class HoldStrategy(Strategy):
    """Strategy that always holds."""

    @property
    def name(self) -> str:
        return "hold"

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        return [TradingSignal(
            symbol=list(market_data.prices.keys())[0],
            action=SignalAction.HOLD,
            confidence=0.5,
            strategy_name=self.name,
        )]


class ErrorStrategy(Strategy):
    """Strategy that raises an error."""

    @property
    def name(self) -> str:
        return "error"

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        raise RuntimeError("Strategy error")


# ── Test data generators ─────────────────────────────────────


def make_uptrend_data(n=200, start_price=100.0):
    """Generate upward trending OHLCV data."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, n)  # Slight upward bias
    close = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(5000, 50000, n),
    })


def make_downtrend_data(n=200, start_price=100.0):
    """Generate downward trending OHLCV data."""
    rng = np.random.default_rng(42)
    returns = rng.normal(-0.001, 0.01, n)
    close = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(5000, 50000, n),
    })


def make_flat_data(n=200, price=100.0):
    """Generate flat OHLCV data."""
    rng = np.random.default_rng(42)
    close = np.full(n, price) + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": close,
        "high": close + abs(rng.normal(0, 0.3, n)),
        "low": close - abs(rng.normal(0, 0.3, n)),
        "close": close,
        "volume": rng.integers(5000, 50000, n),
    })


# ── Basic engine tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_backtest_returns_result():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert result.strategy_name == "alternating"
    assert result.symbol == "AAPL"
    assert len(result.equity_curve) > 0
    assert result.total_bars == 150  # 200 - 50 (min_bars)
    assert "total_return_pct" in result.metrics


@pytest.mark.asyncio
async def test_backtest_insufficient_data():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=HoldStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        min_bars=100,
    )
    data = make_uptrend_data(50)

    with pytest.raises(ValueError, match="Insufficient data"):
        await engine.run(config, data)


# ── Trading logic tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_buy_and_hold_strategy():
    """AlwaysBuy should buy once and hold (no sell signal)."""
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlwaysBuyStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    # Should have at least 1 trade (end-of-backtest close)
    assert len(result.trades) >= 1
    # Final equity should differ from initial
    assert result.metrics["final_equity"] != 5000.0


@pytest.mark.asyncio
async def test_alternating_strategy_generates_trades():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert len(result.trades) > 1


@pytest.mark.asyncio
async def test_hold_strategy_no_trades():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=HoldStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert len(result.trades) == 0
    assert result.metrics["final_equity"] == 5000.0


@pytest.mark.asyncio
async def test_sell_without_position_ignored():
    """AlwaysSell with no position should not generate trades."""
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlwaysSellStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert len(result.trades) == 0


# ── Stop-loss tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_loss_triggers_on_downtrend():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlwaysBuyStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        stop_loss_pct=2.0,  # Tight 2% stop
    )
    data = make_downtrend_data(200)
    result = await engine.run(config, data)

    # Should have stop_loss exits
    stop_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
    assert len(stop_trades) > 0


# ── Commission and slippage tests ────────────────────────────


@pytest.mark.asyncio
async def test_commission_impacts_equity():
    engine = BacktestEngine()

    # High commission
    config_high = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        commission_pct=1.0,
    )
    # Low commission
    config_low = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        commission_pct=0.01,
    )

    data = make_uptrend_data(200)
    result_high = await engine.run(config_high, data)
    result_low = await engine.run(config_low, data)

    # Higher commission should result in lower equity
    assert result_high.metrics["final_equity"] < result_low.metrics["final_equity"]


@pytest.mark.asyncio
async def test_total_commission_tracked():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert "total_commission" in result.metrics
    assert result.metrics["total_commission"] > 0


# ── Equity curve tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_equity_curve_starts_at_initial():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=HoldStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        initial_capital=10000.0,
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert result.equity_curve[0]["equity"] == 10000.0


@pytest.mark.asyncio
async def test_equity_curve_length():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=HoldStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        min_bars=50,
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    # equity_curve entries = total_bars
    assert len(result.equity_curve) == result.total_bars


# ── Benchmark comparison tests ───────────────────────────────


@pytest.mark.asyncio
async def test_benchmark_metrics_present():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert "benchmark_return_pct" in result.benchmark_metrics
    assert "strategy_return_pct" in result.benchmark_metrics
    assert "excess_return_pct" in result.benchmark_metrics


# ── Trade metadata tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_trade_has_bars_held():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlternatingStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    if result.trades:
        trade = result.trades[0]
        assert hasattr(trade, "bars_held")
        assert trade.bars_held >= 0


@pytest.mark.asyncio
async def test_trade_exit_reasons():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=AlwaysBuyStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
        stop_loss_pct=1.0,  # Very tight stop
    )
    data = make_downtrend_data(200)
    result = await engine.run(config, data)

    reasons = {t.exit_reason for t in result.trades}
    # Should have at least stop_loss or end_of_backtest
    assert len(reasons) > 0


# ── Strategy error handling ──────────────────────────────────


@pytest.mark.asyncio
async def test_strategy_error_does_not_crash():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=ErrorStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-09",
    )
    data = make_uptrend_data(200)
    # Should not raise
    result = await engine.run(config, data)
    assert len(result.trades) == 0


# ── Config tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_result_config_populated():
    engine = BacktestEngine()
    config = BacktestConfig(
        strategy=HoldStrategy(),
        symbol="AAPL",
        start_date="2025-01-01",
        end_date="2025-06-01",
        initial_capital=10000.0,
        commission_pct=0.2,
    )
    data = make_uptrend_data(200)
    result = await engine.run(config, data)

    assert result.config["initial_capital"] == 10000.0
    assert result.config["commission_pct"] == 0.2
    assert result.config["start_date"] == "2025-01-01"
