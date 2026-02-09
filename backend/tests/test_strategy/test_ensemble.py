from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal
from app.strategy.ensemble import EnsembleStrategy


class MockStrategy(Strategy):
    def __init__(self, strategy_name: str, signals: list[TradingSignal]):
        self._name = strategy_name
        self._signals = signals

    @property
    def name(self) -> str:
        return self._name

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        return self._signals


def make_signal(symbol="AAPL", action=SignalAction.BUY, confidence=0.8, strategy="test"):
    return TradingSignal(
        symbol=symbol, action=action, confidence=confidence, strategy_name=strategy,
    )


def make_snapshot():
    return MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        prices={"AAPL": 150.0},
        ohlcv={},
        features={},
    )


# ── Weighted voting tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_all_buy_gives_buy():
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.BUY, confidence=0.8, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.BUY, confidence=0.7, strategy="strat2")])
    ensemble = EnsembleStrategy([s1, s2])

    signals = await ensemble.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.BUY


@pytest.mark.asyncio
async def test_all_sell_gives_sell():
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.SELL, confidence=0.9, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.SELL, confidence=0.8, strategy="strat2")])
    ensemble = EnsembleStrategy([s1, s2])

    signals = await ensemble.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.SELL


@pytest.mark.asyncio
async def test_all_hold_gives_hold():
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.HOLD, confidence=0.5, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.HOLD, confidence=0.5, strategy="strat2")])
    ensemble = EnsembleStrategy([s1, s2])

    signals = await ensemble.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.HOLD


# ── Conflict resolution tests ────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_reduces_confidence():
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.BUY, confidence=0.9, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.SELL, confidence=0.5, strategy="strat2")])
    ensemble = EnsembleStrategy([s1, s2])

    signals = await ensemble.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].metadata["conflict"] is True
    # Confidence should be reduced due to conflict
    assert signals[0].confidence < 0.9


@pytest.mark.asyncio
async def test_weak_conflict_becomes_hold():
    """Equal BUY and SELL → HOLD (no agreement)."""
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.BUY, confidence=0.5, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.SELL, confidence=0.5, strategy="strat2")])
    ensemble = EnsembleStrategy([s1, s2], agreement_threshold=0.3)

    signals = await ensemble.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.HOLD


# ── Weight tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_weights():
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.BUY, confidence=0.7, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.SELL, confidence=0.7, strategy="strat2")])
    # strat1 has much higher weight → BUY should win
    ensemble = EnsembleStrategy([s1, s2], weights={"strat1": 3.0, "strat2": 1.0})

    signals = await ensemble.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.BUY


def test_get_weights():
    s1 = MockStrategy("strat1", [])
    s2 = MockStrategy("strat2", [])
    ensemble = EnsembleStrategy([s1, s2], weights={"strat1": 0.6, "strat2": 0.4})
    assert ensemble.get_weights() == {"strat1": 0.6, "strat2": 0.4}


# ── Dynamic weight adjustment tests ──────────────────────────


def test_record_outcome():
    s1 = MockStrategy("strat1", [])
    ensemble = EnsembleStrategy([s1])

    ensemble.record_outcome("AAPL", "strat1", True)
    ensemble.record_outcome("AAPL", "strat1", False)
    assert len(ensemble._signal_history["strat1"]) == 2


def test_update_weights_insufficient_data():
    s1 = MockStrategy("strat1", [])
    ensemble = EnsembleStrategy([s1], weights={"strat1": 1.0})

    # Only 5 records, need 10 minimum
    for _ in range(5):
        ensemble.record_outcome("AAPL", "strat1", True)

    weights = ensemble.update_weights_from_history()
    assert weights["strat1"] == 1.0  # Unchanged


def test_update_weights_adjusts_with_data():
    s1 = MockStrategy("strat1", [])
    s2 = MockStrategy("strat2", [])
    ensemble = EnsembleStrategy([s1, s2])

    # strat1: 80% accuracy
    for i in range(20):
        ensemble.record_outcome("AAPL", "strat1", i % 5 != 0)

    # strat2: 30% accuracy
    for i in range(20):
        ensemble.record_outcome("AAPL", "strat2", i % 3 == 0)

    weights = ensemble.update_weights_from_history()
    assert weights["strat1"] > weights["strat2"]


# ── Metadata tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metadata_includes_votes():
    s1 = MockStrategy("strat1", [make_signal(action=SignalAction.BUY, confidence=0.8, strategy="strat1")])
    s2 = MockStrategy("strat2", [make_signal(action=SignalAction.HOLD, confidence=0.6, strategy="strat2")])
    ensemble = EnsembleStrategy([s1, s2])

    signals = await ensemble.generate_signals(make_snapshot())
    assert signals[0].metadata["strategy_votes"]["strat1"] == "BUY"
    assert signals[0].metadata["strategy_votes"]["strat2"] == "HOLD"
    assert signals[0].metadata["num_strategies"] == 2
