from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.broker.mock_adapter import MockBrokerAdapter
from app.data.market_data import BarAccumulator, MarketDataService, Tick

# ── BarAccumulator tests ─────────────────────────────────────────


def make_tick(symbol: str, price: float, volume: int = 100, minutes_offset: int = 0) -> Tick:
    ts = datetime(2026, 1, 1, 10, 0, tzinfo=UTC) + timedelta(minutes=minutes_offset)
    return Tick(symbol=symbol, price=price, volume=volume, timestamp=ts)


def test_bar_accumulator_first_tick():
    acc = BarAccumulator(symbol="AAPL", bar_size_minutes=5)
    result = acc.add_tick(make_tick("AAPL", 150.0))
    assert result is None
    assert acc.open == 150.0
    assert acc.high == 150.0
    assert acc.low == 150.0
    assert acc.close == 150.0
    assert acc.tick_count == 1


def test_bar_accumulator_updates_ohlcv():
    acc = BarAccumulator(symbol="AAPL", bar_size_minutes=5)
    acc.add_tick(make_tick("AAPL", 150.0, minutes_offset=0))
    acc.add_tick(make_tick("AAPL", 155.0, volume=200, minutes_offset=1))
    acc.add_tick(make_tick("AAPL", 148.0, volume=150, minutes_offset=2))
    acc.add_tick(make_tick("AAPL", 152.0, volume=100, minutes_offset=3))

    assert acc.open == 150.0
    assert acc.high == 155.0
    assert acc.low == 148.0
    assert acc.close == 152.0
    assert acc.volume == 100 + 200 + 150 + 100  # cumulative from all ticks
    assert acc.tick_count == 4


def test_bar_accumulator_completes_bar():
    acc = BarAccumulator(symbol="AAPL", bar_size_minutes=5)
    acc.add_tick(make_tick("AAPL", 150.0, minutes_offset=0))
    acc.add_tick(make_tick("AAPL", 155.0, minutes_offset=2))

    # This tick is at minute 5 → should complete the bar
    completed = acc.add_tick(make_tick("AAPL", 160.0, minutes_offset=5))
    assert completed is not None
    assert completed.open == 150.0
    assert completed.high == 155.0
    assert completed.close == 155.0

    # New bar should have started
    assert acc.open == 160.0
    assert acc.tick_count == 1


def test_bar_accumulator_to_dict():
    acc = BarAccumulator(symbol="AAPL", bar_size_minutes=5)
    acc.add_tick(make_tick("AAPL", 150.0, minutes_offset=0))
    acc.add_tick(make_tick("AAPL", 155.0, minutes_offset=2))

    completed = acc.add_tick(make_tick("AAPL", 160.0, minutes_offset=5))
    bar_dict = completed.to_dict()

    assert "timestamp" in bar_dict
    assert "open" in bar_dict
    assert "high" in bar_dict
    assert "low" in bar_dict
    assert "close" in bar_dict
    assert "volume" in bar_dict


def test_bar_accumulator_aligns_to_boundary():
    acc = BarAccumulator(symbol="AAPL", bar_size_minutes=5)
    # Tick at 10:03 should align bar start to 10:00
    tick = Tick(
        symbol="AAPL", price=150.0, volume=100,
        timestamp=datetime(2026, 1, 1, 10, 3, tzinfo=UTC),
    )
    acc.add_tick(tick)
    assert acc.bar_start == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


# ── MarketDataService tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_market_data_service_get_prices():
    broker = MockBrokerAdapter(initial_cash=5000.0)
    service = MarketDataService(broker)
    service.update_price("AAPL", 150.0)
    service.update_price("MSFT", 380.0)

    prices = await service.get_current_prices(["AAPL", "MSFT", "UNKNOWN"])
    assert prices["AAPL"] == 150.0
    assert prices["MSFT"] == 380.0
    assert prices["UNKNOWN"] == 0.0


@pytest.mark.asyncio
async def test_market_data_service_cache_clear():
    broker = MockBrokerAdapter(initial_cash=5000.0)
    service = MarketDataService(broker)
    service._historical_cache["AAPL_30 D_1 hour"] = pd.DataFrame()
    service._cache_timestamps["AAPL_30 D_1 hour"] = datetime.now(UTC)

    service.clear_cache()
    assert len(service._historical_cache) == 0
    assert len(service._cache_timestamps) == 0


@pytest.mark.asyncio
async def test_market_data_service_invalidate_cache():
    broker = MockBrokerAdapter(initial_cash=5000.0)
    service = MarketDataService(broker)
    service._historical_cache["AAPL_30 D_1 hour"] = pd.DataFrame()
    service._historical_cache["MSFT_30 D_1 hour"] = pd.DataFrame()
    service._cache_timestamps["AAPL_30 D_1 hour"] = datetime.now(UTC)
    service._cache_timestamps["MSFT_30 D_1 hour"] = datetime.now(UTC)

    service._invalidate_cache("AAPL")
    assert "AAPL_30 D_1 hour" not in service._historical_cache
    assert "MSFT_30 D_1 hour" in service._historical_cache
