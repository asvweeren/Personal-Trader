import pytest

from app.broker.base import OrderRequest, OrderSide, OrderType
from app.broker.mock_adapter import MockBrokerAdapter


@pytest.fixture
def broker():
    b = MockBrokerAdapter(initial_cash=5000.0)
    b.set_price("AAPL", 150.0)
    return b


async def test_connect(broker):
    await broker.connect()
    assert await broker.is_connected()
    await broker.disconnect()
    assert not await broker.is_connected()


async def test_place_buy_order(broker):
    await broker.connect()
    order = OrderRequest(
        symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
    )
    result = await broker.place_order(order)
    assert result.status == "FILLED"
    assert result.filled_price == 150.0
    assert result.filled_quantity == 10

    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == 10


async def test_place_sell_order(broker):
    await broker.connect()
    # First buy
    buy = OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=10)
    await broker.place_order(buy)

    # Then sell
    sell = OrderRequest(symbol="AAPL", side=OrderSide.SELL, quantity=5)
    result = await broker.place_order(sell)
    assert result.status == "FILLED"

    positions = await broker.get_positions()
    assert positions[0].quantity == 5


async def test_insufficient_funds(broker):
    await broker.connect()
    order = OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=1000)
    result = await broker.place_order(order)
    assert result.status == "REJECTED"


async def test_portfolio(broker):
    await broker.connect()
    portfolio = await broker.get_portfolio()
    assert portfolio.account_summary.total_value == 5000.0
    assert portfolio.account_summary.cash == 5000.0
    assert len(portfolio.positions) == 0

    # Buy some shares
    order = OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=10)
    await broker.place_order(order)

    portfolio = await broker.get_portfolio()
    assert portfolio.account_summary.cash == 3500.0  # 5000 - 10*150
    assert len(portfolio.positions) == 1


async def test_historical_data(broker):
    await broker.connect()
    df = await broker.get_historical_data("AAPL")
    assert not df.empty
    assert "close" in df.columns
    assert "volume" in df.columns
    assert len(df) == 100


# ── Tick-size rounding ────────────────────────────────────────

from app.broker.base import round_to_tick


def test_round_to_tick_us_stock():
    """US stocks: 0.01 tick → 2 decimal places."""
    assert round_to_tick(145.576, 0.01) == 145.58
    assert round_to_tick(145.574, 0.01) == 145.57
    assert round_to_tick(145.53, 0.01) == 145.53
    assert round_to_tick(100.0, 0.01) == 100.0


def test_round_to_tick_eu_stock():
    """Some EU stocks use 0.05 tick increments."""
    assert round_to_tick(81.77, 0.05) == 81.75
    assert round_to_tick(81.78, 0.05) == 81.80
    assert round_to_tick(81.725, 0.05) == 81.70


def test_round_to_tick_uk_pence():
    """UK stocks in GBX may have 0.50 or 1.0 tick."""
    assert round_to_tick(14818.69, 0.50) == 14818.50
    assert round_to_tick(14818.80, 0.50) == 14819.00
    assert round_to_tick(14818.24, 0.50) == 14818.00
    assert round_to_tick(101.39, 1.0) == 101.0
    assert round_to_tick(101.60, 1.0) == 102.0


def test_round_to_tick_zero_fallback():
    """Zero or negative tick falls back to 2 decimal places."""
    assert round_to_tick(145.576, 0.0) == 145.58
    assert round_to_tick(145.574, -1.0) == 145.57
