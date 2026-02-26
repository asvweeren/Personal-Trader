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
