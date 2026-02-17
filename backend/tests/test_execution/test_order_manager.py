from unittest.mock import AsyncMock, MagicMock

import pytest

from app.broker.base import OrderResult, OrderType
from app.execution.order_manager import OrderManager
from app.models.order import OrderStatus


def make_mock_broker():
    broker = MagicMock()
    broker.place_order = AsyncMock()
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_order_status = AsyncMock()
    return broker


def make_mock_db():
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    return db


# ── Submit order tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_market_order_filled():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(
        order_id="ord-1", status="FILLED", filled_price=150.0, filled_quantity=10
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    result = await om.submit_order(
        trade_id=1, symbol="AAPL", side="BUY", quantity=10,
    )

    assert result.status == "FILLED"
    assert result.filled_price == 150.0
    assert om.pending_count == 0
    db.add.assert_called_once()
    db.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_order_submitted_goes_pending():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(
        order_id="ord-2", status="SUBMITTED"
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    result = await om.submit_order(
        trade_id=1, symbol="AAPL", side="BUY", quantity=10,
        order_type=OrderType.LIMIT, limit_price=145.0,
    )

    assert result.status == "SUBMITTED"
    assert om.pending_count == 1


@pytest.mark.asyncio
async def test_submit_order_rejected():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(
        order_id="ord-3", status="REJECTED", message="Insufficient funds"
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    result = await om.submit_order(
        trade_id=1, symbol="AAPL", side="BUY", quantity=100,
    )

    assert result.status == "REJECTED"
    assert om.pending_count == 0


# ── Poll pending orders tests ───────────────────────────────


@pytest.mark.asyncio
async def test_poll_filled_order():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(order_id="ord-4", status="SUBMITTED")
    broker.get_order_status.return_value = OrderResult(
        order_id="ord-4", status="FILLED", filled_price=151.0, filled_quantity=10
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    assert om.pending_count == 1

    filled = await om.poll_pending_orders()
    assert len(filled) == 1
    assert filled[0]["filled_price"] == 151.0
    assert filled[0]["trade_id"] == 1
    assert filled[0]["symbol"] == "AAPL"
    assert om.pending_count == 0


@pytest.mark.asyncio
async def test_poll_partial_fill():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(order_id="ord-5", status="SUBMITTED")
    broker.get_order_status.return_value = OrderResult(
        order_id="ord-5", status="PARTIALLY_FILLED", filled_price=150.0, filled_quantity=5
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    filled = await om.poll_pending_orders()

    assert len(filled) == 0  # Not fully filled yet
    assert om.pending_count == 1  # Still pending


@pytest.mark.asyncio
async def test_poll_cancelled_order():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(order_id="ord-6", status="SUBMITTED")
    broker.get_order_status.return_value = OrderResult(
        order_id="ord-6", status="CANCELLED", message="Timeout"
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    filled = await om.poll_pending_orders()

    assert len(filled) == 0
    assert om.pending_count == 0  # Removed from pending


@pytest.mark.asyncio
async def test_poll_no_pending():
    broker = make_mock_broker()
    db = make_mock_db()
    om = OrderManager(broker, db)

    filled = await om.poll_pending_orders()
    assert filled == []


# ── Cancel order tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_order():
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(order_id="ord-7", status="SUBMITTED")
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    assert om.pending_count == 1

    success = await om.cancel_order("ord-7")
    assert success is True
    assert om.pending_count == 0


@pytest.mark.asyncio
async def test_cancel_all_pending():
    broker = make_mock_broker()
    broker.place_order.side_effect = [
        OrderResult(order_id="ord-8", status="SUBMITTED"),
        OrderResult(order_id="ord-9", status="SUBMITTED"),
    ]
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    await om.submit_order(trade_id=2, symbol="MSFT", side="BUY", quantity=5)
    assert om.pending_count == 2

    cancelled = await om.cancel_all_pending()
    assert cancelled == 2
    assert om.pending_count == 0


# ── Status mapping tests ────────────────────────────────────


def test_status_mapping():
    broker = make_mock_broker()
    db = make_mock_db()
    om = OrderManager(broker, db)

    assert om._map_status("FILLED") == OrderStatus.FILLED
    assert om._map_status("SUBMITTED") == OrderStatus.SUBMITTED
    assert om._map_status("UNKNOWN") == OrderStatus.ERROR


def test_ibkr_status_mapping():
    """IBKR returns mixed-case statuses like PendingSubmit, Filled, etc."""
    broker = make_mock_broker()
    db = make_mock_db()
    om = OrderManager(broker, db)

    assert om._map_status("PendingSubmit") == OrderStatus.SUBMITTED
    assert om._map_status("PreSubmitted") == OrderStatus.SUBMITTED
    assert om._map_status("Submitted") == OrderStatus.SUBMITTED
    assert om._map_status("Filled") == OrderStatus.FILLED
    assert om._map_status("Cancelled") == OrderStatus.CANCELLED
    assert om._map_status("Inactive") == OrderStatus.CANCELLED
    assert om._map_status("ApiCancelled") == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_submit_ibkr_pending_submit_goes_pending():
    """IBKR returns PendingSubmit for newly placed orders — should track as pending."""
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(
        order_id="ord-10", status="PendingSubmit"
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    result = await om.submit_order(
        trade_id=1, symbol="ASML", side="BUY", quantity=50,
    )

    assert result.status == "PendingSubmit"
    assert om.pending_count == 1  # Should be tracked for polling


# ── Quantity validation tests ─────────────────────────────


@pytest.mark.asyncio
async def test_submit_order_zero_quantity_raises():
    broker = make_mock_broker()
    db = make_mock_db()
    om = OrderManager(broker, db)

    with pytest.raises(ValueError, match="quantity must be positive"):
        await om.submit_order(
            trade_id=1, symbol="AAPL", side="BUY", quantity=0,
        )


@pytest.mark.asyncio
async def test_submit_order_negative_quantity_raises():
    broker = make_mock_broker()
    db = make_mock_db()
    om = OrderManager(broker, db)

    with pytest.raises(ValueError, match="quantity must be positive"):
        await om.submit_order(
            trade_id=1, symbol="AAPL", side="SELL", quantity=-5,
        )


# ── Cancel orders for symbol tests ────────────────────────


@pytest.mark.asyncio
async def test_cancel_orders_for_symbol():
    """Cancel all pending orders for a specific symbol."""
    broker = make_mock_broker()
    broker.place_order.side_effect = [
        OrderResult(order_id="ord-a1", status="SUBMITTED"),
        OrderResult(order_id="ord-a2", status="SUBMITTED"),
        OrderResult(order_id="ord-b1", status="SUBMITTED"),
    ]
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    await om.submit_order(trade_id=1, symbol="AAPL", side="SELL", quantity=10)
    await om.submit_order(trade_id=2, symbol="MSFT", side="BUY", quantity=5)
    assert om.pending_count == 3

    cancelled = await om.cancel_orders_for_symbol("AAPL")
    assert cancelled == 2
    assert om.pending_count == 1  # Only MSFT remains


@pytest.mark.asyncio
async def test_cancel_orders_for_symbol_by_side():
    """Cancel only SELL orders for a symbol, keep BUY."""
    broker = make_mock_broker()
    broker.place_order.side_effect = [
        OrderResult(order_id="ord-c1", status="SUBMITTED"),
        OrderResult(order_id="ord-c2", status="SUBMITTED"),
    ]
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="BUY", quantity=10)
    await om.submit_order(trade_id=1, symbol="AAPL", side="SELL", quantity=10)
    assert om.pending_count == 2

    cancelled = await om.cancel_orders_for_symbol("AAPL", side="SELL")
    assert cancelled == 1
    assert om.pending_count == 1  # BUY order remains


@pytest.mark.asyncio
async def test_cancel_orders_for_symbol_none_pending():
    """No orders for the symbol — returns 0."""
    broker = make_mock_broker()
    db = make_mock_db()
    om = OrderManager(broker, db)

    cancelled = await om.cancel_orders_for_symbol("AAPL")
    assert cancelled == 0


@pytest.mark.asyncio
async def test_pending_by_symbol_cleaned_on_poll():
    """Verify _pending_by_symbol is cleaned when orders complete via polling."""
    broker = make_mock_broker()
    broker.place_order.return_value = OrderResult(order_id="ord-d1", status="SUBMITTED")
    broker.get_order_status.return_value = OrderResult(
        order_id="ord-d1", status="FILLED", filled_price=150.0, filled_quantity=10
    )
    db = make_mock_db()
    om = OrderManager(broker, db)

    await om.submit_order(trade_id=1, symbol="AAPL", side="SELL", quantity=10)
    assert "AAPL" in om._pending_by_symbol

    await om.poll_pending_orders()
    assert om.pending_count == 0
    assert "AAPL" not in om._pending_by_symbol
