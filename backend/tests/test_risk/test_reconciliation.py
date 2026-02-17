"""Tests for position reconciliation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from app.broker.base import Portfolio, AccountSummary, Position
from app.models.trade import Trade, TradeSide, TradeStatus
from app.risk.reconciliation import reconcile, auto_fix, ReconciliationResult


def _make_trade(symbol: str, qty: int) -> Trade:
    trade = Trade(
        symbol=symbol,
        side=TradeSide.BUY,
        quantity=qty,
        status=TradeStatus.OPEN,
        strategy_name="test",
        signal_id=1,
    )
    trade.id = 1
    trade.entry_price = 100.0
    trade.created_at = datetime.now(timezone.utc)
    return trade


def _make_portfolio(positions: list[tuple[str, int]]) -> Portfolio:
    return Portfolio(
        account_summary=AccountSummary(
            total_value=10000.0, cash=5000.0, buying_power=5000.0,
            unrealized_pnl=0.0, realized_pnl=0.0,
        ),
        positions=[
            Position(symbol=s, quantity=q, avg_cost=100.0, market_price=100.0,
                     market_value=100.0 * q, unrealized_pnl=0.0)
            for s, q in positions
        ],
    )


@pytest.mark.asyncio
async def test_reconcile_all_match():
    trades = {"AAPL": _make_trade("AAPL", 10), "MSFT": _make_trade("MSFT", 5)}
    portfolio = _make_portfolio([("AAPL", 10), ("MSFT", 5)])
    result = await reconcile(trades, portfolio)
    assert result.is_clean
    assert set(result.matches) == {"AAPL", "MSFT"}
    assert len(result.mismatches) == 0


@pytest.mark.asyncio
async def test_reconcile_quantity_mismatch():
    trades = {"AAPL": _make_trade("AAPL", 10)}
    portfolio = _make_portfolio([("AAPL", 15)])
    result = await reconcile(trades, portfolio)
    assert not result.is_clean
    assert len(result.mismatches) == 1
    assert result.mismatches[0]["symbol"] == "AAPL"
    assert result.mismatches[0]["internal_qty"] == 10
    assert result.mismatches[0]["broker_qty"] == 15


@pytest.mark.asyncio
async def test_reconcile_orphaned_broker():
    trades = {}
    portfolio = _make_portfolio([("AAPL", 10)])
    result = await reconcile(trades, portfolio)
    assert not result.is_clean
    assert len(result.orphaned_broker) == 1
    assert result.orphaned_broker[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_reconcile_orphaned_internal():
    trades = {"AAPL": _make_trade("AAPL", 10)}
    portfolio = _make_portfolio([])
    result = await reconcile(trades, portfolio)
    assert not result.is_clean
    assert "AAPL" in result.orphaned_internal


@pytest.mark.asyncio
async def test_reconcile_empty():
    result = await reconcile({}, _make_portfolio([]))
    assert result.is_clean
    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_auto_fix_quantity():
    trade = _make_trade("AAPL", 10)
    engine = MagicMock()
    engine._open_trades = {"AAPL": trade}
    db = AsyncMock()

    result = ReconciliationResult(
        mismatches=[{
            "symbol": "AAPL",
            "internal_qty": 10,
            "broker_qty": 15,
            "action": "sync_to_broker",
        }],
    )
    actions = await auto_fix(result, engine, db)
    assert len(actions) == 1
    assert trade.quantity == 15


@pytest.mark.asyncio
async def test_auto_fix_orphaned_internal():
    trade = _make_trade("AAPL", 10)
    engine = MagicMock()
    engine._open_trades = {"AAPL": trade}
    db = AsyncMock()

    result = ReconciliationResult(orphaned_internal=["AAPL"])
    actions = await auto_fix(result, engine, db)
    assert len(actions) == 1
    assert trade.status == TradeStatus.CLOSED
    assert "AAPL" not in engine._open_trades


@pytest.mark.asyncio
async def test_reconcile_detects_short_position():
    """A negative broker quantity should be detected as a short to close."""
    trades = {}
    portfolio = _make_portfolio([])
    # Manually add a short position (negative qty)
    portfolio.positions.append(
        Position(
            symbol="APH",
            quantity=-50,
            avg_cost=100.0,
            market_price=100.0,
            market_value=-5000.0,
            unrealized_pnl=0.0,
        )
    )
    result = await reconcile(trades, portfolio)
    assert not result.is_clean
    assert len(result.orphaned_broker) == 1
    assert result.orphaned_broker[0]["symbol"] == "APH"
    assert result.orphaned_broker[0]["broker_qty"] == -50
    assert result.orphaned_broker[0]["action"] == "close_short"
    assert result.orphaned_broker[0]["severity"] == "critical"


@pytest.mark.asyncio
async def test_auto_fix_closes_short():
    """Auto-fix should place a BUY order to flatten a short position."""
    engine = MagicMock()
    engine._open_trades = {}
    order_mgr = MagicMock()
    order_mgr.submit_order = AsyncMock()
    engine._order_manager = order_mgr
    db = AsyncMock()

    result = ReconciliationResult(
        orphaned_broker=[{
            "symbol": "APH",
            "broker_qty": -50,
            "action": "close_short",
            "severity": "critical",
        }],
    )
    actions = await auto_fix(result, engine, db)
    assert len(actions) == 1
    assert "CRITICAL" in actions[0]
    assert "APH" in actions[0]
    assert "50" in actions[0]
    order_mgr.submit_order.assert_awaited_once()
    call_kwargs = order_mgr.submit_order.call_args
    assert call_kwargs.kwargs["symbol"] == "APH"
    assert call_kwargs.kwargs["side"] == "BUY"
    assert call_kwargs.kwargs["quantity"] == 50


def test_result_to_dict():
    result = ReconciliationResult(
        matches=["AAPL"],
        mismatches=[{"symbol": "MSFT", "internal_qty": 10, "broker_qty": 5}],
    )
    d = result.to_dict()
    assert d["match_count"] == 1
    assert d["mismatch_count"] == 1
    assert not d["is_clean"]
