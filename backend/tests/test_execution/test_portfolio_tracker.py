from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.broker.base import AccountSummary, Portfolio, Position
from app.execution.portfolio_tracker import PortfolioTracker
from app.models.trade import Trade, TradeSide, TradeStatus
from app.monitoring.performance import PerformanceTracker


def make_portfolio(total=5000.0, cash=3000.0, positions=None):
    positions = positions or []
    unrealized = sum(p.unrealized_pnl for p in positions)
    realized = 0.0
    return Portfolio(
        account_summary=AccountSummary(
            total_value=total,
            cash=cash,
            buying_power=cash,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
        ),
        positions=positions,
    )


def make_mock_broker(portfolio=None):
    broker = MagicMock()
    broker.get_portfolio = AsyncMock(return_value=portfolio or make_portfolio())
    return broker


def make_mock_db():
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


# ── Get current tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_get_current():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    portfolio = await tracker.get_current()
    assert portfolio.account_summary.total_value == 5000.0
    assert tracker._last_portfolio is not None


@pytest.mark.asyncio
async def test_get_current_updates_performance():
    broker = make_mock_broker(make_portfolio(total=5100.0, cash=3000.0))
    # Fix the portfolio to have correct unrealized
    portfolio = make_portfolio()
    portfolio.account_summary.unrealized_pnl = 100.0
    broker.get_portfolio = AsyncMock(return_value=portfolio)

    db = make_mock_db()
    perf = PerformanceTracker(initial_capital=5000.0)
    tracker = PortfolioTracker(broker, db, perf)

    await tracker.get_current()
    assert perf.unrealized_pnl == 100.0


# ── Initialize daily tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_daily():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    value = await tracker.initialize_daily()
    assert value == 5000.0
    assert tracker._daily_start_value == 5000.0


@pytest.mark.asyncio
async def test_initialize_daily_resets_performance():
    broker = make_mock_broker()
    db = make_mock_db()
    perf = PerformanceTracker(initial_capital=5000.0)
    perf.daily_pnl = 100.0
    tracker = PortfolioTracker(broker, db, perf)

    await tracker.initialize_daily()
    assert perf.daily_pnl == 0.0


# ── Daily P&L tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_pnl_positive():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    # Initialize at 5000
    await tracker.initialize_daily()

    # Now portfolio is worth 5200
    broker.get_portfolio = AsyncMock(return_value=make_portfolio(total=5200.0))
    await tracker.get_current()

    assert tracker.get_daily_pnl() == 200.0


@pytest.mark.asyncio
async def test_daily_pnl_negative():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    await tracker.initialize_daily()

    broker.get_portfolio = AsyncMock(return_value=make_portfolio(total=4800.0))
    await tracker.get_current()

    assert tracker.get_daily_pnl() == -200.0


def test_daily_pnl_no_init():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)
    assert tracker.get_daily_pnl() == 0.0


# ── Record trade close tests ────────────────────────────────


@pytest.mark.asyncio
async def test_record_trade_close_buy():
    broker = make_mock_broker()
    db = make_mock_db()
    perf = PerformanceTracker(initial_capital=5000.0)
    tracker = PortfolioTracker(broker, db, perf)

    trade = MagicMock(spec=Trade)
    trade.id = 1
    trade.symbol = "AAPL"
    trade.side = TradeSide.BUY
    trade.quantity = 10
    trade.entry_price = 150.0
    trade.strategy_name = "ml_xgboost"
    trade.created_at = datetime.now(timezone.utc)

    pnl = await tracker.record_trade_close(trade, exit_price=160.0)

    assert pnl == 100.0  # (160 - 150) * 10
    assert trade.exit_price == 160.0
    assert trade.realized_pnl == 100.0
    assert trade.status == TradeStatus.CLOSED
    assert perf.total_trades == 1
    assert perf.winning_trades == 1


@pytest.mark.asyncio
async def test_record_trade_close_with_commission():
    broker = make_mock_broker()
    db = make_mock_db()
    perf = PerformanceTracker(initial_capital=5000.0)
    tracker = PortfolioTracker(broker, db, perf)

    trade = MagicMock(spec=Trade)
    trade.id = 2
    trade.symbol = "AAPL"
    trade.side = TradeSide.BUY
    trade.quantity = 10
    trade.entry_price = 150.0
    trade.strategy_name = "ml_xgboost"
    trade.created_at = datetime.now(timezone.utc)

    pnl = await tracker.record_trade_close(trade, exit_price=160.0, commission=5.0)

    assert pnl == 95.0  # (160 - 150) * 10 - 5


@pytest.mark.asyncio
async def test_record_trade_close_no_entry_price():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    trade = MagicMock(spec=Trade)
    trade.id = 3
    trade.entry_price = None

    pnl = await tracker.record_trade_close(trade, exit_price=160.0)
    assert pnl == 0.0


# ── Snapshot tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_take_snapshot():
    positions = [
        Position("AAPL", 10, 150.0, 155.0, 1550.0, 50.0),
    ]
    broker = make_mock_broker(make_portfolio(total=5050.0, cash=3500.0, positions=positions))
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    snapshot = await tracker.take_snapshot()

    assert snapshot.total_value == 5050.0
    assert snapshot.cash == 3500.0
    assert snapshot.positions_value == 1550.0
    db.add.assert_called_once()
    db.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_take_snapshot_with_daily_pnl():
    broker = make_mock_broker(make_portfolio(total=5000.0))
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)

    await tracker.initialize_daily()

    broker.get_portfolio = AsyncMock(return_value=make_portfolio(total=5100.0))
    snapshot = await tracker.take_snapshot()

    assert snapshot.daily_pnl == 100.0


# ── Status tests ────────────────────────────────────────────


def test_get_status_not_initialized():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)
    status = tracker.get_status()
    assert status["initialized"] is False


@pytest.mark.asyncio
async def test_get_status_initialized():
    broker = make_mock_broker()
    db = make_mock_db()
    tracker = PortfolioTracker(broker, db)
    await tracker.get_current()

    status = tracker.get_status()
    assert status["initialized"] is True
    assert status["total_value"] == 5000.0
