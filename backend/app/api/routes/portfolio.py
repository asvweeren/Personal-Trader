import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter, OrderType
from app.config import settings
from app.dependencies import get_broker, get_db, get_performance_tracker, get_trading_engine
from app.models.order import OrderStatus
from app.models.portfolio_snapshot import PortfolioSnapshot
from app.monitoring.alerts import send_alert
from app.monitoring.performance import PerformanceTracker

logger = structlog.get_logger()
router = APIRouter()


@router.get("/portfolio")
async def get_portfolio(broker: BrokerAdapter = Depends(get_broker)):
    try:
        portfolio = await broker.get_portfolio()
        return {
            "total_value": portfolio.account_summary.total_value,
            "cash": portfolio.account_summary.cash,
            "buying_power": portfolio.account_summary.buying_power,
            "unrealized_pnl": portfolio.account_summary.unrealized_pnl,
            "realized_pnl": portfolio.account_summary.realized_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in portfolio.positions
            ],
        }
    except Exception:
        logger.warning("portfolio.broker_unavailable")
        return {
            "total_value": settings.initial_capital,
            "cash": settings.initial_capital,
            "buying_power": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "positions": [],
            "broker_connected": False,
        }


@router.get("/positions")
async def get_positions(broker: BrokerAdapter = Depends(get_broker)):
    try:
        positions = await broker.get_positions()
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "market_price": p.market_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ]
    except Exception:
        logger.warning("positions.broker_unavailable")
        return []


@router.get("/performance")
async def get_performance(
    broker: BrokerAdapter = Depends(get_broker),
    tracker: PerformanceTracker = Depends(get_performance_tracker),
):
    data = tracker.to_dict()
    # Override with actual broker values for accuracy
    try:
        portfolio = await broker.get_portfolio()
        actual_value = portfolio.account_summary.total_value
        initial = tracker.initial_capital
        data["total_value"] = round(actual_value, 2)
        data["unrealized_pnl"] = round(
            portfolio.account_summary.unrealized_pnl, 2
        )
        # Keep lifetime realized_pnl from tracker (IBKR resets daily)
        # data["realized_pnl"] stays from tracker.to_dict()
        if initial > 0:
            data["total_return_pct"] = round(
                (actual_value - initial) / initial * 100, 2
            )
            data["daily_pnl"] = round(
                actual_value - tracker.daily_start_value, 2
            )
            data["daily_return_pct"] = round(
                (actual_value - tracker.daily_start_value)
                / tracker.daily_start_value * 100, 2
            ) if tracker.daily_start_value > 0 else 0.0
        # Update drawdown from actual value
        if actual_value > tracker.peak_value:
            tracker.peak_value = actual_value
        dd = (
            (tracker.peak_value - actual_value)
            / tracker.peak_value * 100
            if tracker.peak_value > 0 else 0.0
        )
        data["max_drawdown"] = round(max(dd, tracker.max_drawdown), 2)
    except Exception:
        logger.warning("performance.broker_unavailable_for_metrics")
    return data


async def _flush_pending_orders(engine) -> None:
    """Poll pending orders and process fills before taking action."""
    if engine._order_manager.pending_count > 0:
        fills = await engine._order_manager.poll_pending_orders()
        for fill in fills:
            symbol = fill["symbol"]
            if fill["side"] == "SELL" and symbol in engine._open_trades:
                from datetime import UTC, datetime

                trade = engine._open_trades[symbol]
                await engine._portfolio_tracker.record_trade_close(
                    trade, fill["filled_price"]
                )
                engine._open_trades.pop(symbol, None)
                engine._last_close_time[symbol] = datetime.now(UTC)
                logger.info(
                    "engine.position_closed_fill_processed",
                    symbol=symbol,
                    exit_price=fill["filled_price"],
                )


async def _get_broker_position_qty(engine, symbol: str) -> int:
    """Get the actual broker position quantity for a symbol."""
    try:
        portfolio = await engine._broker.get_portfolio()
        for p in portfolio.positions:
            if p.symbol == symbol:
                return int(p.quantity)
    except Exception:
        pass
    return 0


@router.post("/positions/{symbol}/close")
async def close_position(symbol: str):
    """Manually close an open position by placing a MARKET SELL order."""
    try:
        engine = get_trading_engine()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    # Flush any pending order fills first
    await _flush_pending_orders(engine)

    open_trade = engine._open_trades.get(symbol)

    # Check broker for orphaned position (not tracked by engine)
    broker_qty = await _get_broker_position_qty(engine, symbol)

    if not open_trade and broker_qty <= 0:
        raise HTTPException(status_code=404, detail=f"No open trade or broker position for {symbol}")

    if open_trade:
        # Cancel any pending SELL orders (e.g. stop-loss) to avoid accidental short
        await engine._cancel_pending_sells(symbol)
        # Force-clear stale pending tracking for this symbol
        engine._order_manager._pending_by_symbol.pop(symbol, None)

    sell_qty = broker_qty if broker_qty > 0 else (open_trade.quantity if open_trade else 0)

    # Get current price for the sell order
    try:
        portfolio = await engine._broker.get_portfolio()
        current_price = 0.0
        for p in portfolio.positions:
            if p.symbol == symbol:
                current_price = p.market_price
                break
        if current_price <= 0:
            current_price = (open_trade.entry_price if open_trade else 0.0) or 0.0
    except Exception:
        current_price = (open_trade.entry_price if open_trade else 0.0) or 0.0

    # For orphaned positions (no engine trade), sell directly via broker
    if not open_trade:
        from app.broker.base import OrderRequest, OrderSide
        order_req = OrderRequest(
            symbol=symbol, side=OrderSide.SELL, quantity=sell_qty, order_type=OrderType.MARKET,
        )
        result = await engine._broker.place_order(order_req)
        return {
            "status": "submitted",
            "symbol": symbol,
            "quantity": sell_qty,
            "order_id": result.order_id,
            "message": f"Orphaned position: SELL {sell_qty} {symbol} submitted",
        }

    # Place MARKET SELL order
    result = await engine._order_manager.submit_order(
        trade_id=open_trade.id,
        symbol=symbol,
        side="SELL",
        quantity=sell_qty,
        order_type=OrderType.MARKET,
        expected_price=current_price,
    )

    mapped_status = engine._order_manager._map_status(result.status)
    if mapped_status == OrderStatus.FILLED and result.filled_price:
        from datetime import UTC, datetime

        await engine._portfolio_tracker.record_trade_close(
            open_trade, result.filled_price
        )
        engine._open_trades.pop(symbol, None)
        engine._last_close_time[symbol] = datetime.now(UTC)
        pnl = open_trade.realized_pnl or 0.0
        logger.info(
            "engine.position_closed_manual",
            symbol=symbol,
            exit_price=result.filled_price,
            pnl=pnl,
        )
        await send_alert(
            "Position Closed (Manual)",
            f"{symbol}: SOLD {sell_qty} @ {result.filled_price:.2f}\n"
            f"P&L: {pnl:+.2f}\nExit reason: manual close",
        )
        return {
            "status": "closed",
            "symbol": symbol,
            "quantity": sell_qty,
            "exit_price": result.filled_price,
            "pnl": pnl,
        }

    return {
        "status": "submitted",
        "symbol": symbol,
        "order_status": result.status,
        "message": "Sell order submitted, awaiting fill",
    }


@router.post("/positions/close-all")
async def close_all_positions():
    """Close all open positions."""
    try:
        engine = get_trading_engine()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    # Flush any pending order fills first
    await _flush_pending_orders(engine)

    if not engine._open_trades:
        return {"status": "no_positions", "closed": []}

    results = []
    symbols = list(engine._open_trades.keys())
    for symbol in symbols:
        try:
            open_trade = engine._open_trades.get(symbol)
            if not open_trade:
                continue

            await engine._cancel_pending_sells(symbol)
            # Force-clear stale pending tracking for this symbol
            engine._order_manager._pending_by_symbol.pop(symbol, None)

            # Use actual broker quantity
            broker_qty = await _get_broker_position_qty(engine, symbol)
            sell_qty = broker_qty if broker_qty > 0 else open_trade.quantity

            try:
                portfolio = await engine._broker.get_portfolio()
                current_price = 0.0
                for p in portfolio.positions:
                    if p.symbol == symbol:
                        current_price = p.market_price
                        break
                if current_price <= 0:
                    current_price = open_trade.entry_price or 0.0
            except Exception:
                current_price = open_trade.entry_price or 0.0

            result = await engine._order_manager.submit_order(
                trade_id=open_trade.id,
                symbol=symbol,
                side="SELL",
                quantity=sell_qty,
                order_type=OrderType.MARKET,
                expected_price=current_price,
            )

            mapped_status = engine._order_manager._map_status(result.status)
            if mapped_status == OrderStatus.FILLED and result.filled_price:
                from datetime import UTC, datetime

                await engine._portfolio_tracker.record_trade_close(
                    open_trade, result.filled_price
                )
                engine._open_trades.pop(symbol, None)
                engine._last_close_time[symbol] = datetime.now(UTC)
                pnl = open_trade.realized_pnl or 0.0
                results.append({
                    "symbol": symbol,
                    "status": "closed",
                    "quantity": sell_qty,
                    "exit_price": result.filled_price,
                    "pnl": pnl,
                })
            else:
                results.append({
                    "symbol": symbol,
                    "status": "submitted",
                    "order_status": result.status,
                })
        except Exception as e:
            results.append({"symbol": symbol, "status": "error", "reason": str(e)})

    closed_count = sum(1 for r in results if r["status"] == "closed")
    if closed_count > 0:
        summary = "\n".join(
            f"  {r['symbol']}: P&L {r.get('pnl', 0):+.2f}"
            for r in results if r["status"] == "closed"
        )
        await send_alert(
            "Positions Closed (Manual)",
            f"Closed {closed_count}/{len(symbols)} positions:\n{summary}",
        )

    return {"status": "done", "closed": results}


@router.get("/portfolio/snapshots")
async def get_snapshots(
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PortfolioSnapshot)
        .order_by(desc(PortfolioSnapshot.timestamp))
        .limit(limit)
    )
    snapshots = result.scalars().all()

    return [
        {
            "id": s.id,
            "total_value": s.total_value,
            "cash": s.cash,
            "positions_value": s.positions_value,
            "unrealized_pnl": s.unrealized_pnl,
            "realized_pnl": s.realized_pnl,
            "daily_pnl": s.daily_pnl,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
        }
        for s in reversed(snapshots)  # Chronological order
    ]
