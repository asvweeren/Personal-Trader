from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.trade import Trade

router = APIRouter()


@router.get("/trades")
async def get_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    symbol: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Trade).order_by(desc(Trade.created_at))
    if symbol:
        query = query.where(Trade.symbol == symbol)
    if status:
        query = query.where(Trade.status == status)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    trades = result.scalars().all()

    count_query = select(func.count(Trade.id))
    if symbol:
        count_query = count_query.where(Trade.symbol == symbol)
    if status:
        count_query = count_query.where(Trade.status == status)
    count_result = await db.execute(count_query)
    total = count_result.scalar()

    return {
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "status": t.status.value,
                "strategy_name": t.strategy_name,
                "realized_pnl": t.realized_pnl,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            }
            for t in trades
        ],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/trades/{trade_id}")
async def get_trade(trade_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Trade).where(Trade.id == trade_id))
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "side": trade.side.value,
        "quantity": trade.quantity,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "stop_loss": trade.stop_loss,
        "take_profit": trade.take_profit,
        "status": trade.status.value,
        "strategy_name": trade.strategy_name,
        "signal_id": trade.signal_id,
        "realized_pnl": trade.realized_pnl,
        "commission": trade.commission,
        "created_at": trade.created_at.isoformat() if trade.created_at else None,
        "updated_at": trade.updated_at.isoformat() if trade.updated_at else None,
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
    }
