"""API endpoints for the daily stock screener."""

from datetime import date, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from app.config import settings
from app.data.screener import StockScreener
from app.dependencies import get_db
from app.models.screening_result import ScreeningResult

router = APIRouter()


@router.get("/screener/latest")
async def get_latest_screening(db: AsyncSession = Depends(get_db)):
    """Return the most recent screening result."""
    result = await db.execute(
        select(ScreeningResult)
        .order_by(ScreeningResult.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return {"screening_date": None, "total_scanned": 0, "candidates": [], "config": None}
    return {
        "id": row.id,
        "screening_date": row.screening_date.isoformat() if row.screening_date else None,
        "total_scanned": row.total_scanned,
        "candidates": row.candidates,
        "config": row.config,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/screener/history")
async def get_screening_history(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Return screening results from the last N days."""
    cutoff = date.today() - timedelta(days=days)
    result = await db.execute(
        select(ScreeningResult)
        .where(ScreeningResult.screening_date >= cutoff)
        .order_by(ScreeningResult.created_at.desc())
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "screening_date": r.screening_date.isoformat() if r.screening_date else None,
            "total_scanned": r.total_scanned,
            "candidates": r.candidates,
            "config": r.config,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.post("/screener/run")
async def run_screener(db: AsyncSession = Depends(get_db)):
    """Manually trigger a screening run."""
    screener = StockScreener()
    data = await screener.run_screening()

    # Persist to DB
    row = ScreeningResult(
        screening_date=date.today(),
        total_scanned=data["total_scanned"],
        candidates=data["candidates"],
        config=data["config"],
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    # Update engine + pipeline symbols if screener is enabled
    if settings.screener_enabled and data["candidates"]:
        symbols = [c["symbol"] for c in data["candidates"]]
        try:
            from app.dependencies import get_trading_engine
            engine = get_trading_engine()
            engine.update_symbols(symbols)
        except RuntimeError:
            pass
        try:
            from app.dependencies import get_data_pipeline
            pipeline = get_data_pipeline()
            await pipeline.update_symbols(symbols)
        except Exception:
            pass

    return {
        "id": row.id,
        "screening_date": row.screening_date.isoformat() if row.screening_date else None,
        "total_scanned": row.total_scanned,
        "candidates": row.candidates,
        "config": row.config,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
