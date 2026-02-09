from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.dependencies import get_broker, get_db
from app.models.backtest_result import BacktestResult
from app.strategy.ml_strategy import MLStrategy
from app.strategy.sentiment_strategy import SentimentStrategy

router = APIRouter()

# Map of available strategies for backtesting
_STRATEGY_REGISTRY = {
    "ml_xgboost": lambda params: MLStrategy(
        model_path=params.get("model_path", ""),
        confidence_threshold=params.get("confidence_threshold", 0.6),
    ),
    "sentiment": lambda params: SentimentStrategy(
        buy_threshold=params.get("buy_threshold", 0.3),
        sell_threshold=params.get("sell_threshold", -0.3),
    ),
}


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 5000.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    max_position_pct: float = 20.0
    stop_loss_pct: float = 3.0
    params: dict | None = None


async def _run_backtest_task(
    backtest_id: int,
    request: BacktestRequest,
    db: AsyncSession,
):
    """Background task to run the actual backtest."""
    from app.dependencies import get_broker

    try:
        # Get strategy
        factory = _STRATEGY_REGISTRY.get(request.strategy_name)
        if not factory:
            await _update_result(db, backtest_id, error=f"Unknown strategy: {request.strategy_name}")
            return

        strategy = factory(request.params or {})

        # Get historical data from broker
        broker = get_broker()
        if not await broker.is_connected():
            await broker.connect()

        data = await broker.get_historical_data(
            symbol=request.symbol,
            duration="365 D",
            bar_size="1 hour",
        )

        if data.empty or len(data) < 50:
            await _update_result(db, backtest_id, error="Insufficient historical data")
            return

        # Run backtest
        config = BacktestConfig(
            strategy=strategy,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            commission_pct=request.commission_pct,
            slippage_pct=request.slippage_pct,
            max_position_pct=request.max_position_pct,
            stop_loss_pct=request.stop_loss_pct,
        )

        engine = BacktestEngine()
        result = await engine.run(config, data)

        # Persist results
        trades_summary = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "commission": t.commission,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
            for t in result.trades
        ]

        bt = await db.get(BacktestResult, backtest_id)
        if bt:
            bt.metrics = {**result.metrics, "benchmark": result.benchmark_metrics}
            bt.trades_summary = trades_summary
            bt.equity_curve = result.equity_curve
            await db.commit()

    except Exception as e:
        await _update_result(db, backtest_id, error=str(e))


async def _update_result(db: AsyncSession, backtest_id: int, error: str):
    bt = await db.get(BacktestResult, backtest_id)
    if bt:
        bt.metrics = {"status": "error", "error": error}
        await db.commit()


@router.post("/backtest/run")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if request.strategy_name not in _STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy. Available: {list(_STRATEGY_REGISTRY.keys())}",
        )

    # Create pending result
    bt = BacktestResult(
        strategy_name=request.strategy_name,
        params={
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "commission_pct": request.commission_pct,
            "slippage_pct": request.slippage_pct,
            "max_position_pct": request.max_position_pct,
            "stop_loss_pct": request.stop_loss_pct,
            **(request.params or {}),
        },
        metrics={"status": "running"},
    )
    db.add(bt)
    await db.commit()
    await db.refresh(bt)

    # Schedule background task
    background_tasks.add_task(_run_backtest_task, bt.id, request, db)

    return {"id": bt.id, "status": "running"}


@router.get("/backtest/{backtest_id}")
async def get_backtest(backtest_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BacktestResult).where(BacktestResult.id == backtest_id)
    )
    bt = result.scalar_one_or_none()
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return {
        "id": bt.id,
        "strategy_name": bt.strategy_name,
        "params": bt.params,
        "metrics": bt.metrics,
        "trades_summary": bt.trades_summary,
        "equity_curve": bt.equity_curve,
        "created_at": bt.created_at.isoformat() if bt.created_at else None,
    }


@router.get("/backtests")
async def list_backtests(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    strategy_name: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(BacktestResult).order_by(desc(BacktestResult.created_at))
    if strategy_name:
        query = query.where(BacktestResult.strategy_name == strategy_name)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    backtests = result.scalars().all()

    count_query = select(func.count(BacktestResult.id))
    if strategy_name:
        count_query = count_query.where(BacktestResult.strategy_name == strategy_name)
    total = (await db.execute(count_query)).scalar()

    return {
        "backtests": [
            {
                "id": bt.id,
                "strategy_name": bt.strategy_name,
                "params": bt.params,
                "metrics": bt.metrics,
                "created_at": bt.created_at.isoformat() if bt.created_at else None,
            }
            for bt in backtests
        ],
        "total": total,
    }
