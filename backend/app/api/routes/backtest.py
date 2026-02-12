import asyncio

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.dependencies import get_broker, get_db
from app.models.backtest_result import BacktestResult
from app.strategy.ml_strategy import MLStrategy
from app.strategy.sentiment_strategy import SentimentStrategy

logger = structlog.get_logger()

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
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 2.0
    enable_eod_close: bool = True
    trailing_stop_tiers: str = "1.0:0.5,2.0:0.75,3.0:1.0,5.0:1.5"
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

        # Get historical data - try broker first, fallback to yfinance
        data = None
        broker = get_broker()
        try:
            if await broker.is_connected():
                data = await broker.get_historical_data(
                    symbol=request.symbol,
                    duration="365 D",
                    bar_size="1 hour",
                )
        except Exception:
            logger.info("backtest.broker_unavailable", symbol=request.symbol)

        if data is None or data.empty:
            # Fallback: download from yfinance
            try:
                import yfinance as yf

                ticker = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: yf.download(
                        request.symbol,
                        start=request.start_date,
                        end=request.end_date,
                        interval="1h",
                        progress=False,
                    ),
                )
                if ticker is not None and not ticker.empty:
                    # Flatten MultiIndex columns (yfinance >= 0.2.36 always returns MultiIndex)
                    import pandas as _pd
                    if isinstance(ticker.columns, _pd.MultiIndex):
                        ticker.columns = [
                            str(c[0]) if isinstance(c, tuple) else str(c)
                            for c in ticker.columns
                        ]
                    ticker = ticker.reset_index()
                    rename_map = {}
                    for col in ticker.columns:
                        lc = str(col).strip().lower()
                        if lc in ("datetime", "date", "index"):
                            rename_map[col] = "timestamp"
                        elif lc == "open":
                            rename_map[col] = "open"
                        elif lc == "high":
                            rename_map[col] = "high"
                        elif lc == "low":
                            rename_map[col] = "low"
                        elif lc == "close":
                            rename_map[col] = "close"
                        elif lc == "volume":
                            rename_map[col] = "volume"
                    data = ticker.rename(columns=rename_map)
                    # Drop duplicate columns (yfinance sometimes returns both Date and Datetime)
                    data = data.loc[:, ~data.columns.duplicated()]
                    logger.info("backtest.yfinance_data", symbol=request.symbol, rows=len(data))
            except Exception as e:
                logger.exception("backtest.yfinance_error", symbol=request.symbol)
                await _update_result(db, backtest_id, error=f"Failed to download data: {e}")
                return

        if data is None or data.empty or len(data) < 50:
            await _update_result(db, backtest_id, error="Insufficient historical data. Check symbol and date range.")
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
            take_profit_pct=request.take_profit_pct,
            enable_eod_close=request.enable_eod_close,
            trailing_stop_tiers=request.trailing_stop_tiers,
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
            "take_profit_pct": request.take_profit_pct,
            "enable_eod_close": request.enable_eod_close,
            "trailing_stop_tiers": request.trailing_stop_tiers,
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


class WalkForwardRequest(BaseModel):
    strategy_name: str = "ml_xgboost"
    symbol: str = "SPY"
    start_date: str = ""
    end_date: str = ""
    train_days: int = 180
    test_days: int = 30
    step_days: int = 30
    initial_capital: float = 5000.0
    params: dict | None = None


class MonteCarloRequest(BaseModel):
    backtest_id: int
    num_simulations: int = 1000


@router.post("/backtest/walk-forward")
async def run_walk_forward(
    request: WalkForwardRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Run walk-forward validation analysis."""
    from app.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine

    factory = _STRATEGY_REGISTRY.get(request.strategy_name)
    if not factory:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy. Available: {list(_STRATEGY_REGISTRY.keys())}",
        )

    # Create pending result
    bt = BacktestResult(
        strategy_name=request.strategy_name,
        params={
            "type": "walk_forward",
            "symbol": request.symbol,
            "train_days": request.train_days,
            "test_days": request.test_days,
            "step_days": request.step_days,
        },
        metrics={"status": "running"},
    )
    db.add(bt)
    await db.commit()
    await db.refresh(bt)

    async def _run_wf(bt_id: int, req: WalkForwardRequest, session: AsyncSession):
        try:
            strategy = factory(req.params or {})
            # Download data
            data = None
            try:
                import yfinance as yf
                data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: yf.download(
                        req.symbol,
                        start=req.start_date or None,
                        end=req.end_date or None,
                        period="2y" if not req.start_date else None,
                        interval="1h",
                        progress=False,
                    ),
                )
                if data is not None and not data.empty:
                    import pandas as _pd
                    if isinstance(data.columns, _pd.MultiIndex):
                        data.columns = [
                            str(c[0]) if isinstance(c, tuple) else str(c)
                            for c in data.columns
                        ]
                    data = data.reset_index()
                    rename_map = {}
                    for col in data.columns:
                        lc = str(col).strip().lower()
                        if lc in ("datetime", "date", "index"):
                            rename_map[col] = "timestamp"
                        elif lc == "open":
                            rename_map[col] = "open"
                        elif lc == "high":
                            rename_map[col] = "high"
                        elif lc == "low":
                            rename_map[col] = "low"
                        elif lc == "close":
                            rename_map[col] = "close"
                        elif lc == "volume":
                            rename_map[col] = "volume"
                    data = data.rename(columns=rename_map)
                    data = data.loc[:, ~data.columns.duplicated()]
            except Exception as e:
                await _update_result(session, bt_id, error=f"Data download error: {e}")
                return

            if data is None or data.empty or len(data) < 100:
                await _update_result(session, bt_id, error="Insufficient data")
                return

            config = WalkForwardConfig(
                train_days=req.train_days,
                test_days=req.test_days,
                step_days=req.step_days,
                initial_capital=req.initial_capital,
            )
            engine = WalkForwardEngine()
            result = await engine.run(config, strategy, data, req.symbol)

            bt_row = await session.get(BacktestResult, bt_id)
            if bt_row:
                bt_row.metrics = result.to_dict()
                await session.commit()

        except Exception as e:
            await _update_result(session, bt_id, error=str(e))

    background_tasks.add_task(_run_wf, bt.id, request, db)
    return {"id": bt.id, "status": "running", "type": "walk_forward"}


@router.post("/backtest/monte-carlo")
async def run_monte_carlo(
    request: MonteCarloRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run Monte Carlo simulation on an existing backtest's trade results."""
    from app.backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig

    bt = await db.get(BacktestResult, request.backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    if not bt.trades_summary:
        raise HTTPException(status_code=400, detail="Backtest has no trades")

    # Extract trade P&Ls
    trade_pnls = [t.get("pnl", 0.0) for t in bt.trades_summary if "pnl" in t]
    if not trade_pnls:
        raise HTTPException(status_code=400, detail="No trade P&Ls found")

    initial_capital = bt.params.get("initial_capital", 5000.0) if bt.params else 5000.0
    config = MonteCarloConfig(num_simulations=request.num_simulations)

    simulator = MonteCarloSimulator()
    result = simulator.run(trade_pnls, initial_capital, config)

    return {
        "backtest_id": request.backtest_id,
        "result": result.to_dict(),
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
