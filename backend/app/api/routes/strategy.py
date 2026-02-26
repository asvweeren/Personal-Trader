import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.strategy_config import StrategyConfig
from app.models.trade import Trade, TradeStatus

logger = structlog.get_logger()

router = APIRouter()

# Default values used when no DB row exists yet
_DEFAULTS = {
    "active_strategies": ["ml_xgboost", "sentiment"],
    "confidence_threshold": 0.6,
    "ensemble_method": "weighted_average",
    "weights": {"ml_xgboost": 0.5, "sentiment": 0.3, "nn_lstm": 0.2},
    "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "IWM", "EFA", "VGK"],
    "trading_enabled": False,
}


class StrategyConfigUpdate(BaseModel):
    active_strategies: list[str] | None = None
    confidence_threshold: float | None = None
    ensemble_method: str | None = None
    weights: dict[str, float] | None = None
    trading_enabled: bool | None = None
    symbols: list[str] | None = None


async def _get_or_create_config(db: AsyncSession) -> StrategyConfig:
    """Get the singleton config row, creating it with defaults if missing."""
    result = await db.execute(select(StrategyConfig).where(StrategyConfig.id == 1))
    config = result.scalar_one_or_none()
    if config is None:
        config = StrategyConfig(
            id=1,
            active_strategies=_DEFAULTS["active_strategies"],
            confidence_threshold=_DEFAULTS["confidence_threshold"],
            ensemble_method=_DEFAULTS["ensemble_method"],
            weights=_DEFAULTS["weights"],
            symbols=_DEFAULTS["symbols"],
            trading_enabled=_DEFAULTS["trading_enabled"],
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


def _config_to_dict(config: StrategyConfig) -> dict:
    return {
        "active_strategies": config.active_strategies,
        "confidence_threshold": config.confidence_threshold,
        "ensemble_method": config.ensemble_method,
        "weights": config.weights,
        "symbols": config.symbols,
        "trading_enabled": config.trading_enabled,
    }


@router.get("/strategy/status")
async def get_strategy_status(db: AsyncSession = Depends(get_db)):
    config = await _get_or_create_config(db)
    return {
        "config": _config_to_dict(config),
        "available_strategies": [
            {
                "name": "ml_xgboost",
                "type": "ML",
                "description": "XGBoost classifier on technical indicators",
            },
            {
                "name": "sentiment",
                "type": "LLM",
                "description": "Claude LLM sentiment analysis of news",
            },
            {
                "name": "nn_lstm",
                "type": "Neural Network",
                "description": "PyTorch LSTM on feature sequences",
            },
            {
                "name": "ensemble",
                "type": "Ensemble",
                "description": "Weighted voting across active strategies",
            },
        ],
    }


@router.put("/strategy/config")
async def update_strategy_config(update: StrategyConfigUpdate, db: AsyncSession = Depends(get_db)):
    config = await _get_or_create_config(db)

    if update.active_strategies is not None:
        config.active_strategies = update.active_strategies
    if update.confidence_threshold is not None:
        config.confidence_threshold = update.confidence_threshold
    if update.ensemble_method is not None:
        config.ensemble_method = update.ensemble_method
    if update.weights is not None:
        config.weights = update.weights
    if update.trading_enabled is not None:
        config.trading_enabled = update.trading_enabled
    if update.symbols is not None:
        config.symbols = update.symbols
        try:
            from app.dependencies import get_trading_engine
            engine = get_trading_engine()
            engine.update_symbols(update.symbols)
        except RuntimeError:
            logger.warning("strategy.symbols_update_no_engine")

    await db.commit()
    await db.refresh(config)
    return _config_to_dict(config)


@router.get("/strategy/performance")
async def get_strategy_performance(db: AsyncSession = Depends(get_db)):
    """Return performance metrics grouped by strategy."""
    query = (
        select(
            Trade.strategy_name,
            func.count(Trade.id).label("total_trades"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("winning_trades"),
            func.sum(case((Trade.realized_pnl <= 0, 1), else_=0)).label("losing_trades"),
            func.coalesce(func.sum(Trade.realized_pnl), 0).label("total_pnl"),
            func.coalesce(func.avg(Trade.realized_pnl), 0).label("avg_pnl"),
            func.coalesce(
                func.sum(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=0)), 0
            ).label("gross_profit"),
            func.coalesce(
                func.sum(case((Trade.realized_pnl < 0, func.abs(Trade.realized_pnl)), else_=0)), 0
            ).label("gross_loss"),
        )
        .where(Trade.status == TradeStatus.CLOSED)
        .group_by(Trade.strategy_name)
    )

    result = await db.execute(query)
    rows = result.all()

    strategies = []
    for row in rows:
        total = row.total_trades or 0
        winning = row.winning_trades or 0
        gross_profit = float(row.gross_profit or 0)
        gross_loss = float(row.gross_loss or 0)
        win_rate = (winning / total * 100) if total > 0 else 0.0
        profit_factor = (
            (gross_profit / gross_loss) if gross_loss > 0
            else float("inf") if gross_profit > 0
            else 0.0
        )

        strategies.append({
            "strategy_name": row.strategy_name,
            "total_trades": total,
            "winning_trades": winning,
            "losing_trades": row.losing_trades or 0,
            "total_pnl": round(float(row.total_pnl or 0), 2),
            "avg_pnl": round(float(row.avg_pnl or 0), 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
        })

    return {"strategies": strategies}
