from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()

# In-memory strategy config (will be persisted to DB later)
_strategy_config = {
    "active_strategies": ["ml_xgboost", "sentiment"],
    "confidence_threshold": 0.6,
    "ensemble_method": "weighted_average",
    "weights": {"ml_xgboost": 0.5, "sentiment": 0.3, "nn_lstm": 0.2},
    "trading_enabled": False,
}


class StrategyConfigUpdate(BaseModel):
    active_strategies: list[str] | None = None
    confidence_threshold: float | None = None
    ensemble_method: str | None = None
    weights: dict[str, float] | None = None
    trading_enabled: bool | None = None


@router.get("/strategy/status")
async def get_strategy_status():
    return {
        "config": _strategy_config,
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
async def update_strategy_config(update: StrategyConfigUpdate):
    if update.active_strategies is not None:
        _strategy_config["active_strategies"] = update.active_strategies
    if update.confidence_threshold is not None:
        _strategy_config["confidence_threshold"] = update.confidence_threshold
    if update.ensemble_method is not None:
        _strategy_config["ensemble_method"] = update.ensemble_method
    if update.weights is not None:
        _strategy_config["weights"] = update.weights
    if update.trading_enabled is not None:
        _strategy_config["trading_enabled"] = update.trading_enabled
    return _strategy_config
