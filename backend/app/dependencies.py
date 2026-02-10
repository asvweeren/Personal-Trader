from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from app.broker.base import BrokerAdapter
from app.broker.mock_adapter import MockBrokerAdapter
from app.config import settings
from app.core.event_bus import event_bus, EventBus
from app.data.pipeline import DataPipeline
from app.execution.engine import TradingEngine
from app.models.database import get_session
from app.models.strategy_config import StrategyConfig
from app.monitoring.performance import PerformanceTracker
from app.risk.manager import RiskManager
from app.strategy.base import Strategy

logger = structlog.get_logger()

# Singletons
_broker: BrokerAdapter | None = None
_risk_manager: RiskManager | None = None
_performance_tracker: PerformanceTracker | None = None
_data_pipeline: DataPipeline | None = None
_trading_engine: TradingEngine | None = None


def get_event_bus() -> EventBus:
    return event_bus


async def get_db() -> AsyncSession:
    async for session in get_session():
        yield session


def get_broker() -> BrokerAdapter:
    global _broker
    if _broker is None:
        if settings.app_env == "test":
            _broker = MockBrokerAdapter()
        else:
            from app.broker.ibkr_adapter import IBKRAdapter

            _broker = IBKRAdapter(
                host=settings.ibkr_host,
                port=settings.ibkr_port,
                client_id=settings.ibkr_client_id,
            )
    return _broker


def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_position_pct=settings.max_position_pct,
            max_open_positions=settings.max_open_positions,
            min_cash_reserve_pct=settings.min_cash_reserve_pct,
        )
    return _risk_manager


def get_performance_tracker() -> PerformanceTracker:
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(initial_capital=settings.initial_capital)
    return _performance_tracker


def get_data_pipeline() -> DataPipeline:
    global _data_pipeline
    if _data_pipeline is None:
        _data_pipeline = DataPipeline(broker=get_broker())
    return _data_pipeline


def get_trading_engine() -> TradingEngine:
    """Get the trading engine singleton. Must be initialized via init_trading_engine()."""
    if _trading_engine is None:
        raise RuntimeError("Trading engine not initialized. Call init_trading_engine() first.")
    return _trading_engine


def load_strategies() -> list[Strategy]:
    """Load and initialize all available trading strategies."""
    strategies: list[Strategy] = []

    # 1. ML Strategy (XGBoost) - loads trained model from disk
    try:
        from app.strategy.ml_strategy import MLStrategy
        ml = MLStrategy(confidence_threshold=0.5)
        if ml._model is not None:
            strategies.append(ml)
            logger.info("strategies.loaded", name="ml_xgboost", features=len(ml._feature_columns))
        else:
            logger.warning("strategies.skipped", name="ml_xgboost", reason="no trained model")
    except Exception:
        logger.exception("strategies.load_error", name="ml_xgboost")

    # 2. Sentiment Strategy (Claude LLM) - requires Anthropic API key
    if settings.anthropic_api_key:
        try:
            from app.strategy.sentiment_strategy import SentimentStrategy
            sentiment = SentimentStrategy(min_confidence=0.5)
            strategies.append(sentiment)
            logger.info("strategies.loaded", name="sentiment")
        except Exception:
            logger.exception("strategies.load_error", name="sentiment")
    else:
        logger.warning("strategies.skipped", name="sentiment", reason="no API key")

    # 3. Ensemble Strategy - wraps the above if we have 2+ strategies
    if len(strategies) >= 2:
        try:
            from app.strategy.ensemble import EnsembleStrategy
            ensemble = EnsembleStrategy(
                strategies=list(strategies),
                weights={"ml_xgboost": 0.6, "sentiment": 0.4},
                agreement_threshold=0.3,
            )
            strategies.append(ensemble)
            logger.info("strategies.loaded", name="ensemble", sub_strategies=len(strategies) - 1)
        except Exception:
            logger.exception("strategies.load_error", name="ensemble")

    logger.info("strategies.ready", count=len(strategies))
    return strategies


async def init_trading_engine(db: AsyncSession) -> TradingEngine:
    """Initialize the trading engine with all dependencies."""
    global _trading_engine
    if _trading_engine is not None:
        return _trading_engine

    broker = get_broker()
    risk_manager = get_risk_manager()
    performance = get_performance_tracker()
    pipeline = get_data_pipeline()
    strategies = load_strategies()

    # Read persisted trading_enabled from DB
    trading_enabled = False
    try:
        result = await db.execute(select(StrategyConfig).limit(1))
        config = result.scalar_one_or_none()
        if config:
            trading_enabled = config.trading_enabled
            logger.info("engine.trading_restored", enabled=trading_enabled)
    except Exception:
        logger.warning("engine.trading_restore_failed")

    _trading_engine = TradingEngine(
        broker=broker,
        strategies=strategies,
        risk_manager=risk_manager,
        market_data=pipeline._market_data,
        performance=performance,
        db=db,
        symbols=settings.symbols_list,
        trading_enabled=trading_enabled,
    )

    return _trading_engine
