from datetime import date, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter
from app.broker.mock_adapter import MockBrokerAdapter
from app.config import settings
from app.core.event_bus import EventBus, event_bus
from app.data.pipeline import DataPipeline
from app.execution.engine import TradingEngine
from app.models.database import async_session as _session_factory
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

    # 1. Swing Strategy (primary — daily trend following)
    try:
        from app.strategy.swing_strategy import SwingStrategy
        swing = SwingStrategy(confidence_threshold=settings.confidence_threshold)
        strategies.append(swing)
        logger.info("strategies.loaded", name="swing")
    except Exception:
        logger.exception("strategies.load_error", name="swing")

    # 2. ML Strategy (XGBoost) — restricted to symbols with proven edge (walk-forward)
    try:
        from app.strategy.ml_strategy import MLStrategy
        ml = MLStrategy(allowed_symbols=settings.ml_xgboost_allowed_symbols_set or None)
        strategies.append(ml)
        logger.info(
            "strategies.loaded",
            name="ml_xgboost",
            allowed_symbols=sorted(settings.ml_xgboost_allowed_symbols_set) or "all",
        )
    except Exception:
        logger.exception("strategies.load_error", name="ml_xgboost")

    # 3. Agentic Strategy (Claude AI) — swing trading decisions
    if settings.anthropic_api_key:
        try:
            from app.strategy.agentic_strategy import AgenticStrategy
            agentic = AgenticStrategy(confidence_threshold=settings.confidence_threshold)
            strategies.append(agentic)
            logger.info("strategies.loaded", name="agentic")
        except Exception:
            logger.exception("strategies.load_error", name="agentic")

    # 3b. Sentiment Strategy (Claude LLM) - disabled for day trading (too slow)
    if settings.anthropic_api_key:
        try:
            from app.strategy.sentiment_strategy import SentimentStrategy
            sentiment = SentimentStrategy(min_confidence=settings.confidence_threshold)
            strategies.append(sentiment)
            logger.info("strategies.loaded", name="sentiment")
        except Exception:
            logger.exception("strategies.load_error", name="sentiment")
    else:
        logger.warning("strategies.skipped", name="sentiment", reason="no API key")

    # 3. Neural Network Strategy (LSTM) - if trained model exists
    try:
        from app.strategy.nn_strategy import NNStrategy
        nn = NNStrategy()
        if nn._model is not None:
            strategies.append(nn)
            logger.info("strategies.loaded", name="nn_lstm")
        else:
            logger.info("strategies.skipped", name="nn_lstm", reason="no trained model")
    except Exception:
        logger.exception("strategies.load_error", name="nn_lstm")

    # 4. Ensemble Strategy - combines all available strategies
    if len(strategies) >= 2:
        try:
            from app.strategy.ensemble import EnsembleStrategy
            # Initial weights: sentiment leads (proven better live P&L),
            # ML reduced until retrained on live data distribution.
            # update_weights_from_history() will auto-adjust from here.
            initial_weights = {}
            for s in strategies:
                if s.name == "ml_xgboost":
                    initial_weights[s.name] = 0.5
                else:
                    initial_weights[s.name] = 1.0
            ensemble = EnsembleStrategy(
                strategies=list(strategies),
                weights=initial_weights,
            )
            strategies.append(ensemble)
            logger.info("strategies.loaded", name="ensemble", sub_strategies=len(strategies) - 1)
        except Exception:
            logger.exception("strategies.load_error", name="ensemble")

    logger.info("strategies.ready", count=len(strategies))
    return strategies


def _has_eu_suffix(symbol: str) -> bool:
    """Check if a symbol has a European exchange suffix."""
    return any(symbol.upper().endswith(s) for s in (".AS", ".PA", ".BR", ".L", ".DE"))


def should_skip_eu(symbol: str) -> bool:
    """Return True if this EU symbol should be skipped (EU trading disabled)."""
    return not settings.enable_eu_trading and _has_eu_suffix(symbol)


async def get_startup_symbols() -> list[str]:
    """Load symbols from the latest screener result (max 3 days old), fallback to config.

    EU-suffixed symbols are excluded because the IBKR paper account does not have
    European market data subscriptions, causing 30s timeouts per symbol.
    """
    if not settings.screener_enabled:
        symbols = settings.symbols_list
    else:
        symbols = None
        try:
            from app.models.screening_result import ScreeningResult
            cutoff = date.today() - timedelta(days=3)
            async with _session_factory() as session:
                result = await session.execute(
                    select(ScreeningResult)
                    .where(ScreeningResult.screening_date >= cutoff)
                    .order_by(ScreeningResult.created_at.desc())
                    .limit(1)
                )
                row = result.scalar_one_or_none()
                if row and row.candidates:
                    symbols = [c["symbol"] for c in row.candidates if "symbol" in c]
        except Exception:
            logger.warning("startup.screener_symbols_failed")

        if not symbols:
            symbols = settings.symbols_list

    # Filter out EU symbols when EU trading is disabled (no IBKR data subscription)
    eu_removed = [s for s in symbols if should_skip_eu(s)]
    symbols = [s for s in symbols if not should_skip_eu(s)]
    if eu_removed:
        logger.info(
            "startup.eu_symbols_filtered",
            removed=eu_removed,
            remaining=len(symbols),
        )

    if symbols:
        logger.info(
            "startup.symbols_from_screener",
            count=len(symbols),
        )
    return symbols


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

    symbols = await get_startup_symbols()

    _trading_engine = TradingEngine(
        broker=broker,
        strategies=strategies,
        risk_manager=risk_manager,
        market_data=pipeline._market_data,
        performance=performance,
        db=db,
        symbols=symbols,
        trading_enabled=trading_enabled,
        session_factory=_session_factory,
    )

    return _trading_engine
