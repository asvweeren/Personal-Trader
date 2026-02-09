from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from app.data.market_data import MarketSnapshot


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    symbol: str
    action: SignalAction
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    features_snapshot: dict | None = None
    metadata: dict | None = None


@dataclass
class TrainResult:
    success: bool
    metrics: dict
    message: str = ""


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""

    @abstractmethod
    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        """Generate trading signals from market data."""

    async def train(self, historical_data: pd.DataFrame) -> TrainResult:
        """Train the strategy on historical data. Override in ML strategies."""
        return TrainResult(success=True, metrics={}, message="No training needed")

    def get_confidence(self) -> float:
        """Return the current confidence level of the strategy."""
        return 0.5
