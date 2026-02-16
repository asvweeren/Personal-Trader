class TraderError(Exception):
    """Base exception for all trader errors."""


class BrokerConnectionError(TraderError):
    """Failed to connect to the broker."""


class BrokerOrderError(TraderError):
    """Error placing or managing an order."""


class RiskLimitExceeded(TraderError):
    """A risk limit has been exceeded."""


class DailyLossLimitExceeded(RiskLimitExceeded):
    """Daily loss limit has been hit - all trading stopped."""


class MaxPositionsExceeded(RiskLimitExceeded):
    """Maximum number of open positions reached."""


class InsufficientCashReserve(RiskLimitExceeded):
    """Cash reserve would drop below minimum."""


class PositionSizeLimitExceeded(RiskLimitExceeded):
    """Position would exceed maximum portfolio percentage."""


class MaxLeverageExceeded(RiskLimitExceeded):
    """Total positions value would exceed maximum leverage ratio."""


class MarketClosedError(TraderError):
    """Attempted to trade outside market hours."""


class StrategyError(TraderError):
    """Error in strategy execution."""


class DataError(TraderError):
    """Error fetching or processing data."""
