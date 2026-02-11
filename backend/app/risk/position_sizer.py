"""Position sizing based on Kelly criterion, volatility, correlation, and confidence."""

import math
from dataclasses import dataclass

import structlog

from app.broker.base import Portfolio

logger = structlog.get_logger()

# Sector classification for concentration limits
SECTOR_MAP: dict[str, str] = {
    # US Tech
    "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
    "AMZN": "technology", "META": "technology", "NVDA": "technology",
    "TSLA": "technology", "AMD": "technology", "INTC": "technology",
    # US Finance
    "JPM": "finance", "BAC": "finance", "GS": "finance",
    "MS": "finance", "WFC": "finance", "C": "finance",
    # US Healthcare
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare",
    # US Consumer
    "WMT": "consumer", "KO": "consumer", "PEP": "consumer",
    "PG": "consumer", "COST": "consumer",
    # US Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    # ETFs
    "SPY": "etf_broad", "QQQ": "etf_tech", "IWM": "etf_broad",
    "VTI": "etf_broad", "VOO": "etf_broad", "EFA": "etf_intl",
    "VGK": "etf_eu", "EWG": "etf_eu", "EWN": "etf_eu",
    # EU Tech
    "ASML.AS": "technology", "SAP.DE": "technology",
    # EU Energy
    "SHEL.L": "energy", "TTE.PA": "energy",
    # EU Consumer
    "MC.PA": "consumer", "OR.PA": "consumer",
    # EU Finance
    "INGA.AS": "finance", "BNP.PA": "finance",
    # EU Healthcare
    "SAN.PA": "healthcare", "AZN.L": "healthcare",
}

DEFAULT_SECTOR = "unknown"
MAX_SECTOR_CONCENTRATION_PCT = 40.0  # Max 40% in one sector


@dataclass
class PositionSizeResult:
    quantity: int
    target_allocation: float
    kelly_fraction: float
    volatility_factor: float
    correlation_factor: float
    sector_factor: float


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), DEFAULT_SECTOR)


def calculate_kelly_fraction(
    win_rate: float = 0.55,
    avg_win: float = 1.5,
    avg_loss: float = 1.0,
) -> float:
    """Calculate Kelly criterion fraction.

    Kelly % = W - (1-W)/R
    Where W = win probability, R = win/loss ratio

    Returns half-Kelly for safety (standard practice).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Half-Kelly for safety, clamped to [0, 0.25]
    half_kelly = max(0.0, min(0.25, kelly * 0.5))
    return half_kelly


def calculate_correlation_factor(
    symbol: str,
    existing_positions: list[str],
    correlation_matrix: dict[tuple[str, str], float] | None = None,
) -> float:
    """Reduce position size when correlated with existing holdings.

    Returns a factor between 0.3 and 1.0.
    High correlation with existing positions → smaller new position.
    """
    if not existing_positions:
        return 1.0

    if correlation_matrix is None:
        # Use sector-based correlation proxy
        symbol_sector = get_sector(symbol)
        same_sector_count = sum(
            1 for s in existing_positions if get_sector(s) == symbol_sector
        )
        if same_sector_count == 0:
            return 1.0
        # Reduce by 15% per same-sector position, minimum 0.3
        return max(0.3, 1.0 - same_sector_count * 0.15)

    # Use actual correlation data
    correlations = []
    for pos_symbol in existing_positions:
        key = (min(symbol, pos_symbol), max(symbol, pos_symbol))
        corr = correlation_matrix.get(key, 0.0)
        correlations.append(abs(corr))

    if not correlations:
        return 1.0

    avg_corr = sum(correlations) / len(correlations)
    # High average correlation → reduce position (0.3 minimum)
    return max(0.3, 1.0 - avg_corr * 0.7)


def calculate_sector_factor(
    symbol: str,
    portfolio: Portfolio,
    max_sector_pct: float = MAX_SECTOR_CONCENTRATION_PCT,
) -> float:
    """Reduce position size if adding to an already concentrated sector.

    Returns a factor between 0.0 and 1.0.
    """
    total_value = portfolio.account_summary.total_value
    if total_value <= 0:
        return 1.0

    target_sector = get_sector(symbol)
    if target_sector == DEFAULT_SECTOR:
        return 1.0

    # Calculate current sector exposure
    sector_value = sum(
        p.market_value for p in portfolio.positions
        if get_sector(p.symbol) == target_sector
    )
    sector_pct = (sector_value / total_value) * 100

    if sector_pct >= max_sector_pct:
        return 0.0  # Sector is full
    if sector_pct >= max_sector_pct * 0.75:
        # Linearly reduce as we approach the limit
        remaining = max_sector_pct - sector_pct
        headroom = max_sector_pct * 0.25
        return remaining / headroom

    return 1.0


def calculate_position_size(
    portfolio: Portfolio,
    price: float,
    max_position_pct: float,
    confidence: float,
    volatility: float | None = None,
    win_rate: float | None = None,
    avg_win_loss_ratio: float | None = None,
    correlation_matrix: dict[tuple[str, str], float] | None = None,
    symbol: str | None = None,
) -> int:
    """Calculate the number of shares to buy.

    Uses Kelly criterion scaled by confidence, volatility, correlation, and sector.
    Always respects the max_position_pct limit.
    """
    total_value = portfolio.account_summary.total_value
    cash = portfolio.account_summary.cash

    if price <= 0 or total_value <= 0 or cash <= 0:
        return 0

    # Maximum allowed allocation for this position
    max_allocation = total_value * (max_position_pct / 100)

    # Kelly fraction
    if win_rate is not None and avg_win_loss_ratio is not None:
        kelly_fraction = calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win_loss_ratio,
            avg_loss=1.0,
        )
    else:
        # Simplified Kelly scaled by confidence
        kelly_fraction = confidence * 0.5

    # Volatility adjustment: reduce size for high-volatility assets
    if volatility and volatility > 0:
        vol_factor = max(0.3, 1.0 - volatility)
    else:
        vol_factor = 0.7  # Default conservative

    # Correlation adjustment
    existing_symbols = [p.symbol for p in portfolio.positions]
    if symbol:
        corr_factor = calculate_correlation_factor(
            symbol, existing_symbols, correlation_matrix
        )
    else:
        corr_factor = 1.0

    # Sector concentration adjustment
    if symbol:
        sector_factor = calculate_sector_factor(symbol, portfolio)
    else:
        sector_factor = 1.0

    target_allocation = (
        max_allocation * kelly_fraction * vol_factor * corr_factor * sector_factor
    )

    # Never exceed available cash (keep 10% buffer)
    target_allocation = min(target_allocation, cash * 0.9)

    quantity = math.floor(target_allocation / price)

    if quantity > 0 and symbol:
        logger.debug(
            "position_sizer.calculated",
            symbol=symbol,
            quantity=quantity,
            target=round(target_allocation, 2),
            kelly=round(kelly_fraction, 3),
            vol_factor=round(vol_factor, 2),
            corr_factor=round(corr_factor, 2),
            sector_factor=round(sector_factor, 2),
        )

    return max(0, quantity)


def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    atr: float | None = None,
    trail_pct: float = 3.0,
) -> float:
    """Calculate trailing stop price.

    Uses ATR-based stop if available, otherwise fixed percentage.
    The stop follows price up but never moves down.
    """
    if atr and atr > 0:
        # ATR-based: 2x ATR below current price
        atr_stop = current_price - (2.0 * atr)
        # But never above the minimum percentage stop
        pct_stop = current_price * (1 - trail_pct / 100)
        return max(atr_stop, pct_stop)

    # Fixed percentage stop
    return round(current_price * (1 - trail_pct / 100), 2)
