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
    "NFLX": "technology", "CRM": "technology", "AVGO": "technology",
    # US Finance
    "JPM": "finance", "BAC": "finance", "GS": "finance",
    "MS": "finance", "WFC": "finance", "C": "finance",
    # US Healthcare
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare",
    # US Consumer
    "WMT": "consumer", "KO": "consumer", "PEP": "consumer",
    "PG": "consumer", "COST": "consumer", "HD": "consumer",
    # US Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    # ETFs
    "SPY": "etf_broad", "QQQ": "etf_tech", "IWM": "etf_broad",
    "DIA": "etf_broad", "VTI": "etf_broad", "VOO": "etf_broad",
    "EFA": "etf_intl", "XLF": "etf_finance", "XLE": "etf_energy",
    "VGK": "etf_eu", "EWG": "etf_eu", "EWN": "etf_eu",
    # EU Tech
    "ASML.AS": "technology", "SAP.DE": "technology", "IFX.DE": "technology",
    "ADYEN.AS": "technology", "CAP.PA": "technology",
    # EU Energy
    "SHEL.L": "energy", "TTE.PA": "energy", "BP.L": "energy",
    # EU Consumer
    "MC.PA": "consumer", "OR.PA": "consumer", "KER.PA": "consumer",
    "RI.PA": "consumer",
    # EU Finance
    "INGA.AS": "finance", "BNP.PA": "finance", "BARC.L": "finance",
    "LLOY.L": "finance", "LSEG.L": "finance", "CS.PA": "finance",
    # EU Healthcare
    "SAN.PA": "healthcare", "AZN.L": "healthcare", "GSK.L": "healthcare",
    "BAYN.DE": "healthcare",
    # EU Industrials
    "SIE.DE": "industrials", "AIR.PA": "industrials", "REL.L": "industrials",
    # EU Telecom / Other
    "AAL.L": "industrials",
}

# Dynamic sector cache for symbols discovered by the screener
_sector_cache: dict[str, str] = {}

DEFAULT_SECTOR = "unknown"
MAX_SECTOR_CONCENTRATION_PCT = 40.0  # Max 40% in one sector

# Track pending allocations to prevent double-sizing on same symbol
_pending_allocations: dict[str, float] = {}

# yfinance sector → our sector mapping
_YF_SECTOR_MAP: dict[str, str] = {
    "technology": "technology",
    "communication services": "technology",
    "consumer cyclical": "consumer",
    "consumer defensive": "consumer",
    "financial services": "finance",
    "healthcare": "healthcare",
    "energy": "energy",
    "industrials": "industrials",
    "basic materials": "materials",
    "real estate": "real_estate",
    "utilities": "utilities",
}


@dataclass
class PositionSizeResult:
    quantity: int
    target_allocation: float
    kelly_fraction: float
    volatility_factor: float
    correlation_factor: float
    sector_factor: float


def get_sector(symbol: str) -> str:
    """Get sector for a symbol. Checks hardcoded map first, then dynamic cache."""
    upper = symbol.upper()
    # 1. Hardcoded map (fast)
    sector = SECTOR_MAP.get(upper)
    if sector:
        return sector
    # 2. Dynamic cache (populated by screener or yfinance lookup)
    sector = _sector_cache.get(upper)
    if sector:
        return sector
    # 3. Infer from exchange suffix
    if "." in symbol:
        suffix = symbol.split(".")[-1]
        if suffix in ("AS", "DE", "PA", "L"):
            return "eu_equity"
    return DEFAULT_SECTOR


def cache_sector(symbol: str, sector: str) -> None:
    """Add a symbol→sector mapping to the dynamic cache."""
    normalized = _YF_SECTOR_MAP.get(sector.lower(), sector.lower())
    _sector_cache[symbol.upper()] = normalized


def reserve_allocation(symbol: str, amount: float) -> None:
    """Reserve allocation for a symbol to prevent double-sizing."""
    _pending_allocations[symbol] = _pending_allocations.get(symbol, 0.0) + amount


def release_allocation(symbol: str) -> None:
    """Release reserved allocation after fill or cancel."""
    _pending_allocations.pop(symbol, None)


def get_pending_allocation(symbol: str) -> float:
    """Get currently reserved allocation for a symbol."""
    return _pending_allocations.get(symbol, 0.0)


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


def _confidence_to_kelly(confidence: float) -> float:
    """Linear scaling: threshold -> 0.10, max -> 0.50.

    Below threshold: 0 allocation. Above: linear from 0.10 to 0.50.
    """
    threshold = 0.40
    if confidence < threshold:
        return 0.0
    # Linear from 0.10 to 0.50 as confidence goes from threshold to 1.0
    fraction = 0.10 + (confidence - threshold) / (1.0 - threshold) * 0.40
    return min(0.50, fraction)


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
    ai_modifier: float = 1.0,
) -> int:
    """Calculate the number of shares to buy.

    Uses Kelly criterion scaled by confidence, volatility, correlation, and sector.
    Always respects the max_position_pct limit.

    Supports margin accounts: when cash is negative (margin loan), buying_power
    is used to determine available funds for new positions.
    """
    total_value = portfolio.account_summary.total_value
    cash = portfolio.account_summary.cash
    buying_power = portfolio.account_summary.buying_power

    if price <= 0 or total_value <= 0:
        return 0

    # For margin accounts, cash can be negative (margin loan).
    # Use buying_power as the available funds limit instead.
    available_funds = buying_power if cash <= 0 else cash
    if available_funds <= 0:
        return 0

    # Maximum allowed allocation for this position
    max_allocation = total_value * (max_position_pct / 100)

    # Never exceed total_value * max_position_pct regardless of margin/buying_power
    available_funds = min(available_funds, max_allocation)

    # Subtract any pending allocation for this symbol
    if symbol:
        pending = get_pending_allocation(symbol)
        if pending > 0:
            max_allocation = max(0, max_allocation - pending)
            if max_allocation <= 0:
                logger.debug("position_sizer.duplicate_blocked", symbol=symbol)
                return 0

    # Kelly fraction
    if win_rate is not None and avg_win_loss_ratio is not None:
        kelly_fraction = calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win_loss_ratio,
            avg_loss=1.0,
        )
    else:
        # Tiered Kelly: high-confidence trades get proportionally more capital
        kelly_fraction = _confidence_to_kelly(confidence)

    # Volatility adjustment: mild reduction for extreme volatility only
    if volatility and volatility > 0.05:
        vol_factor = max(0.7, 1.0 - (volatility - 0.05) * 0.3)
    else:
        vol_factor = 1.0  # No penalty for normal volatility

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

    # AI sizing modifier (clamped to 0.5–1.5)
    ai_factor = max(0.5, min(1.5, ai_modifier))

    target_allocation = (
        max_allocation * kelly_fraction * vol_factor * corr_factor * sector_factor * ai_factor
    )

    # Never exceed available funds (keep 10% buffer)
    target_allocation = min(target_allocation, available_funds * 0.9)

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
            ai_factor=round(ai_factor, 2),
        )

    return max(0, quantity)


def calculate_take_profit(
    entry_price: float,
    symbol: str | None = None,
    atr: float | None = None,
) -> float:
    """Calculate take-profit target price using ATR with a minimum percentage floor.

    Returns take-profit price above entry.
    """
    from app.config import settings

    # ATR-based target (convert ratio ATR to absolute if needed)
    if atr and atr > 0:
        atr_raw = atr * entry_price if atr < 1.0 else atr
        atr_target = entry_price + settings.atr_take_profit_multiplier * atr_raw
    else:
        atr_target = 0.0

    # Minimum percentage floor
    pct_target = entry_price * (1 + settings.min_take_profit_pct / 100)

    tp = max(atr_target, pct_target)

    if symbol:
        logger.debug(
            "position_sizer.take_profit",
            symbol=symbol,
            entry=entry_price,
            take_profit=round(tp, 2),
            atr=atr,
        )

    return round(tp, 2)


def check_risk_reward_ratio(
    entry_price: float,
    stop_price: float,
    take_profit: float,
    min_ratio: float = 1.5,
) -> tuple[bool, float]:
    """Check if a trade meets the minimum risk:reward ratio.

    Returns (passes, ratio). A ratio of 2.0 means reward is 2x the risk.
    """
    risk = entry_price - stop_price
    reward = take_profit - entry_price
    if risk <= 0:
        return False, 0.0
    ratio = reward / risk
    return ratio >= min_ratio, round(ratio, 2)


def calculate_progressive_trailing_stop(
    entry_price: float,
    current_price: float,
    current_stop: float,
    atr: float | None = None,
    tiers: list[tuple[float, float]] | None = None,
) -> float:
    """Calculate a progressive trailing stop that tightens as profit grows.

    Tiers are (gain_pct, trail_pct) pairs sorted highest-gain first.
    The trail percentage used is determined by the highest tier the current
    gain exceeds.

    Returns a new stop price (never lower than current_stop).
    """
    if entry_price <= 0:
        return current_stop

    gain_pct = ((current_price - entry_price) / entry_price) * 100

    if tiers is None:
        from app.config import settings
        tiers = settings.trailing_stop_tiers_parsed

    # Find the applicable tier (highest gain threshold we exceed)
    trail_pct = None
    for threshold, trail in tiers:
        if gain_pct >= threshold:
            trail_pct = trail
            break

    if trail_pct is None:
        # Below all tiers — use default ATR-based trailing
        return current_stop

    # Calculate stop from current price using the tier trail percentage
    new_stop = round(current_price * (1 - trail_pct / 100), 2)

    # Never lower the stop
    return max(new_stop, current_stop)


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
        from app.config import settings
        atr_stop = current_price - (settings.atr_stop_multiplier * atr)
        # Floor: never less than the minimum percentage stop
        pct_stop = current_price * (1 - trail_pct / 100)
        return max(atr_stop, pct_stop)

    # Fixed percentage stop
    return round(current_price * (1 - trail_pct / 100), 2)
