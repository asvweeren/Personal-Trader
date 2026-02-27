"""Correlation matrix computation from cached market data.

Uses daily close prices already loaded by the data pipeline — no extra API calls.
Results are cached for 1 hour to avoid redundant computation.
"""

import time

import numpy as np
import structlog

logger = structlog.get_logger()

# Module-level cache
_correlation_cache: dict[str, tuple[float, dict[tuple[str, str], float]]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour


async def get_correlation_matrix(
    symbols: list[str],
    market_data_service,
    snapshot=None,
) -> dict[tuple[str, str], float]:
    """Compute pairwise Pearson correlation from daily returns.

    Uses pre-loaded snapshot data when available, avoiding extra API calls.
    Returns a dict mapping (symbol_a, symbol_b) → correlation.
    """
    if len(symbols) < 2:
        return {}

    # Check cache
    cache_key = ",".join(sorted(symbols))
    now = time.monotonic()
    if cache_key in _correlation_cache:
        cached_at, cached_matrix = _correlation_cache[cache_key]
        if now - cached_at < _CACHE_TTL_SECONDS:
            return cached_matrix

    # Gather close prices — prefer snapshot data (already loaded) over extra API calls
    returns_by_symbol: dict[str, np.ndarray] = {}
    for symbol in symbols:
        try:
            df = None
            if snapshot and hasattr(snapshot, "ohlcv"):
                df = snapshot.ohlcv.get(symbol)
                # Use last 30 days from the 1Y snapshot
                if df is not None and len(df) > 30:
                    df = df.tail(30)
            if df is None or df.empty:
                df = await market_data_service.get_historical_data(
                    symbol, duration="30 D", bar_size="1 day"
                )
            if df is not None and len(df) >= 5:
                closes = df["close"].values if "close" in df.columns else None
                if closes is None and "Close" in df.columns:
                    closes = df["Close"].values
                if closes is not None and len(closes) >= 5:
                    prices = closes.astype(float)
                    rets = np.diff(prices) / prices[:-1]
                    returns_by_symbol[symbol] = rets
        except Exception:
            continue

    if len(returns_by_symbol) < 2:
        logger.debug("correlation.insufficient_data", symbols_with_data=len(returns_by_symbol))
        return {}

    # Align lengths (use minimum overlapping length)
    min_len = min(len(r) for r in returns_by_symbol.values())
    aligned = {s: r[-min_len:] for s, r in returns_by_symbol.items()}

    # Compute pairwise correlations
    result: dict[tuple[str, str], float] = {}
    sym_list = sorted(aligned.keys())
    for i in range(len(sym_list)):
        for j in range(i + 1, len(sym_list)):
            a, b = sym_list[i], sym_list[j]
            try:
                corr = float(np.corrcoef(aligned[a], aligned[b])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                result[(a, b)] = corr
            except Exception:
                result[(a, b)] = 0.0

    # Cache result
    _correlation_cache[cache_key] = (now, result)
    logger.info(
        "correlation.computed",
        symbols=len(returns_by_symbol),
        pairs=len(result),
    )

    return result
