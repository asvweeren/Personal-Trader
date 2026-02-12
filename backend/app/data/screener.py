"""Daily stock screener — selects top trading candidates from a broad universe.

Downloads 20 days of OHLCV data via yfinance, scores each symbol on momentum,
volume, and volatility, then returns the top N candidates.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import structlog

from app.config import settings

logger = structlog.get_logger()

# ── European blue-chip symbols ─────────────────────────────────────────────
EU_SYMBOLS: list[str] = [
    # Netherlands (.AS)
    "ASML.AS", "ADYEN.AS", "INGA.AS", "PHIA.AS", "UNA.AS", "HEIA.AS",
    "RAND.AS", "WKL.AS", "DSM.AS", "AKZA.AS",
    # Germany (.DE)
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE",
    "BMW.DE", "MBG.DE", "MUV2.DE", "ADS.DE", "AIR.DE", "DHL.DE",
    "IFX.DE", "HEN3.DE", "DB1.DE",
    # France (.PA)
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "SU.PA",
    "BNP.PA", "ACA.PA", "CS.PA", "DG.PA", "SAF.PA", "RI.PA",
    "KER.PA", "CAP.PA", "SGO.PA",
    # UK (.L) — prices in pence, but scoring still works on returns/ratios
    "SHEL.L", "AZN.L", "ULVR.L", "HSBA.L", "BP.L", "GSK.L",
    "RIO.L", "LSEG.L", "BATS.L", "DGE.L", "REL.L", "AAL.L",
    "LLOY.L", "BARC.L", "VOD.L",
]

# ── Fallback S&P 500 list (top ~100 by market cap) ────────────────────────
_SP500_FALLBACK: list[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "TMO", "ACN", "ABT", "DHR", "CRM", "ADBE", "AMD", "NFLX", "CMCSA",
    "TXN", "INTC", "QCOM", "NEE", "HON", "UPS", "PM", "LOW", "UNP",
    "RTX", "SPGI", "GS", "BA", "CAT", "ISRG", "BLK", "AMAT", "AMGN",
    "LMT", "BKNG", "MDLZ", "ELV", "GILD", "ADP", "PLD", "VRTX",
    "SYK", "CI", "CB", "ADI", "REGN", "SO", "DUK", "MMC", "BDX",
    "LRCX", "BSX", "ZTS", "PANW", "KLAC", "CL", "EOG", "ICE",
    "HUM", "APD", "MO", "SLB", "PNC", "CME", "FDX", "ORLY",
    "SNPS", "WM", "NOC", "GD", "MCK", "SHW", "CDNS", "EMR",
    "PSA", "AJG", "MET", "AIG", "TGT", "APH", "MCHP", "FTNT",
    "MSI", "GM", "F", "PYPL", "COIN", "SQ", "SHOP", "SNOW",
    "DDOG", "CRWD", "NET", "ZS", "ABNB", "UBER", "DASH", "RBLX",
    "SPY", "QQQ", "IWM", "DIA", "EFA", "VGK", "XLF", "XLE",
]


def get_sp500_symbols() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia, with fallback to hardcoded list."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Symbol",
        )
        if tables:
            df = tables[0]
            symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            if len(symbols) > 400:
                logger.info("screener.sp500_fetched", count=len(symbols))
                return symbols
    except Exception:
        logger.warning("screener.sp500_fetch_failed, using fallback")

    logger.info("screener.using_fallback_sp500", count=len(_SP500_FALLBACK))
    return list(_SP500_FALLBACK)


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI for the last value of a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Compute ATR for the last value."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


class StockScreener:
    """Screens a broad stock universe and selects top trading candidates."""

    async def run_screening(
        self,
        max_candidates: int | None = None,
        min_avg_volume: int | None = None,
        momentum_weight: float | None = None,
        volume_weight: float | None = None,
        volatility_weight: float | None = None,
        include_eu: bool | None = None,
    ) -> dict:
        """Run the full screening pipeline.

        Returns a dict matching the ScreeningResult model fields:
        {screening_date, total_scanned, candidates, config}
        """
        max_candidates = max_candidates or settings.screener_max_candidates
        min_avg_volume = min_avg_volume or settings.screener_min_avg_volume
        momentum_weight = momentum_weight if momentum_weight is not None else settings.screener_momentum_weight
        volume_weight = volume_weight if volume_weight is not None else settings.screener_volume_weight
        volatility_weight = volatility_weight if volatility_weight is not None else settings.screener_volatility_weight
        include_eu = include_eu if include_eu is not None else settings.screener_include_eu

        config = {
            "max_candidates": max_candidates,
            "min_avg_volume": min_avg_volume,
            "momentum_weight": momentum_weight,
            "volume_weight": volume_weight,
            "volatility_weight": volatility_weight,
            "include_eu": include_eu,
        }

        # Build universe
        us_symbols = get_sp500_symbols()
        universe = list(us_symbols)
        if include_eu:
            universe.extend(EU_SYMBOLS)

        # Deduplicate
        universe = list(dict.fromkeys(universe))
        total_scanned = len(universe)
        logger.info("screener.start", total_symbols=total_scanned)

        # Download data in a thread (yfinance is sync / blocking)
        try:
            data = await asyncio.to_thread(
                self._download_data, universe
            )
        except Exception:
            logger.exception("screener.download_failed")
            return {
                "screening_date": date.today().isoformat(),
                "total_scanned": total_scanned,
                "candidates": [],
                "config": config,
            }

        # Score each symbol
        scored = []
        for symbol in universe:
            try:
                score_data = self._score_symbol(
                    symbol, data,
                    min_avg_volume=min_avg_volume,
                    momentum_weight=momentum_weight,
                    volume_weight=volume_weight,
                    volatility_weight=volatility_weight,
                )
                if score_data is not None:
                    scored.append(score_data)
            except Exception:
                continue

        # Sort by composite score descending, take top N
        scored.sort(key=lambda x: x["score"], reverse=True)
        candidates = scored[:max_candidates]

        logger.info(
            "screener.complete",
            total_scanned=total_scanned,
            passed_filter=len(scored),
            selected=len(candidates),
        )

        return {
            "screening_date": date.today().isoformat(),
            "total_scanned": total_scanned,
            "candidates": candidates,
            "config": config,
        }

    def _download_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Download 20d OHLCV via yfinance. Returns {symbol: DataFrame}."""
        import yfinance as yf

        result: dict[str, pd.DataFrame] = {}

        # Batch download — yfinance handles multiple tickers efficiently
        batch_size = 200
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tickers_str = " ".join(batch)
            try:
                raw = yf.download(
                    tickers_str,
                    period="20d",
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                if raw.empty:
                    continue

                if len(batch) == 1:
                    # Single ticker: columns are flat (Open, High, ...)
                    sym = batch[0]
                    if not raw.empty and len(raw) >= 5:
                        result[sym] = raw
                else:
                    # Multi-ticker: MultiIndex columns (ticker, field)
                    for sym in batch:
                        try:
                            df = raw[sym].dropna(how="all")
                            if not df.empty and len(df) >= 5:
                                result[sym] = df
                        except (KeyError, TypeError):
                            continue
            except Exception:
                logger.warning("screener.batch_download_error", batch_start=i)

        logger.info("screener.downloaded", symbols_with_data=len(result))
        return result

    def _score_symbol(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        min_avg_volume: int,
        momentum_weight: float,
        volume_weight: float,
        volatility_weight: float,
    ) -> dict | None:
        """Score a single symbol. Returns candidate dict or None if filtered out."""
        df = data.get(symbol)
        if df is None or len(df) < 15:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Filter: minimum average volume
        avg_vol = float(volume.mean())
        if avg_vol < min_avg_volume:
            return None

        price = float(close.iloc[-1])
        if price <= 0 or np.isnan(price):
            return None

        # ── Momentum score (0–1) ───────────────────────────────────────
        ret_5d = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0.0
        ret_20d = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
        rsi = _compute_rsi(close)

        # Normalize components to 0–1 range
        # Stronger positive returns → higher score
        mom_ret5 = min(1.0, max(0.0, (ret_5d + 10) / 20))  # -10% to +10% → 0 to 1
        mom_ret20 = min(1.0, max(0.0, (ret_20d + 20) / 40))  # -20% to +20% → 0 to 1
        # RSI between 40-70 is ideal (not overbought, not oversold)
        mom_rsi = 1.0 - abs(rsi - 55) / 55  # Peak at RSI=55
        mom_rsi = max(0.0, mom_rsi)

        momentum_score = (mom_ret5 * 0.4 + mom_ret20 * 0.3 + mom_rsi * 0.3)

        # ── Volume score (0–1) ─────────────────────────────────────────
        vol_5d = float(volume.iloc[-5:].mean())
        vol_20d = float(volume.mean())
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
        # Higher recent volume relative to 20d average → higher score
        volume_score = min(1.0, max(0.0, (vol_ratio - 0.5) / 1.5))  # 0.5x to 2x → 0 to 1

        # ── Volatility score (0–1) ─────────────────────────────────────
        atr = _compute_atr(high, low, close)
        atr_pct = (atr / price) * 100 if price > 0 else 0
        # Higher volatility = more opportunity, but cap extreme values
        volatility_score = min(1.0, max(0.0, atr_pct / 5))  # 0% to 5% ATR → 0 to 1

        # ── Composite score ────────────────────────────────────────────
        score = (
            momentum_weight * momentum_score
            + volume_weight * volume_score
            + volatility_weight * volatility_score
        )

        return {
            "symbol": symbol,
            "score": round(score, 4),
            "momentum_score": round(momentum_score, 4),
            "volume_score": round(volume_score, 4),
            "volatility_score": round(volatility_score, 4),
            "price": round(price, 2),
            "change_5d_pct": round(ret_5d, 2),
            "avg_volume": int(avg_vol),
            "sector": _get_sector_hint(symbol),
        }


def _get_sector_hint(symbol: str) -> str:
    """Return a sector hint based on known symbols or suffix."""
    from app.risk.position_sizer import SECTOR_MAP, DEFAULT_SECTOR
    sector = SECTOR_MAP.get(symbol, "")
    if sector:
        return sector
    # Infer from exchange suffix
    suffix = symbol.split(".")[-1] if "." in symbol else ""
    if suffix in ("AS", "DE", "PA", "L"):
        return "eu_equity"
    return DEFAULT_SECTOR
