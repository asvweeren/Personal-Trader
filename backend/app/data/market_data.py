from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter
from app.core.event_bus import MARKET_DATA_UPDATE, event_bus
from app.models.market_data_bar import MarketDataBar

logger = structlog.get_logger()


@dataclass
class Tick:
    """A single price tick from the broker."""

    symbol: str
    price: float
    volume: int
    timestamp: datetime


@dataclass
class BarAccumulator:
    """Accumulates ticks into OHLCV bars."""

    symbol: str
    bar_size_minutes: int
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0
    bar_start: datetime | None = None

    def add_tick(self, tick: Tick) -> "BarAccumulator | None":
        """Add a tick. Returns a completed bar if the time window has passed, else None."""
        now = tick.timestamp
        if self.bar_start is None:
            self._start_new_bar(tick)
            return None

        bar_end = self.bar_start + timedelta(minutes=self.bar_size_minutes)
        if now >= bar_end:
            # Current bar is complete, return it and start new
            completed = BarAccumulator(
                symbol=self.symbol,
                bar_size_minutes=self.bar_size_minutes,
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close,
                volume=self.volume,
                tick_count=self.tick_count,
                bar_start=self.bar_start,
            )
            self._start_new_bar(tick)
            return completed

        # Update current bar
        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.volume
        self.tick_count += 1
        return None

    def _start_new_bar(self, tick: Tick) -> None:
        self.open = tick.price
        self.high = tick.price
        self.low = tick.price
        self.close = tick.price
        self.volume = tick.volume
        self.tick_count = 1
        # Align bar start to the bar size boundary
        minutes = tick.timestamp.minute - (tick.timestamp.minute % self.bar_size_minutes)
        self.bar_start = tick.timestamp.replace(minute=minutes, second=0, microsecond=0)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.bar_start,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class MarketSnapshot:
    """A snapshot of market data for multiple symbols at a point in time."""

    timestamp: datetime
    prices: dict[str, float]
    ohlcv: dict[str, pd.DataFrame]
    features: dict[str, dict]
    computed_features_df: dict[str, pd.DataFrame] = field(default_factory=dict)
    intraday_features: dict[str, dict] = field(default_factory=dict)


class MarketDataService:
    """Manages market data collection, real-time streaming, bar aggregation, and persistence."""

    def __init__(
        self,
        broker: BrokerAdapter,
        db: AsyncSession | None = None,
        bar_size_minutes: int = 60,
    ):
        self._broker = broker
        self._db = db
        self._bar_size_minutes = bar_size_minutes
        self._current_prices: dict[str, float] = {}
        self._historical_cache: dict[str, pd.DataFrame] = {}
        self._cache_timestamps: dict[str, datetime] = {}
        self._bar_accumulators: dict[str, BarAccumulator] = {}
        self._recent_bars: dict[str, list[dict]] = defaultdict(list)
        self._streaming = False
        self._cache_ttl = timedelta(minutes=5)

    # ── Real-time streaming ──────────────────────────────────────────

    async def start_streaming(self, symbols: list[str]) -> None:
        """Subscribe to real-time market data from the broker."""
        for symbol in symbols:
            self._bar_accumulators[symbol] = BarAccumulator(
                symbol=symbol, bar_size_minutes=self._bar_size_minutes
            )

        await self._broker.subscribe_market_data(symbols, self._on_tick)
        self._streaming = True
        logger.info("market_data.streaming_started", symbols=symbols)

    async def stop_streaming(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time market data."""
        await self._broker.unsubscribe_market_data(symbols)
        self._streaming = False
        logger.info("market_data.streaming_stopped")

    async def _on_tick(self, data: dict) -> None:
        """Callback for incoming ticks from the broker."""
        symbol = data.get("symbol", "")
        price = data.get("price", 0.0)
        volume = data.get("volume", 0)

        if not symbol or price <= 0:
            return

        tick = Tick(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=datetime.now(UTC),
        )

        # Update current price
        self._current_prices[symbol] = price

        # Aggregate into bars
        accumulator = self._bar_accumulators.get(symbol)
        if accumulator:
            completed_bar = accumulator.add_tick(tick)
            if completed_bar:
                bar_dict = completed_bar.to_dict()
                self._recent_bars[symbol].append(bar_dict)
                # Keep only last 500 bars in memory
                if len(self._recent_bars[symbol]) > 500:
                    self._recent_bars[symbol] = self._recent_bars[symbol][-500:]
                # Persist to DB
                await self._persist_bar(completed_bar)
                # Invalidate cache for this symbol
                self._invalidate_cache(symbol)

        # Broadcast update
        await event_bus.publish(MARKET_DATA_UPDATE, {
            "symbol": symbol,
            "price": price,
            "volume": volume,
        })

    # ── Price data ───────────────────────────────────────────────────

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get latest known prices for symbols."""
        return {s: self._current_prices.get(s, 0.0) for s in symbols}

    def update_price(self, symbol: str, price: float) -> None:
        """Manually update a price (useful for non-streaming scenarios)."""
        self._current_prices[symbol] = price

    # ── Historical data ──────────────────────────────────────────────

    async def get_historical_data(
        self, symbol: str, duration: str = "30 D", bar_size: str = "1 hour"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data. First checks DB, then in-memory cache,
        then fetches from broker and persists.
        """
        cache_key = f"{symbol}_{duration}_{bar_size}"

        # Check in-memory cache with TTL
        if cache_key in self._historical_cache:
            cached_at = self._cache_timestamps.get(cache_key)
            if cached_at and datetime.now(UTC) - cached_at < self._cache_ttl:
                return self._historical_cache[cache_key]

        # Try loading from DB
        if self._db:
            df = await self._load_from_db(symbol, duration, bar_size)
            if not df.empty:
                self._historical_cache[cache_key] = df
                self._cache_timestamps[cache_key] = datetime.now(UTC)
                return df

        # Fetch from broker
        df = await self._broker.get_historical_data(symbol, duration, bar_size)
        if not df.empty:
            df = self.validate_data(df)
            self._historical_cache[cache_key] = df
            self._cache_timestamps[cache_key] = datetime.now(UTC)
            # Persist to DB in background
            if self._db:
                await self._persist_dataframe(symbol, df, bar_size)

        return df

    async def download_and_store(
        self, symbol: str, duration: str = "1 Y", bar_size: str = "1 hour"
    ) -> int:
        """Download historical data from broker and store in DB. Returns number of bars stored."""
        df = await self._broker.get_historical_data(symbol, duration, bar_size)
        if df.empty:
            return 0
        await self._persist_dataframe(symbol, df, bar_size)
        logger.info("market_data.downloaded", symbol=symbol, bars=len(df))
        return len(df)

    # ── Snapshots ────────────────────────────────────────────────────

    @staticmethod
    def _drop_incomplete_daily_bar(df: pd.DataFrame) -> pd.DataFrame:
        """Drop the current day's incomplete bar from daily OHLCV data.

        IBKR returns a partial bar for the current trading day whose volume
        (and high/low) are not yet final.  The ML model was trained on
        completed daily bars, so using the partial bar introduces feature
        drift (e.g. volume_ratio ≈ 0.2 instead of ~1.0) and causes the
        model to generate spurious SELL signals.
        """
        if df.empty or "timestamp" not in df.columns:
            return df
        today = datetime.now(UTC).date()
        last_ts = df["timestamp"].iloc[-1]
        # Handle datetime.date, datetime.datetime, and string dates
        if isinstance(last_ts, datetime):
            last_date = last_ts.date()
        elif isinstance(last_ts, date):
            last_date = last_ts
        elif isinstance(last_ts, str):
            last_date = pd.Timestamp(last_ts).date()
        else:
            return df
        if last_date == today:
            return df.iloc[:-1].reset_index(drop=True)
        return df

    async def get_snapshot(self, symbols: list[str]) -> MarketSnapshot:
        """Get a complete market snapshot with prices and OHLCV data for all symbols."""
        prices = await self.get_current_prices(symbols)

        # Fetch daily and intraday bars for all symbols in parallel
        async def _fetch_daily(symbol: str):
            df = await self.get_historical_data(symbol, duration="1 Y", bar_size="1 day")
            df = self._drop_incomplete_daily_bar(df)
            return symbol, df

        async def _fetch_intraday(symbol: str):
            df = await self.get_historical_data(symbol, duration="1 D", bar_size="5 mins")
            return symbol, df

        import asyncio
        daily_results, intraday_results = await asyncio.gather(
            asyncio.gather(*[_fetch_daily(s) for s in symbols], return_exceptions=True),
            asyncio.gather(*[_fetch_intraday(s) for s in symbols], return_exceptions=True),
        )

        ohlcv = {}
        for result in daily_results:
            if isinstance(result, Exception):
                logger.warning("market_data.fetch_error", error=str(result))
                continue
            symbol, df = result
            ohlcv[symbol] = df
            if (not prices.get(symbol)) and not df.empty:
                prices[symbol] = float(df["close"].iloc[-1])

        # Pre-compute daily features once for all symbols — avoids 3-4x redundant
        # compute_features() calls per symbol per cycle across strategies
        computed_features_df: dict[str, pd.DataFrame] = {}
        from app.data.indicators import compute_features, compute_intraday_features

        for symbol, df in ohlcv.items():
            if df.empty or len(df) < 50:
                continue
            try:
                computed_features_df[symbol] = compute_features(df)
            except Exception:
                logger.warning("market_data.compute_features_error", symbol=symbol)

        # Compute intraday features from 5-min bars
        intraday_features: dict[str, dict] = {}
        for result in intraday_results:
            if isinstance(result, Exception):
                logger.debug("market_data.intraday_fetch_error", error=str(result))
                continue
            symbol, df = result
            if df.empty or len(df) < 14:
                continue
            try:
                features = compute_intraday_features(df)
                if features:
                    intraday_features[symbol] = features
            except Exception:
                logger.debug("market_data.intraday_features_error", symbol=symbol)

        return MarketSnapshot(
            timestamp=datetime.now(UTC),
            prices=prices,
            ohlcv=ohlcv,
            features={},
            computed_features_df=computed_features_df,
            intraday_features=intraday_features,
        )

    # ── Data quality validation ─────────────────────────────────────

    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean historical data. Returns cleaned DataFrame."""
        if df.empty:
            return df

        # Remove rows where all OHLCV are NaN
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        existing_cols = [c for c in ohlcv_cols if c in df.columns]
        df = df.dropna(subset=existing_cols, how="all")

        if df.empty:
            return df

        # Fix OHLC consistency: high must be >= max(open, close), low <= min(open, close)
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

        # Remove zero/negative price rows
        if "close" in df.columns:
            df = df[df["close"] > 0]

        # Remove duplicate timestamps
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # ── Cache management ─────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._historical_cache.clear()
        self._cache_timestamps.clear()

    def _invalidate_cache(self, symbol: str) -> None:
        """Invalidate cache entries for a specific symbol."""
        keys_to_remove = [k for k in self._historical_cache if k.startswith(f"{symbol}_")]
        for k in keys_to_remove:
            self._historical_cache.pop(k, None)
            self._cache_timestamps.pop(k, None)

    # ── DB persistence ───────────────────────────────────────────────

    async def _persist_bar(self, bar: BarAccumulator) -> None:
        """Persist a single completed bar to the database."""
        if not self._db or not bar.bar_start:
            return
        try:
            size_map = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h", 1440: "1d"}
            bar_size = size_map.get(bar.bar_size_minutes, f"{bar.bar_size_minutes}m")
            db_bar = MarketDataBar(
                symbol=bar.symbol,
                timestamp=bar.bar_start,
                bar_size=bar_size,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            self._db.add(db_bar)
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            logger.debug("market_data.persist_bar_duplicate", symbol=bar.symbol)

    async def _persist_dataframe(self, symbol: str, df: pd.DataFrame, bar_size: str) -> None:
        """Persist a DataFrame of bars to the database."""
        if not self._db:
            return
        size_map = {
            "1 min": "1m", "5 mins": "5m",
            "15 mins": "15m", "1 hour": "1h",
            "4 hours": "4h", "1 day": "1d",
        }
        db_size = size_map.get(bar_size, bar_size)

        count = 0
        for _, row in df.iterrows():
            try:
                db_bar = MarketDataBar(
                    symbol=symbol,
                    timestamp=row["timestamp"],
                    bar_size=db_size,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                self._db.add(db_bar)
                count += 1
            except Exception:
                continue

        try:
            await self._db.commit()
            logger.info("market_data.persisted", symbol=symbol, bars=count)
        except Exception:
            await self._db.rollback()
            logger.warning("market_data.persist_error", symbol=symbol)

    async def _load_from_db(
        self, symbol: str, duration: str = "30 D", bar_size: str = "1 hour"
    ) -> pd.DataFrame:
        """Load historical bars from the database."""
        if not self._db:
            return pd.DataFrame()

        # Parse duration string (e.g., "30 D", "1 Y")
        parts = duration.strip().split()
        amount = int(parts[0])
        unit = parts[1].upper() if len(parts) > 1 else "D"
        if unit.startswith("Y"):
            start_date = datetime.now(UTC) - timedelta(days=amount * 365)
        elif unit.startswith("M"):
            start_date = datetime.now(UTC) - timedelta(days=amount * 30)
        else:
            start_date = datetime.now(UTC) - timedelta(days=amount)

        size_map = {
            "1 min": "1m", "5 mins": "5m",
            "15 mins": "15m", "1 hour": "1h",
            "4 hours": "4h", "1 day": "1d",
        }
        db_size = size_map.get(bar_size, bar_size)

        result = await self._db.execute(
            select(MarketDataBar)
            .where(
                and_(
                    MarketDataBar.symbol == symbol,
                    MarketDataBar.bar_size == db_size,
                    MarketDataBar.timestamp >= start_date,
                )
            )
            .order_by(MarketDataBar.timestamp)
        )
        bars = result.scalars().all()

        if not bars:
            return pd.DataFrame()

        data = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        return pd.DataFrame(data)
