from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trader"
    postgres_user: str = "trader"
    postgres_password: str = "trader_secret"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Interactive Brokers
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497  # 7497=paper, 7496=live
    ibkr_client_id: int = 1
    ibkr_paper_trading: bool = True

    # Claude API
    anthropic_api_key: str = ""

    # News API
    news_api_key: str = ""

    # Alerts - Telegram
    telegram_bot_token: str = ""
    telegram_chat_ids: str = ""  # Comma-separated chat IDs

    # Alerts - Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""  # From address (defaults to smtp_user if empty)
    alert_email_to: str = ""

    # Authentication
    admin_username: str = "admin"
    admin_password: str = "changeme"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    # AI Position Sizing
    ai_sizing_enabled: bool = True
    ai_sizing_cache_ttl: int = 900  # 15 minutes

    # Stock Screener
    screener_enabled: bool = True
    screener_max_candidates: int = 40
    screener_min_avg_volume: int = 500_000
    screener_momentum_weight: float = 0.4
    screener_volume_weight: float = 0.3
    screener_volatility_weight: float = 0.3
    screener_include_eu: bool = True
    screener_us_max_candidates: int = 25
    screener_eu_max_candidates: int = 15
    screener_eu_min_avg_volume: int = 100_000

    # EU Trading (requires IBKR EU market data subscriptions)
    enable_eu_trading: bool = False

    # Trading Configuration
    trading_symbols: str = (
        "SPY,QQQ,AAPL,MSFT,GOOGL,NVDA,AMZN,META,"
        "TSLA,AMD,NFLX,CRM,AVGO,JPM,BAC,GS,"
        "JNJ,UNH,LLY,XOM,CVX,WMT,HD,KO,COST,"
        "IWM,EFA,VGK,DIA,XLF,XLE"
    )
    initial_capital: float = 5000.0

    # ── SWING TRADING CONFIGURATION ──
    # Hold positions for days/weeks, follow established daily trends.
    # Fewer trades (0-2/day), wider stops, no forced EOD close.

    max_daily_loss_pct: float = 5.0          # Wider daily drawdown for swing holds
    max_position_pct: float = 12.0           # Larger positions, fewer concurrent
    max_open_positions: int = 5              # Concentrated portfolio
    min_cash_reserve_pct: float = 15.0       # Swing uses more capital
    max_sector_concentration_pct: float = 35.0
    max_total_exposure_pct: float = 80.0     # More capital deployed (held longer)
    confidence_threshold: float = 0.68       # Selective: raised from 0.60 to cut low-conviction noise
    max_hourly_loss_pct: float = 3.0         # Wider hourly tolerance for swing

    # Daily-loss halt behaviour — enforced continuously, not just on new signals
    daily_loss_force_close: bool = True      # Force-close all positions when daily loss halt fires

    # Absolute notional cap per position (0 = disabled). Belt-and-suspenders
    # backstop on top of max_position_pct so a single oversized order can never
    # be placed regardless of how position size was computed.
    max_position_notional: float = 0.0

    # Short selling
    enable_short_selling: bool = True        # Profit from downtrends
    max_short_exposure_pct: float = 30.0     # Max 30% of portfolio in shorts

    # ATR-based stop-loss — daily ATR, wide stops for multi-day holds.
    # Stop must sit OUTSIDE normal daily noise or it gets tagged on ruis.
    atr_stop_multiplier: float = 2.5         # 2.5x daily ATR stop for swing
    min_stop_loss_pct: float = 3.0           # 3% minimum stop distance

    # Take-profit — 4x ATR / 4.5% floor gives ~1.5:1 R:R (realistic within hold
    # window). Was 6x/5% which was almost never reached (1 TP hit in 178 trades).
    atr_take_profit_multiplier: float = 4.0  # 4x daily ATR target
    min_take_profit_pct: float = 4.5         # 4.5% minimum target

    # Minimum risk:reward gate — reject entries whose TP/stop geometry is worse
    min_risk_reward_ratio: float = 1.5

    # Order execution
    order_fill_timeout_seconds: int = 15
    order_max_retries: int = 2
    max_slippage_pct: float = 0.5
    consecutive_loss_alert_threshold: int = 3

    # Trade management — swing: slow entries, patient exits
    min_hold_minutes: int = 480              # 8 hours minimum hold
    reentry_cooldown_minutes: int = 1440     # 1 day cooldown before re-entry
    max_trades_per_symbol_per_day: int = 1   # One trade per symbol per day

    # EOD close — DISABLED for swing trading (hold overnight)
    eod_close_enabled: bool = False          # Allow overnight holds
    eod_close_minutes_before: int = 15
    eod_use_regular_close: bool = True

    # Swing: hold up to 14 calendar days, max 2 new positions per day
    max_hold_days: int = 14                  # Force close after 2 weeks
    max_new_positions_per_day: int = 2       # Selective: max 2 new entries per day

    # Extended hours
    extended_hours_enabled: bool = False     # Trade only regular hours for swing

    # Smart entry/exit filters — relaxed for swing
    opening_range_minutes: int = 30          # Skip entries in first 30 min after open (US open 14:00 UTC was worst hour: -€23k)
    breakeven_stop_trigger_pct: float = 3.0  # Move stop to breakeven at +3%
    stale_position_hours: float = 0          # Disabled — let swing trades develop
    stale_position_min_pnl_pct: float = 1.0
    partial_profit_enabled: bool = True      # Take 50% at first TP target
    min_relative_volume: float = 1.0         # Relaxed volume filter for swing

    # Smart execution
    smart_execution_enabled: bool = True
    vwap_duration_minutes: int = 15
    twap_slices: int = 4

    # Cycle interval (minutes) — how often the engine checks for signals
    cycle_interval_minutes: int = 60         # Hourly cycles for swing trading

    # Progressive trailing stop tiers — wider for swing moves
    trailing_stop_tiers: str = "5.0:2.0,8.0:3.0,12.0:4.0,20.0:5.0"

    # Symbol blacklist: comma-separated symbols to never trade
    # Blacklisted biggest losers from historical performance analysis
    symbol_blacklist: str = "SHOP,DDOG,LRCX,MCHP,RI.PA,SAP.DE"

    # Per-strategy symbol allowlist for ml_xgboost.
    # Walk-forward showed the model has edge on SPY/QQQ/NVDA but loses on
    # AAPL/MSFT. Empty string = no filter (legacy behaviour).
    ml_xgboost_allowed_symbols: str = "SPY,QQQ,NVDA"

    # Triple-barrier labelling for ML training — label entries by whether the
    # take-profit is hit before the stop within the hold window (matches live
    # exit geometry). Default off; enable to retrain a barrier-aware model, then
    # validate before relying on it. Uses min_take_profit_pct / min_stop_loss_pct.
    ml_use_triple_barrier: bool = False

    @property
    def symbol_blacklist_set(self) -> set[str]:
        """Parse symbol_blacklist string into a set."""
        if not self.symbol_blacklist:
            return set()
        return {s.strip() for s in self.symbol_blacklist.split(",") if s.strip()}

    @property
    def ml_xgboost_allowed_symbols_set(self) -> set[str]:
        """Parse ml_xgboost_allowed_symbols string into a set. Empty = no filter."""
        if not self.ml_xgboost_allowed_symbols:
            return set()
        return {s.strip() for s in self.ml_xgboost_allowed_symbols.split(",") if s.strip()}

    @property
    def trailing_stop_tiers_parsed(self) -> list[tuple[float, float]]:
        """Parse trailing_stop_tiers string into sorted list of (gain_pct, trail_pct) tuples."""
        tiers = []
        for pair in self.trailing_stop_tiers.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            gain_s, trail_s = pair.split(":", 1)
            tiers.append((float(gain_s), float(trail_s)))
        tiers.sort(key=lambda t: t[0], reverse=True)  # highest gain first for matching
        return tiers

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def symbols_list(self) -> list[str]:
        return [s.strip() for s in self.trading_symbols.split(",") if s.strip()]


settings = Settings()
