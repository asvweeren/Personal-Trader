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
    max_daily_loss_pct: float = 5.0
    max_position_pct: float = 5.0
    max_open_positions: int = 10
    min_cash_reserve_pct: float = 20.0
    max_sector_concentration_pct: float = 35.0
    max_total_exposure_pct: float = 100.0  # Max total open notional as % of portfolio value
    confidence_threshold: float = 0.40     # Min confidence for strategies to generate BUY/SELL

    # ATR-based stop-loss
    atr_stop_multiplier: float = 1.5       # ATR multiplier for stop-loss distance
    min_stop_loss_pct: float = 1.0         # Minimum stop-loss percentage as floor

    # Take-profit
    atr_take_profit_multiplier: float = 2.0  # Take-profit at 2x ATR above entry
    min_take_profit_pct: float = 1.0         # Minimum 1.0% profit target (day trading)

    # Order execution
    order_fill_timeout_seconds: int = 15     # Max seconds to wait for market order fill
    order_max_retries: int = 2               # Max retry attempts for failed market orders
    max_slippage_pct: float = 0.5            # Alert when slippage exceeds this %
    consecutive_loss_alert_threshold: int = 5  # Alert after N consecutive losing trades

    # Trade management
    min_hold_minutes: int = 5                # Min hold time before SELL signal can close
    reentry_cooldown_minutes: int = 10       # Min wait time before re-entering same symbol
    max_trades_per_symbol_per_day: int = 2   # Max trades per symbol per day

    # End-of-day close
    eod_close_minutes_before: int = 10       # Close all positions 10 min before market close

    # Smart execution
    smart_execution_enabled: bool = True
    vwap_duration_minutes: int = 15
    twap_slices: int = 4

    # Progressive trailing stop tiers: "gain%:trail%,..."
    # Start trailing early with tight trail widths for day trading
    trailing_stop_tiers: str = "2.0:0.8,3.0:1.2,5.0:1.8,8.0:2.5"

    # Symbol blacklist: comma-separated symbols to never trade
    symbol_blacklist: str = ""

    @property
    def symbol_blacklist_set(self) -> set[str]:
        """Parse symbol_blacklist string into a set."""
        if not self.symbol_blacklist:
            return set()
        return {s.strip() for s in self.symbol_blacklist.split(",") if s.strip()}

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
