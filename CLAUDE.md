# AI Day Trading System - Project Context

## Overview

Fully automated AI day trading system that trades US/EU stocks and ETFs via Interactive Brokers. Uses ML models (XGBoost) and LLM sentiment analysis (Claude API) to generate trading signals. Designed for <€5,000 starting capital with strict risk management. Paper trading must be validated before going live.

**Live deployment**: https://trader.edgedigital.nl
**Repository**: https://github.com/asvweeren/Personal-Trader.git

---

## Architecture

```
React Dashboard (Vite + TypeScript + TailwindCSS)
        │ REST + WebSocket
FastAPI Backend (Python 3.12, async)
        │
   ┌────┼─────────────┐
   │    │              │
Trading  ML Engine    Risk Manager
Engine   (XGBoost +   (Hard limits +
         Sentiment)   AI-driven sizing)
   │         │              │
   ▼         ▼              ▼
IBKR     Data Pipeline   PostgreSQL
Adapter  (Market data,   + Redis
         Indicators,
         News/Sentiment)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Alembic, APScheduler |
| Frontend | React 19, TypeScript, Vite, TailwindCSS, Recharts |
| ML | XGBoost (3-class classifier: BUY/HOLD/SELL), scikit-learn |
| Sentiment | Claude API (Anthropic) for news sentiment analysis |
| Broker | Interactive Brokers via ib_insync |
| Database | PostgreSQL 16, Redis 7 |
| Deployment | Docker Compose, Nginx (reverse proxy + SSL), Let's Encrypt |
| Alerts | Telegram Bot + Email (Resend SMTP) |

## Project Structure

```
trader/
├── CLAUDE.md                    ← You are here
├── docker-compose.yml           ← Local development
├── docker-compose.prod.yml      ← Production deployment
├── Makefile                     ← Dev shortcuts
├── .env.example                 ← Environment template
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml           ← Dependencies (uv)
│   ├── alembic.ini
│   ├── alembic/versions/        ← DB migrations (6 versions)
│   └── app/
│       ├── main.py              ← FastAPI app + lifespan
│       ├── config.py            ← Settings (pydantic-settings, reads .env)
│       ├── dependencies.py      ← Singletons + strategy loader
│       ├── api/
│       │   ├── auth.py          ← JWT authentication (login endpoint)
│       │   ├── websocket.py     ← WebSocket + ConnectionManager
│       │   └── routes/
│       │       ├── portfolio.py ← GET /portfolio, /positions, /performance
│       │       ├── trades.py    ← GET /trades (paginated)
│       │       ├── strategy.py  ← GET/PUT /strategy/config
│       │       ├── risk.py      ← GET /risk/metrics, PUT /risk/limits
│       │       ├── backtest.py  ← POST /backtest/run (yfinance fallback)
│       │       ├── system.py    ← GET /health, /engine, POST /test-alert
│       │       ├── validation.py← Paper trading validation endpoints
│       │       └── screener.py ← Stock screener endpoints
│       ├── broker/
│       │   ├── base.py          ← Abstract BrokerAdapter interface
│       │   ├── ibkr_adapter.py  ← Interactive Brokers implementation
│       │   └── mock_adapter.py  ← Mock for tests (APP_ENV=test)
│       ├── data/
│       │   ├── market_data.py   ← OHLCV data from IBKR
│       │   ├── indicators.py    ← RSI, MACD, BB, ATR, OBV, VWAP
│       │   ├── news_fetcher.py  ← RSS feeds + NewsAPI
│       │   ├── sentiment.py     ← Claude API sentiment scoring
│       │   ├── feature_store.py ← Redis-cached feature store
│       │   ├── pipeline.py      ← Orchestrates data refresh
│       │   └── screener.py      ← Daily stock screener (S&P 500 + EU)
│       ├── strategy/
│       │   ├── base.py          ← Abstract Strategy interface
│       │   ├── ml_strategy.py   ← XGBoost classifier
│       │   ├── sentiment_strategy.py ← Claude LLM-based signals
│       │   ├── ensemble.py      ← Weighted ensemble (60% ML, 40% sentiment)
│       │   ├── nn_strategy.py   ← PyTorch neural net (placeholder)
│       │   └── feature_pipeline.py ← Feature engineering for ML
│       ├── risk/
│       │   ├── manager.py       ← Risk evaluation + portfolio health
│       │   ├── hard_limits.py   ← Non-overridable safety limits
│       │   ├── position_sizer.py← Kelly criterion + vol-adjusted sizing
│       │   └── market_hours.py  ← US/EU market hours check
│       ├── execution/
│       │   ├── engine.py        ← Trading loop (5-min cycle, 441 lines)
│       │   ├── order_manager.py ← Order lifecycle management
│       │   └── portfolio_tracker.py ← Position & P&L tracking
│       ├── backtest/
│       │   ├── engine.py        ← Event-driven backtester
│       │   ├── simulator.py     ← Market simulator with slippage
│       │   └── metrics.py       ← Sharpe, Sortino, drawdown, etc.
│       ├── monitoring/
│       │   ├── alerts.py        ← Telegram + Email alert sender
│       │   ├── daily_reporter.py← Daily paper-trading summary
│       │   ├── performance.py   ← P&L tracking
│       │   ├── logger.py        ← structlog setup
│       │   └── paper_trading_validator.py ← Validation framework
│       ├── models/              ← SQLAlchemy models
│       │   ├── database.py, trade.py, order.py, signal.py
│       │   ├── portfolio_snapshot.py, backtest_result.py
│       │   ├── risk_event.py, market_data_bar.py
│       │   ├── validation_report.py
│       │   └── screening_result.py
│       └── core/
│           ├── event_bus.py     ← Async pub/sub for internal events
│           ├── scheduler.py     ← APScheduler jobs
│           └── exceptions.py    ← Custom exceptions
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx              ← Root component + ErrorBoundary
│       ├── main.tsx             ← Entry point
│       ├── api/client.ts        ← API client (fetchApi + auth)
│       ├── contexts/AuthContext.tsx ← JWT auth context
│       ├── hooks/
│       │   ├── usePortfolio.ts  ← Polls portfolio/performance/risk
│       │   └── useWebSocket.ts  ← WS connection with auto-reconnect
│       ├── types/index.ts       ← All TypeScript interfaces
│       └── components/
│           ├── Dashboard.tsx     ← Main layout with tabs
│           ├── LoginPage.tsx     ← Auth form
│           ├── PortfolioCard.tsx ← Portfolio overview
│           ├── PnLChart.tsx      ← Recharts area/line chart
│           ├── RiskMetrics.tsx   ← Risk bars + warnings
│           ├── EngineControl.tsx ← Start/stop trading
│           ├── PositionList.tsx  ← Open positions
│           ├── TradeTable.tsx    ← Trade history + pagination
│           ├── SignalFeed.tsx    ← Real-time WebSocket events
│           ├── SystemStatus.tsx  ← Status indicators in header
│           ├── BacktestPanel.tsx ← Run & view backtests
│           ├── ValidationDashboard.tsx ← Paper trading validation
│           ├── ScreenerPanel.tsx ← Daily stock screener results
│           └── SettingsPage.tsx  ← Risk limits, strategy config, alerts
│
├── ml/
│   ├── models/                  ← Trained models (gitignored on server)
│   │   ├── xgboost_model.pkl   ← Trained XGBoost 3-class classifier
│   │   └── xgboost_model.json  ← Model metadata + feature columns
│   └── notebooks/               ← Jupyter notebooks for exploration
│
├── nginx/
│   ├── nginx-ssl.conf           ← Production nginx config (SSL + proxy)
│   └── ip-whitelist.conf        ← IP access restrictions
│
└── scripts/
    ├── deploy.sh                ← One-click server deployment
    ├── train_model.py           ← XGBoost training pipeline
    ├── download_historical_data.py
    ├── run_backtest.py
    └── optimize_model.py
```

---

## Server / Deployment

### Production Server
- **Host**: 161.97.72.153 (Contabo VPS, Ubuntu 24.04)
- **SSH**: `ssh trader-server` (uses key `~/.ssh/trader_server`, configured in `~/.ssh/config`)
- **Domain**: trader.edgedigital.nl (SSL via Let's Encrypt)
- **Install dir**: `/root/trader`

### Credentials (configured in server .env)
All credentials are stored in `/root/trader/.env` on the server. Do NOT commit secrets to git.
- **Dashboard login**: admin / (see ADMIN_PASSWORD in .env)
- **Anthropic API**: configured in .env as ANTHROPIC_API_KEY
- **Telegram Bot**: @EdgeDigitalTraderBot (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_IDS in .env)
- **Email**: Resend SMTP on smtp.resend.com:465, from trader@edgedigital.nl to info@edgedigital.nl (SMTP_* in .env)
- **IBKR**: NOT YET CONFIGURED - waiting on user's credentials

### SSH Config (macOS)
An SSH key pair is configured at `~/.ssh/trader_server` with a host alias in `~/.ssh/config`:
```
Host trader-server
    HostName 161.97.72.153
    User root
    IdentityFile ~/.ssh/trader_server
```
Usage: `ssh trader-server "command"` — no password needed.

### Common Server Commands
```bash
# Run commands remotely via SSH
ssh trader-server "cd /root/trader && git pull origin master"
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml up -d --build backend frontend-build"
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml restart nginx"

# Run Alembic migrations
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml exec -T -w /app backend uv run alembic upgrade head"

# View logs
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml logs -f backend"
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml logs --tail 50 backend"

# Service status
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml ps"

# Restart everything
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml down && docker compose -f docker-compose.prod.yml up -d"

# DB shell
ssh trader-server "cd /root/trader && docker compose -f docker-compose.prod.yml exec -T postgres psql -U trader -d trader"
```

### Full deploy (one-liner)
```bash
git push origin master && ssh trader-server "cd /root/trader && git pull origin master && docker compose -f docker-compose.prod.yml up -d --build backend frontend-build && docker compose -f docker-compose.prod.yml exec -T -w /app backend uv run alembic upgrade head && docker compose -f docker-compose.prod.yml restart nginx"
```

### File uploads
```bash
scp -i ~/.ssh/trader_server localfile root@161.97.72.153:/root/trader/path
```

---

## Local Development

### Prerequisites
- Python 3.12+ with `uv` package manager
- Node.js 20+ with npm
- Docker & Docker Compose (for PostgreSQL, Redis)

### Setup
```bash
# Start DB + Redis
docker compose up -d postgres redis

# Backend
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

### Run Tests
```bash
cd backend
uv run pytest tests/ -x -q      # 250 tests, ~3 seconds
```

Tests use `APP_ENV=test` which automatically uses MockBrokerAdapter instead of IBKR.

---

## Key Design Decisions

### Strategy Loading (dependencies.py)
Strategies are loaded at startup in `load_strategies()`:
1. **MLStrategy** - loads XGBoost model from `ml/models/xgboost_model.pkl`
2. **SentimentStrategy** - uses Claude API (only if ANTHROPIC_API_KEY is set)
3. **EnsembleStrategy** - wraps both with weights 60%/40% (only if 2+ strategies available)

### Broker Graceful Degradation
All API endpoints handle broker disconnection gracefully:
- Portfolio returns initial capital defaults when broker is down
- Risk metrics returns "broker not connected" warnings
- Health endpoint shows "degraded" status
- The app fully functions without IBKR (dashboard, backtests, settings all work)

### Backtest yfinance Fallback
When IBKR is not connected, backtests download historical data from yfinance instead. This makes backtests work without any broker credentials.

### Trading Engine Flow (every 5 minutes)
1. Check market hours (US/EU)
2. Refresh market data + indicators
3. Generate signals from all strategies
4. Risk manager evaluates each signal
5. Place orders with automatic stop-loss
6. Track order fills and update portfolio
7. Broadcast updates via WebSocket

### Hard Risk Limits (non-overridable)
- Max 5% daily loss → halts all trading
- Max 20% portfolio per position
- Max 10 open positions
- Min 30% cash reserve
- No trading outside market hours

### Authentication
- JWT tokens (HS256, 24h expiry)
- Single admin user (credentials in .env)
- All API routes require auth except `/api/auth/login`
- WebSocket validates token in query parameter

### Alert Channels
- **Telegram**: via Bot API, sends HTML-formatted messages
- **Email**: via Resend SMTP (port 465, SSL), from trader@edgedigital.nl

---

## Current State (Feb 2026)

### What's Working
- Full dashboard with all tabs (Overview, Trades, Screener, Backtest, Validation, Settings)
- JWT authentication with login/logout
- Portfolio view, risk metrics, engine control (all gracefully handle missing broker)
- Backtests with yfinance data (11 symbols tested: SPY, QQQ, AAPL, MSFT, GOOGL, NVDA, AMZN, META, IWM, EFA, VGK)
- WebSocket with 30s heartbeat
- Telegram + Email alerts (tested and working)
- 250 backend tests passing
- Trained XGBoost model deployed
- SSL certificate on trader.edgedigital.nl
- Daily validation report scheduler (21:00 UTC)
- Daily stock screener: S&P 500 + EU blue chips, scored on momentum/volume/volatility (07:50 UTC)

### What's NOT Working / TODO
- **IBKR credentials not configured** - waiting on user. Once provided:
  - Set IBKR_USERNAME, IBKR_PASSWORD in server .env
  - Restart backend: `docker compose -f docker-compose.prod.yml restart backend`
  - Connect via VNC (port 5900) for 2FA if needed
  - Enable paper trading from the dashboard
- **Resend domain verification** - DNS records for edgedigital.nl need to be set up for email deliverability
- **Model retraining** - current model trained on historical data; should be retrained periodically
- **Paper trading validation** - need 4+ weeks of paper trading data before going live
- **Neural net strategy** (nn_strategy.py) - placeholder, not yet implemented

### Known Limitations
- Negative Sharpe ratios on backtests (few trades, conservative strategy)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login, returns JWT |
| GET | `/api/portfolio` | Portfolio + positions |
| GET | `/api/positions` | Open positions only |
| GET | `/api/performance` | P&L metrics |
| GET | `/api/portfolio/snapshots` | Historical snapshots |
| GET | `/api/trades?skip=0&limit=50` | Trade history (paginated) |
| GET | `/api/risk/metrics` | Risk health + limits |
| PUT | `/api/risk/limits` | Update risk limits |
| GET | `/api/strategy/status` | Strategy config |
| PUT | `/api/strategy/config` | Update strategy |
| POST | `/api/backtest/run` | Start backtest |
| GET | `/api/backtest/{id}` | Backtest results |
| GET | `/api/backtests` | List backtests |
| GET | `/api/system/health` | Health check |
| GET | `/api/system/engine` | Engine status |
| PUT | `/api/system/engine/trading` | Enable/disable trading |
| POST | `/api/system/test-alert` | Send test alert |
| GET | `/api/validation/*` | Validation endpoints |
| GET | `/api/screener/latest` | Latest screening result + candidates |
| GET | `/api/screener/history?days=7` | Historical screenings |
| POST | `/api/screener/run` | Manual screening trigger |
| WS | `/ws/live?token=...` | Real-time updates |

---

## Database Schema

Tables managed by Alembic (4 migrations):
- **trades** - Trade records (symbol, side, qty, prices, status, strategy, P&L)
- **orders** - Broker orders linked to trades
- **signals** - Generated trading signals with confidence + features snapshot (JSONB)
- **portfolio_snapshots** - Periodic portfolio value snapshots
- **risk_events** - Risk limit trigger events
- **backtest_results** - Backtest configs + results (JSONB metrics, equity curve)
- **market_data_bars** - OHLCV bar data
- **validation_reports** - Paper trading validation reports
- **screening_results** - Daily stock screener results (JSONB candidates, config)

---

## Event Bus Events

Internal async pub/sub events (broadcast to WebSocket):
- `signal.generated` - New trading signal
- `order.placed` / `order.filled` / `order.cancelled`
- `position.closed` - With realized P&L
- `portfolio.updated` - Portfolio value change
- `risk.daily_stop` - Daily loss limit hit
- `risk.warning` - Risk threshold warning
- `engine.state_change` - Engine started/stopped
- `engine.cycle` - Trading cycle completed
- `system.heartbeat` - 30s system status pulse
