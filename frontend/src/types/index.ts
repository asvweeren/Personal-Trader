export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  market_price: number;
  market_value: number;
  unrealized_pnl: number;
}

export interface Portfolio {
  total_value: number;
  cash: number;
  buying_power: number;
  unrealized_pnl: number;
  realized_pnl: number;
  positions: Position[];
}

export interface Trade {
  id: number;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  entry_price: number | null;
  exit_price: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  status: "PENDING" | "OPEN" | "CLOSED" | "CANCELLED";
  strategy_name: string;
  signal_id?: number | null;
  realized_pnl: number | null;
  commission?: number | null;
  created_at: string | null;
  updated_at?: string | null;
  closed_at: string | null;
}

export interface TradesResponse {
  trades: Trade[];
  total: number;
  skip: number;
  limit: number;
}

export interface Performance {
  total_value: number;
  initial_capital: number;
  realized_pnl: number;
  unrealized_pnl: number;
  daily_pnl: number;
  total_return_pct: number;
  daily_return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  profit_factor: number | string;
  max_drawdown: number;
  total_commission: number;
}

export interface RiskHealth {
  healthy: boolean;
  checks: Record<string, boolean>;
  warnings: string[];
  daily_loss_pct: number;
  cash_reserve_pct: number;
  position_count: number;
  max_drawdown_pct: number;
  sector_exposure: Record<string, number>;
  largest_position_pct: number;
  market_open: boolean;
}

export interface RiskMetricsData {
  health: RiskHealth;
  limits: {
    max_daily_loss_pct: number;
    max_position_pct: number;
    max_open_positions: number;
    min_cash_reserve_pct: number;
  };
  daily_loss_triggered: boolean;
}

export interface RiskEvent {
  id: number;
  event_type: string;
  severity: string;
  symbol: string | null;
  description: string;
  action_taken: string | null;
  portfolio_value: number | null;
  daily_loss_pct: number | null;
  timestamp: string | null;
}

export interface StrategyConfig {
  active_strategies: string[];
  confidence_threshold: number;
  ensemble_method: string;
  weights: Record<string, number>;
  trading_enabled: boolean;
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "down";
  timestamp: string;
  components: {
    broker: { status: string };
    database: { status: string };
    trading_engine: { status: string };
    environment: string;
    paper_trading: boolean;
  };
}

export interface EngineStatus {
  state: string;
  trading_enabled: boolean;
  cycle_count: number;
  last_cycle_at: string | null;
  open_trades: number;
  pending_orders: number;
  symbols: string[];
  strategies: string[];
  reconnect_attempts: number;
}

export interface PortfolioSnapshot {
  id: number;
  total_value: number;
  cash: number;
  positions_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  daily_pnl: number;
  timestamp: string | null;
}

export interface BacktestRequest {
  strategy_name: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  commission_pct?: number;
  slippage_pct?: number;
  max_position_pct?: number;
  stop_loss_pct?: number;
  params?: Record<string, unknown>;
}

export interface BacktestSummary {
  id: number;
  strategy_name: string;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
  created_at: string | null;
}

export interface BacktestDetail extends BacktestSummary {
  trades_summary: BacktestTrade[];
  equity_curve: { timestamp: string; equity: number }[];
}

export interface BacktestTrade {
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  exit_price: number;
  pnl: number;
  commission: number;
  exit_reason: string;
  bars_held: number;
}

export interface BacktestsResponse {
  backtests: BacktestSummary[];
  total: number;
}

export interface WSMessage {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}
