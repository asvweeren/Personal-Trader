import type {
  Portfolio,
  Position,
  Performance,
  TradesResponse,
  Trade,
  RiskMetricsData,
  RiskEvent,
  StrategyConfig,
  SystemHealth,
  EngineStatus,
  PortfolioSnapshot,
  BacktestRequest,
  BacktestDetail,
  BacktestsResponse,
} from "../types";

const API_BASE = "/api";

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  // Portfolio
  getPortfolio: () => fetchApi<Portfolio>("/portfolio"),
  getPositions: () => fetchApi<Position[]>("/positions"),
  getPerformance: () => fetchApi<Performance>("/performance"),
  getPortfolioSnapshots: (limit = 100) =>
    fetchApi<PortfolioSnapshot[]>(`/portfolio/snapshots?limit=${limit}`),

  // Trades
  getTrades: (skip = 0, limit = 50) =>
    fetchApi<TradesResponse>(`/trades?skip=${skip}&limit=${limit}`),
  getTrade: (id: number) => fetchApi<Trade>(`/trades/${id}`),

  // Risk
  getRiskMetrics: () => fetchApi<RiskMetricsData>("/risk/metrics"),
  getRiskEvents: (limit = 50) =>
    fetchApi<RiskEvent[]>(`/risk/events?limit=${limit}`),

  // Strategy
  getStrategyStatus: () => fetchApi<StrategyConfig>("/strategy/status"),
  updateStrategyConfig: (config: Partial<StrategyConfig>) =>
    fetchApi<StrategyConfig>("/strategy/config", {
      method: "PUT",
      body: JSON.stringify(config),
    }),

  // System
  getSystemHealth: () => fetchApi<SystemHealth>("/system/health"),
  getEngineStatus: () => fetchApi<EngineStatus>("/system/engine"),
  toggleTrading: (enabled: boolean) =>
    fetchApi<{ trading_enabled: boolean; state: string }>(
      "/system/engine/trading",
      { method: "PUT", body: JSON.stringify({ enabled }) },
    ),

  // Backtest
  runBacktest: (request: BacktestRequest) =>
    fetchApi<{ id: number; status: string }>("/backtest/run", {
      method: "POST",
      body: JSON.stringify(request),
    }),
  getBacktest: (id: number) => fetchApi<BacktestDetail>(`/backtest/${id}`),
  getBacktests: (skip = 0, limit = 20, strategyName?: string) => {
    let url = `/backtests?skip=${skip}&limit=${limit}`;
    if (strategyName) url += `&strategy_name=${strategyName}`;
    return fetchApi<BacktestsResponse>(url);
  },
};
