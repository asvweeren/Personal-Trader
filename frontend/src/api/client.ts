import type {
  Portfolio,
  Position,
  Performance,
  TradesResponse,
  Trade,
  RiskMetricsData,
  RiskEvent,
  RiskLimits,
  RiskLimitsUpdate,
  StrategyConfig,
  StrategyStatusResponse,
  StrategyConfigUpdate,
  SystemHealth,
  EngineStatus,
  PortfolioSnapshot,
  BacktestRequest,
  BacktestDetail,
  BacktestsResponse,
  ValidationStatus,
  ReadinessAssessment,
  ValidationReport,
  ValidationReportSummary,
  RollingMetrics,
  BacktestComparison,
} from "../types";

const API_BASE = "/api";
const TOKEN_KEY = "auth_token";

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem(TOKEN_KEY);
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options?.headers as Record<string, string>),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    window.dispatchEvent(new Event("auth:logout"));
    throw new Error("Session expired");
  }

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
  updateRiskLimits: (limits: RiskLimitsUpdate) =>
    fetchApi<RiskLimits>("/risk/limits", {
      method: "PUT",
      body: JSON.stringify(limits),
    }),

  // Strategy
  getStrategyStatus: () => fetchApi<StrategyConfig>("/strategy/status"),
  getStrategyStatusFull: () =>
    fetchApi<StrategyStatusResponse>("/strategy/status"),
  updateStrategyConfig: (config: StrategyConfigUpdate) =>
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
  testAlert: () =>
    fetchApi<{ status: string; channels: Record<string, boolean> }>(
      "/system/test-alert",
      { method: "POST" },
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

  // Validation
  getValidationStatus: () =>
    fetchApi<ValidationStatus>("/validation/status"),
  getValidationReadiness: () =>
    fetchApi<ReadinessAssessment>("/validation/readiness"),
  getLatestValidationReport: () =>
    fetchApi<ValidationReport>("/validation/report/latest"),
  getValidationHistory: (days = 30) =>
    fetchApi<ValidationReportSummary[]>(`/validation/history?days=${days}`),
  getRollingMetrics: () =>
    fetchApi<RollingMetrics>("/validation/metrics/rolling"),
  getBacktestComparison: () =>
    fetchApi<BacktestComparison>("/validation/comparison"),
  generateValidationReport: () =>
    fetchApi<ValidationReport>("/validation/report/generate", {
      method: "POST",
    }),
};
