import { useState, useEffect, useCallback } from "react";
import { api } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import type {
  EngineStatus,
  SystemHealth,
  RiskLimits,
  StrategyConfig,
  AvailableStrategy,
} from "../types";

// ─── Toggle Switch ──────────────────────────────────────────────────

function ToggleSwitch({
  enabled,
  onToggle,
  disabled,
}: {
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={onToggle}
      disabled={disabled}
      className={`relative inline-flex h-8 w-14 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed ${
        enabled ? "bg-green-600" : "bg-gray-600"
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-7 w-7 rounded-full bg-white shadow-lg ring-0 transition-transform duration-200 ease-in-out ${
          enabled ? "translate-x-6" : "translate-x-0"
        }`}
      />
    </button>
  );
}

// ─── Toast / Status Message ─────────────────────────────────────────

type ToastType = "success" | "error";

function Toast({
  message,
  type,
  onDismiss,
}: {
  message: string;
  type: ToastType;
  onDismiss: () => void;
}) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 4000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  return (
    <div
      className={`fixed top-6 right-6 z-50 flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg border ${
        type === "success"
          ? "bg-green-900/80 border-green-700 text-green-300"
          : "bg-red-900/80 border-red-700 text-red-300"
      }`}
    >
      <span className="text-sm">{message}</span>
      <button
        onClick={onDismiss}
        className="text-gray-400 hover:text-gray-200 text-lg leading-none"
      >
        x
      </button>
    </div>
  );
}

// ─── Card Wrapper ───────────────────────────────────────────────────

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-5">{title}</h2>
      {children}
    </div>
  );
}

// ─── Main Settings Page ─────────────────────────────────────────────

export function SettingsPage() {
  const { logout } = useAuth();

  // Toast state
  const [toast, setToast] = useState<{
    message: string;
    type: ToastType;
  } | null>(null);

  const showToast = useCallback((message: string, type: ToastType) => {
    setToast({ message, type });
  }, []);

  // ── Engine / Trading Control ────────────────────────────────────
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [toggling, setToggling] = useState(false);

  // ── System Health ───────────────────────────────────────────────
  const [health, setHealth] = useState<SystemHealth | null>(null);

  // ── Risk Limits ─────────────────────────────────────────────────
  const [riskLimits, setRiskLimits] = useState<RiskLimits>({
    max_daily_loss_pct: 2,
    max_position_pct: 5,
    max_open_positions: 5,
    min_cash_reserve_pct: 20,
  });
  const [riskSaving, setRiskSaving] = useState(false);

  // ── Strategy Config ─────────────────────────────────────────────
  const [strategyConfig, setStrategyConfig] = useState<StrategyConfig>({
    active_strategies: ["ml_xgboost", "sentiment"],
    confidence_threshold: 0.6,
    ensemble_method: "weighted_average",
    weights: {},
    trading_enabled: false,
  });
  const [availableStrategies, setAvailableStrategies] = useState<
    AvailableStrategy[]
  >([]);
  const [symbolsInput, setSymbolsInput] = useState("");
  const [strategySaving, setStrategySaving] = useState(false);

  // ── Fetch all data on mount ─────────────────────────────────────
  useEffect(() => {
    // Engine status
    api
      .getEngineStatus()
      .then((data) => {
        setEngine(data);
        setSymbolsInput(data.symbols.join(", "));
      })
      .catch(() => {});

    // System health
    api
      .getSystemHealth()
      .then(setHealth)
      .catch(() => {});

    // Risk metrics (contains limits)
    api
      .getRiskMetrics()
      .then((data) => {
        setRiskLimits(data.limits);
      })
      .catch(() => {});

    // Strategy status (full response with available_strategies)
    api
      .getStrategyStatusFull()
      .then((data) => {
        setStrategyConfig(data.config);
        setAvailableStrategies(data.available_strategies);
      })
      .catch(() => {});
  }, []);

  // Poll engine status
  useEffect(() => {
    const interval = setInterval(() => {
      api
        .getEngineStatus()
        .then(setEngine)
        .catch(() => {});
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // ── Handlers ────────────────────────────────────────────────────

  const [alertSending, setAlertSending] = useState(false);

  const handleTestAlert = async () => {
    setAlertSending(true);
    try {
      const result = await api.testAlert();
      const channels = Object.entries(result.channels)
        .filter(([, v]) => v)
        .map(([k]) => k)
        .join(", ");
      showToast(`Test alert sent via: ${channels}`, "success");
    } catch (err) {
      showToast(
        `Failed to send test alert: ${err instanceof Error ? err.message : "Unknown error"}`,
        "error",
      );
    } finally {
      setAlertSending(false);
    }
  };

  const handleTradingToggle = async () => {
    if (!engine) return;
    setToggling(true);
    try {
      const result = await api.toggleTrading(!engine.trading_enabled);
      setEngine((prev) =>
        prev
          ? {
              ...prev,
              trading_enabled: result.trading_enabled,
              state: result.state,
            }
          : prev,
      );
      showToast(
        result.trading_enabled ? "Trading enabled" : "Trading disabled",
        "success",
      );
    } catch (err) {
      showToast(
        `Failed to toggle trading: ${err instanceof Error ? err.message : "Unknown error"}`,
        "error",
      );
    } finally {
      setToggling(false);
    }
  };

  const handleSaveRiskLimits = async () => {
    setRiskSaving(true);
    try {
      const updated = await api.updateRiskLimits(riskLimits);
      setRiskLimits(updated);
      showToast("Risk limits saved successfully", "success");
    } catch (err) {
      showToast(
        `Failed to save risk limits: ${err instanceof Error ? err.message : "Unknown error"}`,
        "error",
      );
    } finally {
      setRiskSaving(false);
    }
  };

  const handleSaveStrategyConfig = async () => {
    setStrategySaving(true);
    try {
      const updated = await api.updateStrategyConfig({
        active_strategies: strategyConfig.active_strategies,
        confidence_threshold: strategyConfig.confidence_threshold,
        ensemble_method: strategyConfig.ensemble_method,
      });
      setStrategyConfig(updated);
      showToast("Strategy configuration saved successfully", "success");
    } catch (err) {
      showToast(
        `Failed to save strategy config: ${err instanceof Error ? err.message : "Unknown error"}`,
        "error",
      );
    } finally {
      setStrategySaving(false);
    }
  };

  const handleStrategyToggle = (strategyName: string) => {
    setStrategyConfig((prev) => {
      const isActive = prev.active_strategies.includes(strategyName);
      return {
        ...prev,
        active_strategies: isActive
          ? prev.active_strategies.filter((s) => s !== strategyName)
          : [...prev.active_strategies, strategyName],
      };
    });
  };

  // ── State colors ────────────────────────────────────────────────

  const STATE_COLORS: Record<string, string> = {
    RUNNING: "bg-green-900/50 text-green-400",
    STOPPED: "bg-gray-800 text-gray-400",
    STARTING: "bg-blue-900/50 text-blue-400",
    STOPPING: "bg-yellow-900/50 text-yellow-400",
    ERROR: "bg-red-900/50 text-red-400",
    NOT_INITIALIZED: "bg-gray-800 text-gray-500",
  };

  const componentStatusColor = (status: string) => {
    if (status === "connected" || status === "healthy") return "text-green-400";
    if (status === "degraded" || status === "reconnecting")
      return "text-yellow-400";
    return "text-red-400";
  };

  const componentStatusDot = (status: string) => {
    if (status === "connected" || status === "healthy") return "bg-green-400";
    if (status === "degraded" || status === "reconnecting")
      return "bg-yellow-400";
    return "bg-red-400";
  };

  return (
    <div className="space-y-6">
      {/* Toast notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onDismiss={() => setToast(null)}
        />
      )}

      {/* ── Trading Control ────────────────────────────────────────── */}
      <Card title="Trading Control">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <ToggleSwitch
              enabled={engine?.trading_enabled ?? false}
              onToggle={handleTradingToggle}
              disabled={
                toggling ||
                !engine ||
                engine.state === "NOT_INITIALIZED"
              }
            />
            <div>
              <div className="text-base font-medium text-gray-200">
                {engine?.trading_enabled ? "Trading Active" : "Trading Paused"}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                {engine?.trading_enabled
                  ? "The engine is executing trades automatically"
                  : "The engine will not place any trades"}
              </div>
            </div>
          </div>

          {engine && (
            <div className="flex items-center gap-4">
              <span
                className={`text-xs px-2.5 py-1 rounded font-medium ${
                  STATE_COLORS[engine.state] ?? STATE_COLORS.STOPPED
                }`}
              >
                {engine.state}
              </span>
              <div className="text-right">
                <div className="text-xs text-gray-500">Cycles</div>
                <div className="text-sm font-medium text-gray-300">
                  {engine.cycle_count}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500">Open Trades</div>
                <div className="text-sm font-medium text-gray-300">
                  {engine.open_trades}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500">Last Cycle</div>
                <div className="text-sm font-medium text-gray-400">
                  {engine.last_cycle_at
                    ? new Date(engine.last_cycle_at).toLocaleTimeString()
                    : "--"}
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Risk Limits ────────────────────────────────────────── */}
        <Card title="Risk Limits">
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Max Daily Loss (%)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="100"
                value={riskLimits.max_daily_loss_pct}
                onChange={(e) =>
                  setRiskLimits((prev) => ({
                    ...prev,
                    max_daily_loss_pct: parseFloat(e.target.value) || 0,
                  }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Max Position Size (%)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="100"
                value={riskLimits.max_position_pct}
                onChange={(e) =>
                  setRiskLimits((prev) => ({
                    ...prev,
                    max_position_pct: parseFloat(e.target.value) || 0,
                  }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Max Open Positions
              </label>
              <input
                type="number"
                step="1"
                min="0"
                max="100"
                value={riskLimits.max_open_positions}
                onChange={(e) =>
                  setRiskLimits((prev) => ({
                    ...prev,
                    max_open_positions: parseInt(e.target.value, 10) || 0,
                  }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Min Cash Reserve (%)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="100"
                value={riskLimits.min_cash_reserve_pct}
                onChange={(e) =>
                  setRiskLimits((prev) => ({
                    ...prev,
                    min_cash_reserve_pct: parseFloat(e.target.value) || 0,
                  }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <button
              onClick={handleSaveRiskLimits}
              disabled={riskSaving}
              className="w-full mt-2 py-2 rounded-md text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {riskSaving ? "Saving..." : "Save Risk Limits"}
            </button>
          </div>
        </Card>

        {/* ── Strategy Configuration ─────────────────────────────── */}
        <Card title="Strategy Configuration">
          <div className="space-y-4">
            {/* Active strategies as checkboxes */}
            <div>
              <label className="block text-xs text-gray-500 mb-2">
                Active Strategies
              </label>
              <div className="space-y-2">
                {availableStrategies.map((s) => (
                  <label
                    key={s.name}
                    className="flex items-center gap-3 cursor-pointer group"
                  >
                    <input
                      type="checkbox"
                      checked={strategyConfig.active_strategies.includes(
                        s.name,
                      )}
                      onChange={() => handleStrategyToggle(s.name)}
                      className="h-4 w-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                    />
                    <div>
                      <span className="text-sm text-gray-200 group-hover:text-white">
                        {s.name}
                      </span>
                      <span className="text-xs text-gray-500 ml-2">
                        ({s.type})
                      </span>
                      <div className="text-xs text-gray-500">
                        {s.description}
                      </div>
                    </div>
                  </label>
                ))}
                {availableStrategies.length === 0 && (
                  <div className="text-xs text-gray-500">
                    Loading available strategies...
                  </div>
                )}
              </div>
            </div>

            {/* Confidence threshold slider */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-xs text-gray-500">
                  Confidence Threshold
                </label>
                <span className="text-xs font-medium text-gray-300">
                  {(strategyConfig.confidence_threshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={Math.round(
                  strategyConfig.confidence_threshold * 100,
                )}
                onChange={(e) =>
                  setStrategyConfig((prev) => ({
                    ...prev,
                    confidence_threshold: parseInt(e.target.value, 10) / 100,
                  }))
                }
                className="w-full h-2 bg-gray-700 rounded-full appearance-none cursor-pointer accent-blue-500"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Ensemble method */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Ensemble Method
              </label>
              <select
                value={strategyConfig.ensemble_method}
                onChange={(e) =>
                  setStrategyConfig((prev) => ({
                    ...prev,
                    ensemble_method: e.target.value,
                  }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="weighted_average">Weighted Average</option>
                <option value="majority_vote">Majority Vote</option>
                <option value="max_confidence">Max Confidence</option>
              </select>
            </div>

            {/* Symbol list */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Symbols (comma-separated)
              </label>
              <input
                type="text"
                value={symbolsInput}
                onChange={(e) => setSymbolsInput(e.target.value)}
                placeholder="AAPL, GOOGL, MSFT, TSLA"
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-600"
              />
              <div className="text-xs text-gray-600 mt-1">
                Symbols are configured at the engine level and will take effect
                on the next restart.
              </div>
            </div>

            <button
              onClick={handleSaveStrategyConfig}
              disabled={strategySaving}
              className="w-full mt-2 py-2 rounded-md text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {strategySaving ? "Saving..." : "Save Strategy Config"}
            </button>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── System Info ─────────────────────────────────────────── */}
        <Card title="System Info">
          <div className="space-y-3">
            {/* Broker */}
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <div className="flex items-center gap-2">
                <div
                  className={`h-2 w-2 rounded-full ${componentStatusDot(health?.components.broker.status ?? "unknown")}`}
                />
                <span className="text-sm text-gray-300">
                  Broker Connection
                </span>
              </div>
              <span
                className={`text-sm font-medium ${componentStatusColor(health?.components.broker.status ?? "unknown")}`}
              >
                {health?.components.broker.status ?? "Loading..."}
              </span>
            </div>

            {/* Database */}
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <div className="flex items-center gap-2">
                <div
                  className={`h-2 w-2 rounded-full ${componentStatusDot(health?.components.database.status ?? "unknown")}`}
                />
                <span className="text-sm text-gray-300">Database</span>
              </div>
              <span
                className={`text-sm font-medium ${componentStatusColor(health?.components.database.status ?? "unknown")}`}
              >
                {health?.components.database.status ?? "Loading..."}
              </span>
            </div>

            {/* Trading Engine */}
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <div className="flex items-center gap-2">
                <div
                  className={`h-2 w-2 rounded-full ${componentStatusDot(health?.components.trading_engine.status ?? "unknown")}`}
                />
                <span className="text-sm text-gray-300">Trading Engine</span>
              </div>
              <span
                className={`text-sm font-medium ${componentStatusColor(health?.components.trading_engine.status ?? "unknown")}`}
              >
                {health?.components.trading_engine.status ?? "Loading..."}
              </span>
            </div>

            {/* Trading Mode */}
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <span className="text-sm text-gray-300">Trading Mode</span>
              <span
                className={`text-xs px-2.5 py-1 rounded font-medium ${
                  health?.components.paper_trading
                    ? "bg-yellow-900/50 text-yellow-400"
                    : "bg-red-900/50 text-red-400"
                }`}
              >
                {health?.components.paper_trading ? "Paper" : "Live"}
              </span>
            </div>

            {/* Environment */}
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <span className="text-sm text-gray-300">Environment</span>
              <span className="text-sm text-gray-400">
                {health?.components.environment ?? "Loading..."}
              </span>
            </div>

            {/* System Status */}
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <span className="text-sm text-gray-300">Overall Status</span>
              <span
                className={`text-xs px-2.5 py-1 rounded font-medium ${
                  health?.status === "healthy"
                    ? "bg-green-900/50 text-green-400"
                    : health?.status === "degraded"
                      ? "bg-yellow-900/50 text-yellow-400"
                      : "bg-red-900/50 text-red-400"
                }`}
              >
                {health?.status?.toUpperCase() ?? "LOADING"}
              </span>
            </div>

            {/* Cycle info */}
            {engine && (
              <>
                <div className="flex items-center justify-between py-2 border-b border-gray-800">
                  <span className="text-sm text-gray-300">Cycle Count</span>
                  <span className="text-sm text-gray-400">
                    {engine.cycle_count}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-sm text-gray-300">
                    Reconnect Attempts
                  </span>
                  <span className="text-sm text-gray-400">
                    {engine.reconnect_attempts}
                  </span>
                </div>
              </>
            )}
          </div>
        </Card>

        {/* ── Account ────────────────────────────────────────────── */}
        <Card title="Account">
          <div className="space-y-4">
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <span className="text-sm text-gray-300">Signed in as</span>
              <span className="text-sm font-medium text-gray-200">admin</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-gray-800">
              <span className="text-sm text-gray-300">Session</span>
              <span className="text-sm text-gray-400">Active</span>
            </div>

            <div className="pt-2 space-y-3">
              <button
                onClick={handleTestAlert}
                disabled={alertSending}
                className="w-full py-2.5 rounded-md text-sm font-medium bg-blue-900/40 text-blue-400 hover:bg-blue-900/60 border border-blue-800/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {alertSending ? "Sending..." : "Send Test Alert"}
              </button>
              <button
                onClick={logout}
                className="w-full py-2.5 rounded-md text-sm font-medium bg-red-900/40 text-red-400 hover:bg-red-900/60 border border-red-800/50 transition-colors"
              >
                Sign Out
              </button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
