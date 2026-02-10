import { useState, useEffect, useCallback } from "react";
import { api } from "../api/client";
import type {
  ValidationStatus,
  ReadinessAssessment,
  RollingMetrics,
  BacktestComparison,
} from "../types";

export function ValidationDashboard() {
  const [status, setStatus] = useState<ValidationStatus | null>(null);
  const [readiness, setReadiness] = useState<ReadinessAssessment | null>(null);
  const [rolling, setRolling] = useState<RollingMetrics | null>(null);
  const [comparison, setComparison] = useState<BacktestComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statusRes, readinessRes, rollingRes, comparisonRes] =
        await Promise.allSettled([
          api.getValidationStatus(),
          api.getValidationReadiness(),
          api.getRollingMetrics(),
          api.getBacktestComparison(),
        ]);

      if (statusRes.status === "fulfilled") setStatus(statusRes.value);
      if (readinessRes.status === "fulfilled") setReadiness(readinessRes.value);
      if (rollingRes.status === "fulfilled") setRolling(rollingRes.value);
      if (comparisonRes.status === "fulfilled")
        setComparison(comparisonRes.value);

      // If all failed, show error
      const allFailed = [statusRes, readinessRes, rollingRes, comparisonRes].every(
        (r) => r.status === "rejected",
      );
      if (allFailed) {
        setError("Failed to load validation data. The validation endpoints may not be available yet.");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load validation data",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const handleGenerateReport = async () => {
    setGenerating(true);
    try {
      await api.generateValidationReport();
      // Refresh all data after generating
      await fetchAll();
    } catch {
      // ignore - data will stay as-is
    } finally {
      setGenerating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-gray-400 text-sm">Loading validation data...</div>
      </div>
    );
  }

  if (error && !status && !readiness && !rolling && !comparison) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <div className="text-center py-8">
          <div className="text-gray-500 text-sm mb-2">
            Validation data unavailable
          </div>
          <p className="text-gray-600 text-xs max-w-md mx-auto">
            {error}
          </p>
          <button
            onClick={fetchAll}
            className="mt-4 px-4 py-1.5 rounded text-sm font-medium bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const noDataYet = !status?.is_active && !status?.is_complete;

  return (
    <div className="space-y-6">
      {/* Header row with title and actions */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">
            Paper Trading Validation
          </h2>
          <p className="text-sm text-gray-500">
            Track validation progress and readiness for live trading
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchAll}
            className="px-3 py-1.5 rounded text-sm font-medium bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={handleGenerateReport}
            disabled={generating}
            className="px-3 py-1.5 rounded text-sm font-medium bg-blue-900/50 text-blue-400 hover:bg-blue-900/70 disabled:opacity-50 transition-colors"
          >
            {generating ? "Generating..." : "Generate Report"}
          </button>
        </div>
      </div>

      {noDataYet && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <div className="text-center py-4">
            <div className="text-gray-400 text-sm mb-1">
              No paper trading validation in progress
            </div>
            <p className="text-gray-600 text-xs max-w-md mx-auto">
              Paper trading validation will begin automatically when the trading
              engine runs in paper mode. Complete at least 28 days of paper
              trading to validate your strategy before going live.
            </p>
          </div>
        </div>
      )}

      {/* Top row: Status + Readiness */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PaperTradingStatusCard status={status} />
        <ReadinessCard readiness={readiness} />
      </div>

      {/* Bottom row: Rolling Metrics + Backtest Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RollingMetricsCard metrics={rolling} />
        <BacktestComparisonCard comparison={comparison} />
      </div>
    </div>
  );
}

// ─── Paper Trading Status Card ─────────────────────────────────────

function PaperTradingStatusCard({ status }: { status: ValidationStatus | null }) {
  if (!status) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">
          Paper Trading Status
        </h2>
        <p className="text-gray-600 text-sm">No status data available</p>
      </div>
    );
  }

  const progressPct = Math.min(status.progress_pct, 100);
  const daysRemaining = Math.max(status.min_days_required - status.days_elapsed, 0);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-gray-400">
          Paper Trading Status
        </h2>
        <span
          className={`text-xs px-2 py-1 rounded ${
            status.is_complete
              ? "bg-green-900/50 text-green-400"
              : status.is_active
                ? "bg-blue-900/50 text-blue-400"
                : "bg-gray-800 text-gray-500"
          }`}
        >
          {status.is_complete
            ? "COMPLETE"
            : status.is_active
              ? "ACTIVE"
              : "INACTIVE"}
        </span>
      </div>

      <div className="space-y-4">
        {/* Phase */}
        <div>
          <div className="text-xs text-gray-500 mb-1">Current Phase</div>
          <div className="text-sm font-medium text-gray-200">
            {status.current_phase}
          </div>
        </div>

        {/* Progress bar */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Validation Progress</span>
            <span className="text-gray-400">{progressPct.toFixed(0)}%</span>
          </div>
          <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                status.is_complete
                  ? "bg-green-500"
                  : progressPct > 75
                    ? "bg-blue-500"
                    : progressPct > 40
                      ? "bg-blue-600"
                      : "bg-blue-700"
              }`}
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
          <div>
            <div className="text-xs text-gray-500">Days Elapsed</div>
            <div className="text-sm font-medium text-gray-200">
              {status.days_elapsed}{" "}
              <span className="text-gray-500 text-xs">
                / {status.min_days_required}
              </span>
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Days Remaining</div>
            <div className="text-sm font-medium text-gray-200">
              {status.is_complete ? (
                <span className="text-green-400">Done</span>
              ) : (
                daysRemaining
              )}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Total Trades</div>
            <div className="text-sm font-medium text-gray-200">
              {status.total_trades}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Start Date</div>
            <div className="text-sm font-medium text-gray-200">
              {status.start_date
                ? new Date(status.start_date).toLocaleDateString()
                : "-"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Readiness Assessment Card ─────────────────────────────────────

function ReadinessCard({
  readiness,
}: {
  readiness: ReadinessAssessment | null;
}) {
  if (!readiness) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">
          Go-Live Readiness
        </h2>
        <p className="text-gray-600 text-sm">
          Complete paper trading validation to see readiness assessment
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-gray-400">
          Go-Live Readiness
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {readiness.passed_count}/{readiness.total_count} passed
          </span>
          <span
            className={`text-xs px-2 py-1 rounded ${
              readiness.ready
                ? "bg-green-900/50 text-green-400"
                : "bg-yellow-900/50 text-yellow-400"
            }`}
          >
            {readiness.ready ? "READY" : "NOT READY"}
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {/* Overall score bar */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Overall Score</span>
            <span className="text-gray-400">
              {(readiness.overall_score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                readiness.overall_score >= 0.8
                  ? "bg-green-500"
                  : readiness.overall_score >= 0.5
                    ? "bg-yellow-500"
                    : "bg-red-500"
              }`}
              style={{ width: `${readiness.overall_score * 100}%` }}
            />
          </div>
        </div>

        {/* Criteria list */}
        <div className="pt-3 border-t border-gray-800 space-y-2 max-h-56 overflow-y-auto">
          {readiness.criteria.map((c, i) => (
            <div
              key={i}
              className="flex items-center justify-between text-sm"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className={`flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-xs ${
                    c.passed
                      ? "bg-green-900/50 text-green-400"
                      : "bg-red-900/50 text-red-400"
                  }`}
                >
                  {c.passed ? "\u2713" : "\u2717"}
                </span>
                <span className="text-gray-300 truncate">{c.name}</span>
              </div>
              <div className="text-xs text-gray-500 ml-2 flex-shrink-0">
                {String(c.actual)} / {String(c.required)}
              </div>
            </div>
          ))}
        </div>

        {/* Blockers */}
        {readiness.blockers.length > 0 && (
          <div className="pt-3 border-t border-gray-800">
            <div className="text-xs text-gray-500 mb-1">Blockers</div>
            {readiness.blockers.map((b, i) => (
              <p key={i} className="text-xs text-red-400 mb-1">
                {b}
              </p>
            ))}
          </div>
        )}

        {/* Recommendation */}
        {readiness.recommendation && (
          <div className="pt-3 border-t border-gray-800">
            <div className="text-xs text-gray-500 mb-1">Recommendation</div>
            <p className="text-xs text-gray-400">{readiness.recommendation}</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Rolling Metrics Card ──────────────────────────────────────────

function RollingMetricsCard({ metrics }: { metrics: RollingMetrics | null }) {
  if (!metrics) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">
          Rolling Performance Metrics
        </h2>
        <p className="text-gray-600 text-sm">
          Metrics will appear once paper trading data is available
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-gray-400">
          Rolling Performance Metrics
        </h2>
        <span className="text-xs text-gray-600">
          {metrics.period_days}-day window
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <MetricItem
          label="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(2)}
          color={metrics.sharpe_ratio >= 1 ? "green" : metrics.sharpe_ratio >= 0 ? "yellow" : "red"}
        />
        <MetricItem
          label="Win Rate"
          value={`${metrics.win_rate.toFixed(1)}%`}
          color={metrics.win_rate >= 50 ? "green" : "red"}
        />
        <MetricItem
          label="Profit Factor"
          value={metrics.profit_factor.toFixed(2)}
          color={metrics.profit_factor >= 1.5 ? "green" : metrics.profit_factor >= 1 ? "yellow" : "red"}
        />
        <MetricItem
          label="Max Drawdown"
          value={`${metrics.max_drawdown.toFixed(2)}%`}
          color={metrics.max_drawdown <= 5 ? "green" : metrics.max_drawdown <= 10 ? "yellow" : "red"}
        />
        <MetricItem
          label="Total P&L"
          value={`\u20AC${metrics.total_pnl.toFixed(2)}`}
          color={metrics.total_pnl >= 0 ? "green" : "red"}
        />
        <MetricItem
          label="Total Trades"
          value={String(metrics.total_trades)}
          color="default"
        />
        <MetricItem
          label="Avg Trade P&L"
          value={`\u20AC${metrics.avg_trade_pnl.toFixed(2)}`}
          color={metrics.avg_trade_pnl >= 0 ? "green" : "red"}
        />
        <MetricItem
          label="Calmar Ratio"
          value={metrics.calmar_ratio.toFixed(2)}
          color={metrics.calmar_ratio >= 1 ? "green" : metrics.calmar_ratio >= 0 ? "yellow" : "red"}
        />
        <MetricItem
          label="Volatility"
          value={`${metrics.volatility.toFixed(2)}%`}
          color="default"
        />
      </div>

      {metrics.updated_at && (
        <div className="mt-4 pt-3 border-t border-gray-800 text-xs text-gray-600">
          Last updated: {new Date(metrics.updated_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

function MetricItem({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: "green" | "yellow" | "red" | "default";
}) {
  const colorClass =
    color === "green"
      ? "text-green-400"
      : color === "yellow"
        ? "text-yellow-400"
        : color === "red"
          ? "text-red-400"
          : "text-gray-300";

  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-sm font-medium ${colorClass}`}>{value}</div>
    </div>
  );
}

// ─── Backtest Comparison Card ──────────────────────────────────────

function BacktestComparisonCard({
  comparison,
}: {
  comparison: BacktestComparison | null;
}) {
  if (!comparison) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">
          Backtest vs Live Comparison
        </h2>
        <p className="text-gray-600 text-sm">
          Comparison data will appear after enough paper trades have been
          executed
        </p>
      </div>
    );
  }

  if (!comparison.has_data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">
          Backtest vs Live Comparison
        </h2>
        <p className="text-gray-600 text-sm">
          Not enough data yet to compare backtest and live results. Continue
          paper trading to build comparison metrics.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-gray-400">
          Backtest vs Live Comparison
        </h2>
        <span
          className={`text-xs px-2 py-1 rounded ${
            comparison.overall_deviation <= 15
              ? "bg-green-900/50 text-green-400"
              : comparison.overall_deviation <= 30
                ? "bg-yellow-900/50 text-yellow-400"
                : "bg-red-900/50 text-red-400"
          }`}
        >
          {comparison.overall_deviation.toFixed(1)}% deviation
        </span>
      </div>

      <div className="space-y-3">
        {/* Metrics comparison table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs border-b border-gray-800">
                <th className="text-left pb-2">Metric</th>
                <th className="text-right pb-2">Backtest</th>
                <th className="text-right pb-2">Live</th>
                <th className="text-right pb-2">Deviation</th>
              </tr>
            </thead>
            <tbody>
              {comparison.metrics.map((m, i) => (
                <tr key={i} className="border-b border-gray-800/50">
                  <td className="py-1.5 text-gray-300">{m.metric}</td>
                  <td className="py-1.5 text-right text-gray-400">
                    {formatMetricValue(m.backtest_value)}
                  </td>
                  <td className="py-1.5 text-right text-gray-400">
                    {formatMetricValue(m.live_value)}
                  </td>
                  <td
                    className={`py-1.5 text-right ${
                      m.acceptable ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {m.deviation_pct >= 0 ? "+" : ""}
                    {m.deviation_pct.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Correlation */}
        <div className="pt-3 border-t border-gray-800 flex items-center justify-between">
          <div>
            <div className="text-xs text-gray-500">Return Correlation</div>
            <div
              className={`text-sm font-medium ${
                comparison.correlation >= 0.7
                  ? "text-green-400"
                  : comparison.correlation >= 0.4
                    ? "text-yellow-400"
                    : "text-red-400"
              }`}
            >
              {comparison.correlation.toFixed(3)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-500">Assessment</div>
            <div className="text-sm text-gray-300">{comparison.assessment}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatMetricValue(value: number): string {
  if (Math.abs(value) >= 1000) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}
