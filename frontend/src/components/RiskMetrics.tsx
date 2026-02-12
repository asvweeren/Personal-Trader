import type { RiskMetricsData } from "../types";

interface Props {
  risk: RiskMetricsData | null;
}

export function RiskMetrics({ risk }: Props) {
  if (!risk) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">Risk Management</h2>
        <p className="text-gray-600 text-sm animate-pulse">Loading risk metrics...</p>
      </div>
    );
  }

  const { health, limits, daily_loss_triggered } = risk;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-gray-400">Risk Management</h2>
        <div className="flex items-center gap-2">
          {!health.market_open && (
            <span className="text-xs px-2 py-1 rounded bg-gray-800 text-gray-500">
              MARKET CLOSED
            </span>
          )}
          <span
            className={`text-xs px-2 py-1 rounded ${
              daily_loss_triggered
                ? "bg-red-900/50 text-red-400"
                : health.healthy
                  ? "bg-green-900/50 text-green-400"
                  : "bg-yellow-900/50 text-yellow-400"
            }`}
          >
            {daily_loss_triggered ? "HALTED" : health.healthy ? "HEALTHY" : "WARNING"}
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {/* Daily Loss */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Daily Loss</span>
            <span className="text-gray-400">
              {health.daily_loss_pct.toFixed(1)}% / {limits.max_daily_loss_pct}%
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                health.daily_loss_pct > limits.max_daily_loss_pct * 0.7
                  ? "bg-red-500"
                  : "bg-green-500"
              }`}
              style={{
                width: `${Math.min((health.daily_loss_pct / limits.max_daily_loss_pct) * 100, 100)}%`,
              }}
            />
          </div>
        </div>

        {/* Max Drawdown */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Max Drawdown</span>
            <span className="text-gray-400">{health.max_drawdown_pct.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                health.max_drawdown_pct > 10 ? "bg-red-500" : health.max_drawdown_pct > 5 ? "bg-yellow-500" : "bg-green-500"
              }`}
              style={{ width: `${Math.min(health.max_drawdown_pct * 5, 100)}%` }}
            />
          </div>
        </div>

        {/* Cash Reserve */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Cash Reserve</span>
            <span className="text-gray-400">
              {health.cash_reserve_pct.toFixed(1)}% (min: {limits.min_cash_reserve_pct}%)
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                health.cash_reserve_pct < limits.min_cash_reserve_pct * 1.2
                  ? "bg-yellow-500"
                  : "bg-green-500"
              }`}
              style={{ width: `${Math.min(health.cash_reserve_pct, 100)}%` }}
            />
          </div>
        </div>

        {/* Positions */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Open Positions</span>
            <span className="text-gray-400">
              {health.position_count} / {limits.max_open_positions}
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-500"
              style={{
                width: `${(health.position_count / limits.max_open_positions) * 100}%`,
              }}
            />
          </div>
        </div>

        {/* Largest Position */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Largest Position</span>
            <span className="text-gray-400">
              {health.largest_position_pct.toFixed(1)}% / {limits.max_position_pct}%
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                health.largest_position_pct > limits.max_position_pct * 0.8
                  ? "bg-yellow-500"
                  : "bg-blue-500"
              }`}
              style={{
                width: `${Math.min((health.largest_position_pct / limits.max_position_pct) * 100, 100)}%`,
              }}
            />
          </div>
        </div>

        {/* Sector Exposure */}
        {Object.keys(health.sector_exposure).length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-800">
            <div className="text-xs text-gray-500 mb-2">Sector Exposure</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(health.sector_exposure)
                .sort(([, a], [, b]) => b - a)
                .map(([sector, pct]) => (
                  <div key={sector} className="flex justify-between text-xs">
                    <span className="text-gray-400 truncate">{sector}</span>
                    <span className="text-gray-500 ml-2">{pct.toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* VaR Metrics */}
        {(health as Record<string, unknown>).var_95 != null && Number((health as Record<string, unknown>).var_95) > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-800">
            <div className="text-xs text-gray-500 mb-2">Value at Risk</div>
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-xs text-gray-500">VaR 95%</div>
                <div className="text-sm text-red-400">
                  ${Number((health as Record<string, unknown>).var_95 ?? 0).toFixed(0)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500">VaR 99%</div>
                <div className="text-sm text-red-400">
                  ${Number((health as Record<string, unknown>).var_99 ?? 0).toFixed(0)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500">CVaR 95%</div>
                <div className="text-sm text-red-400">
                  ${Number((health as Record<string, unknown>).cvar_95 ?? 0).toFixed(0)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Warnings */}
        {health.warnings.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-800">
            {health.warnings.map((warning, i) => (
              <p key={i} className="text-xs text-yellow-400 mb-1">
                {warning}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
