import type { Portfolio, Performance } from "../types";

function formatCurrency(value: number | null | undefined): string {
  if (value == null) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "EUR",
    minimumFractionDigits: 2,
  }).format(value);
}

function formatPct(value: number | null | undefined): string {
  if (value == null) return "-";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

interface Props {
  portfolio: Portfolio | null;
  performance: Performance | null;
}

export function PortfolioCard({ portfolio, performance }: Props) {
  if (!portfolio) return null;

  const pnlColor =
    (performance?.total_return_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400";

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">Portfolio</h2>

      <div className="space-y-4">
        <div>
          <div className="text-3xl font-bold text-white">
            {formatCurrency(portfolio.total_value)}
          </div>
          <div className={`text-sm ${pnlColor}`}>
            {formatPct(performance?.total_return_pct)} total return
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
          <div>
            <div className="text-xs text-gray-500">Cash</div>
            <div className="text-sm font-medium">{formatCurrency(portfolio.cash)}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Unrealized P&L</div>
            <div
              className={`text-sm font-medium ${portfolio.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {formatCurrency(portfolio.unrealized_pnl)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Realized P&L</div>
            <div
              className={`text-sm font-medium ${portfolio.realized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {formatCurrency(portfolio.realized_pnl)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Positions</div>
            <div className="text-sm font-medium">{portfolio.positions.length}</div>
          </div>
        </div>

        {performance && (
          <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-800">
            <div>
              <div className="text-xs text-gray-500">Win Rate</div>
              <div className="text-sm font-medium">{performance.win_rate}%</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Trades</div>
              <div className="text-sm font-medium">{performance.total_trades}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Max DD</div>
              <div className="text-sm font-medium text-red-400">
                {performance.max_drawdown}%
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
