import { useState } from "react";
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

function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-block ml-1">
      <span
        className="text-gray-600 hover:text-gray-400 cursor-help text-xs"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onClick={() => setShow(!show)}
      >
        ?
      </span>
      {show && (
        <span className="absolute z-10 bottom-full left-1/2 -translate-x-1/2 mb-1 w-52 px-3 py-2 text-xs text-gray-300 bg-gray-800 border border-gray-700 rounded-lg shadow-lg">
          {text}
        </span>
      )}
    </span>
  );
}

interface Props {
  portfolio: Portfolio | null;
  performance: Performance | null;
}

export function PortfolioCard({ portfolio, performance }: Props) {
  if (!portfolio) return null;

  const pnlColor =
    (performance?.total_return_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400";

  const positionsValue = portfolio.positions.reduce(
    (sum, p) => sum + (p.market_value ?? 0),
    0
  );

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">
        Portfolio
        <Tooltip text="Totaaloverzicht van je beleggingsrekening bij Interactive Brokers." />
      </h2>

      <div className="space-y-4">
        {/* Total portfolio value */}
        <div>
          <div className="text-xs text-gray-500">
            Totale Waarde
            <Tooltip text="De totale waarde van je account: al je posities + beschikbaar geld samen." />
          </div>
          <div className="text-3xl font-bold text-white">
            {formatCurrency(portfolio.total_value)}
          </div>
          <div className={`text-sm ${pnlColor}`}>
            {formatPct(performance?.total_return_pct)} totaal rendement
            <Tooltip text="Winst of verlies sinds de start, als percentage van je startkapitaal." />
          </div>
        </div>

        {/* Key breakdown: what's invested vs what's available */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
          <div>
            <div className="text-xs text-gray-500">
              In Posities
              <Tooltip text="Het bedrag dat momenteel vastzit in aandelen. Dit is de huidige marktwaarde van al je open posities." />
            </div>
            <div className="text-lg font-semibold text-white">
              {formatCurrency(positionsValue)}
            </div>
            <div className="text-xs text-gray-500">
              {portfolio.positions.length} positie{portfolio.positions.length !== 1 ? "s" : ""}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">
              Beschikbaar
              <Tooltip text="Het kassaldo op je rekening. Dit is het bedrag dat niet in posities vastzit." />
            </div>
            <div className="text-lg font-semibold text-white">
              {formatCurrency(portfolio.cash)}
            </div>
          </div>
        </div>

        {/* P&L breakdown */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
          <div>
            <div className="text-xs text-gray-500">
              Open Winst/Verlies
              <Tooltip text="Winst of verlies op posities die nog open staan. Dit verandert voortdurend met de koers." />
            </div>
            <div
              className={`text-sm font-medium ${portfolio.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {formatCurrency(portfolio.unrealized_pnl)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">
              Gerealiseerd
              <Tooltip text="Winst of verlies van posities die al gesloten zijn. Dit is definitief." />
            </div>
            <div
              className={`text-sm font-medium ${portfolio.realized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {formatCurrency(portfolio.realized_pnl)}
            </div>
          </div>
        </div>

        {/* Performance stats */}
        {performance && (
          <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-800">
            <div>
              <div className="text-xs text-gray-500">
                Win Rate
                <Tooltip text="Percentage van gesloten trades met winst." />
              </div>
              <div className="text-sm font-medium">{performance.win_rate}%</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">
                Trades
                <Tooltip text="Totaal aantal uitgevoerde trades." />
              </div>
              <div className="text-sm font-medium">{performance.total_trades}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">
                Max DD
                <Tooltip text="Maximale drawdown: de grootste daling van piek naar dal. Hoe lager hoe beter." />
              </div>
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
