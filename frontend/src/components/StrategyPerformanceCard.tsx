import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { StrategyPerformance } from "../types";

export function StrategyPerformanceCard() {
  const [strategies, setStrategies] = useState<StrategyPerformance[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getStrategyPerformance()
      .then((res) => setStrategies(res.strategies))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">
        Strategy Performance
      </h2>

      {loading ? (
        <div className="text-gray-600 text-sm py-4 text-center">Loading...</div>
      ) : strategies.length === 0 ? (
        <div className="text-gray-600 text-sm py-4 text-center">
          No closed trades yet
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs border-b border-gray-800">
                <th className="text-left py-2 pr-3">Strategy</th>
                <th className="text-right py-2 px-2">Trades</th>
                <th className="text-right py-2 px-2">Win Rate</th>
                <th className="text-right py-2 px-2">P&L</th>
                <th className="text-right py-2 pl-2">PF</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s) => (
                <tr
                  key={s.strategy_name}
                  className="border-b border-gray-800/50"
                >
                  <td className="py-2 pr-3 text-gray-200 font-medium">
                    {s.strategy_name}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {s.total_trades}
                  </td>
                  <td className="py-2 px-2 text-right">
                    <span
                      className={
                        s.win_rate >= 50 ? "text-green-400" : "text-red-400"
                      }
                    >
                      {s.win_rate.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right">
                    <span
                      className={
                        s.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                      }
                    >
                      {s.total_pnl >= 0 ? "+" : ""}
                      {s.total_pnl.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-2 pl-2 text-right text-gray-300">
                    {s.profit_factor === "Inf" ? "\u221e" : s.profit_factor}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
