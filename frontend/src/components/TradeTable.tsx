import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { Trade } from "../types";

export function TradeTable() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getTrades(0, 20)
      .then((data) => {
        setTrades(data.trades);
        setTotal(data.total);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">
        Recent Trades ({total})
      </h2>

      {loading ? (
        <p className="text-gray-600 text-sm">Loading trades...</p>
      ) : trades.length === 0 ? (
        <p className="text-gray-600 text-sm">No trades yet</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs border-b border-gray-800">
                <th className="text-left pb-2">Symbol</th>
                <th className="text-left pb-2">Side</th>
                <th className="text-right pb-2">Qty</th>
                <th className="text-right pb-2">Entry</th>
                <th className="text-right pb-2">P&L</th>
                <th className="text-left pb-2">Status</th>
                <th className="text-left pb-2">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => (
                <tr key={trade.id} className="border-b border-gray-800/50">
                  <td className="py-2 font-medium text-white">{trade.symbol}</td>
                  <td
                    className={`py-2 ${trade.side === "BUY" ? "text-green-400" : "text-red-400"}`}
                  >
                    {trade.side}
                  </td>
                  <td className="py-2 text-right">{trade.quantity}</td>
                  <td className="py-2 text-right">
                    {trade.entry_price?.toFixed(2) ?? "-"}
                  </td>
                  <td
                    className={`py-2 text-right ${(trade.realized_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}
                  >
                    {trade.realized_pnl != null
                      ? `${trade.realized_pnl >= 0 ? "+" : ""}${trade.realized_pnl.toFixed(2)}`
                      : "-"}
                  </td>
                  <td className="py-2">
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        trade.status === "OPEN"
                          ? "bg-blue-900/50 text-blue-400"
                          : trade.status === "CLOSED"
                            ? "bg-gray-800 text-gray-400"
                            : "bg-yellow-900/50 text-yellow-400"
                      }`}
                    >
                      {trade.status}
                    </span>
                  </td>
                  <td className="py-2 text-gray-500 text-xs">{trade.strategy_name}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
