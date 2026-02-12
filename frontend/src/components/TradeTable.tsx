import { useState, useEffect, useCallback } from "react";
import { api } from "../api/client";
import type { Trade } from "../types";

export function TradeTable() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pageSize = 20;

  const fetchTrades = useCallback(async (skip: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getTrades(skip, pageSize);
      setTrades(data.trades);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load trades");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTrades(page * pageSize);
  }, [fetchTrades, page]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">
        Recent Trades ({total})
      </h2>

      {loading ? (
        <p className="text-gray-600 text-sm animate-pulse">Loading trades...</p>
      ) : error ? (
        <div className="text-center py-4">
          <p className="text-red-400 text-sm">{error}</p>
          <button
            onClick={() => fetchTrades(page * pageSize)}
            className="mt-2 text-xs text-blue-400 hover:text-blue-300"
          >
            Retry
          </button>
        </div>
      ) : trades.length === 0 ? (
        <p className="text-gray-600 text-sm">
          No trades yet. Trades will appear here once the trading engine executes orders.
        </p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-800">
                  <th className="text-left pb-2">Symbol</th>
                  <th className="text-left pb-2">Action</th>
                  <th className="text-right pb-2">Qty</th>
                  <th className="text-right pb-2">Entry</th>
                  <th className="text-right pb-2">Exit</th>
                  <th className="text-right pb-2">P&L</th>
                  <th className="text-left pb-2">Status</th>
                  <th className="text-left pb-2">Strategy</th>
                  <th className="text-left pb-2">Date</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade) => {
                  const isClosed = trade.status === "CLOSED";
                  const isCancelled = trade.status === "CANCELLED";
                  const actionLabel = isClosed ? "BUY \u2192 SELL" : trade.side;
                  const actionColor = isClosed
                    ? "text-orange-400"
                    : trade.side === "BUY"
                      ? "text-green-400"
                      : "text-red-400";

                  return (
                    <tr key={trade.id} className="border-b border-gray-800/50">
                      <td className="py-2 font-medium text-white">{trade.symbol}</td>
                      <td className={`py-2 ${actionColor}`}>
                        {actionLabel}
                      </td>
                      <td className="py-2 text-right">{trade.quantity}</td>
                      <td className="py-2 text-right">
                        {trade.entry_price?.toFixed(2) ?? "-"}
                      </td>
                      <td className="py-2 text-right text-gray-400">
                        {trade.exit_price?.toFixed(2) ?? "-"}
                      </td>
                      <td
                        className={`py-2 text-right font-medium ${
                          (trade.realized_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"
                        }`}
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
                              : isClosed
                                ? (trade.realized_pnl ?? 0) >= 0
                                  ? "bg-green-900/30 text-green-400"
                                  : "bg-red-900/30 text-red-400"
                                : isCancelled
                                  ? "bg-gray-800 text-gray-500"
                                  : "bg-yellow-900/50 text-yellow-400"
                          }`}
                        >
                          {trade.status}
                        </span>
                      </td>
                      <td className="py-2 text-gray-500 text-xs">{trade.strategy_name}</td>
                      <td className="py-2 text-gray-500 text-xs">
                        {isClosed && trade.closed_at
                          ? new Date(trade.closed_at).toLocaleDateString()
                          : trade.created_at
                            ? new Date(trade.created_at).toLocaleDateString()
                            : "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-800">
              <span className="text-xs text-gray-500">
                Page {page + 1} of {totalPages}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-3 py-1 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-3 py-1 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
