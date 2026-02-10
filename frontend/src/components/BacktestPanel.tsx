import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { api } from "../api/client";
import type { BacktestSummary, BacktestDetail, BacktestRequest } from "../types";

export function BacktestPanel() {
  const [backtests, setBacktests] = useState<BacktestSummary[]>([]);
  const [selected, setSelected] = useState<BacktestDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<BacktestRequest>({
    strategy_name: "ml_xgboost",
    symbol: "AAPL",
    start_date: "2025-01-01",
    end_date: "2025-06-01",
    initial_capital: 5000,
    commission_pct: 0.1,
    stop_loss_pct: 3.0,
  });

  useEffect(() => {
    api.getBacktests(0, 10).then((r) => setBacktests(r.backtests)).catch(() => {});
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await api.runBacktest(form);
      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const bt = await api.getBacktest(result.id);
          const m = bt.metrics as Record<string, unknown> | undefined;
          if (m && m.status !== "running") {
            clearInterval(poll);
            if (m.status === "error") {
              setError(String(m.error ?? "Backtest failed"));
            }
            setSelected(bt);
            setRunning(false);
            api.getBacktests(0, 10).then((r) => setBacktests(r.backtests)).catch(() => {});
          }
        } catch {
          clearInterval(poll);
          setRunning(false);
        }
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start backtest");
      setRunning(false);
    }
  };

  const handleSelect = async (id: number) => {
    setLoading(true);
    try {
      const bt = await api.getBacktest(id);
      setSelected(bt);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  const metrics = selected?.metrics as Record<string, unknown> | undefined;

  return (
    <div className="space-y-6">
      {/* Run Backtest Form */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-sm font-medium text-gray-400 mb-4">Run Backtest</h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Strategy</label>
            <select
              value={form.strategy_name}
              onChange={(e) => setForm({ ...form, strategy_name: e.target.value })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            >
              <option value="ml_xgboost">ML XGBoost</option>
              <option value="sentiment">Sentiment</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Symbol</label>
            <input
              value={form.symbol}
              onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Start Date</label>
            <input
              type="date"
              value={form.start_date}
              onChange={(e) => setForm({ ...form, start_date: e.target.value })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">End Date</label>
            <input
              type="date"
              value={form.end_date}
              onChange={(e) => setForm({ ...form, end_date: e.target.value })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Capital</label>
            <input
              type="number"
              value={form.initial_capital}
              onChange={(e) => setForm({ ...form, initial_capital: Number(e.target.value) })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Commission %</label>
            <input
              type="number"
              step="0.01"
              value={form.commission_pct}
              onChange={(e) => setForm({ ...form, commission_pct: Number(e.target.value) })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Stop Loss %</label>
            <input
              type="number"
              step="0.1"
              value={form.stop_loss_pct}
              onChange={(e) => setForm({ ...form, stop_loss_pct: Number(e.target.value) })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={running}
              className="w-full py-1.5 rounded text-sm font-medium bg-blue-900/50 text-blue-400 hover:bg-blue-900/70 disabled:opacity-50"
            >
              {running ? "Running..." : "Run Backtest"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-3 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* History */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h2 className="text-sm font-medium text-gray-400 mb-4">History</h2>
          {backtests.length === 0 ? (
            <p className="text-gray-600 text-sm">No backtests yet</p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {backtests.map((bt) => {
                const m = bt.metrics as Record<string, unknown>;
                const isError = m.status === "error";
                const isRunning = m.status === "running";
                return (
                  <button
                    key={bt.id}
                    onClick={() => handleSelect(bt.id)}
                    className={`w-full text-left p-3 rounded border transition-colors ${
                      selected?.id === bt.id
                        ? "border-blue-600 bg-blue-900/20"
                        : "border-gray-800 hover:border-gray-700"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-300">
                        {bt.strategy_name}
                      </span>
                      {isError ? (
                        <span className="text-xs text-red-400">Error</span>
                      ) : isRunning ? (
                        <span className="text-xs text-blue-400">Running</span>
                      ) : (
                        <span
                          className={`text-xs ${
                            Number(m.total_return_pct ?? 0) >= 0
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {Number(m.total_return_pct ?? 0) >= 0 ? "+" : ""}
                          {Number(m.total_return_pct ?? 0).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {(bt.params as Record<string, unknown>).symbol as string} &middot;{" "}
                      {bt.created_at
                        ? new Date(bt.created_at).toLocaleDateString()
                        : "-"}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Detail */}
        <div className="lg:col-span-2 space-y-6">
          {loading ? (
            <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 text-gray-600 text-sm">
              Loading...
            </div>
          ) : selected && metrics ? (
            <>
              {/* Metrics */}
              <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
                <h2 className="text-sm font-medium text-gray-400 mb-4">
                  Results: {selected.strategy_name}
                </h2>
                <div className="grid grid-cols-3 md:grid-cols-5 gap-4">
                  <MetricCard
                    label="Total Return"
                    value={`${Number(metrics.total_return_pct ?? 0).toFixed(2)}%`}
                    color={Number(metrics.total_return_pct ?? 0) >= 0}
                  />
                  <MetricCard
                    label="Final Equity"
                    value={`€${Number(metrics.final_equity ?? 0).toFixed(0)}`}
                  />
                  <MetricCard
                    label="Win Rate"
                    value={`${Number(metrics.win_rate ?? 0).toFixed(1)}%`}
                  />
                  <MetricCard
                    label="Profit Factor"
                    value={String(metrics.profit_factor ?? "-")}
                  />
                  <MetricCard
                    label="Max Drawdown"
                    value={`${Number(metrics.max_drawdown_pct ?? 0).toFixed(2)}%`}
                    color={false}
                  />
                  <MetricCard
                    label="Sharpe"
                    value={Number(metrics.sharpe_ratio ?? 0).toFixed(2)}
                  />
                  <MetricCard
                    label="Sortino"
                    value={Number(metrics.sortino_ratio ?? 0).toFixed(2)}
                  />
                  <MetricCard
                    label="Total Trades"
                    value={String(metrics.total_trades ?? 0)}
                  />
                  <MetricCard
                    label="Commission"
                    value={`€${Number(metrics.total_commission ?? 0).toFixed(2)}`}
                  />
                  <MetricCard
                    label="Expectancy"
                    value={`€${Number(metrics.expectancy ?? 0).toFixed(2)}`}
                  />
                </div>
              </div>

              {/* Equity Curve */}
              {selected.equity_curve && selected.equity_curve.length > 0 && (
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
                  <h2 className="text-sm font-medium text-gray-400 mb-4">Equity Curve</h2>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={selected.equity_curve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis
                        dataKey="timestamp"
                        tick={{ fill: "#6b7280", fontSize: 10 }}
                        axisLine={{ stroke: "#374151" }}
                        tickFormatter={(v: string) =>
                          v ? new Date(v).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : ""
                        }
                      />
                      <YAxis
                        tick={{ fill: "#6b7280", fontSize: 11 }}
                        axisLine={{ stroke: "#374151" }}
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#111827",
                          border: "1px solid #374151",
                          borderRadius: "8px",
                          color: "#f3f4f6",
                        }}
                        formatter={(value: number) => [`€${value.toFixed(2)}`, "Equity"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="equity"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Trades Table */}
              {selected.trades_summary && selected.trades_summary.length > 0 && (
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
                  <h2 className="text-sm font-medium text-gray-400 mb-4">
                    Trades ({selected.trades_summary.length})
                  </h2>
                  <div className="overflow-x-auto max-h-64 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-500 text-xs border-b border-gray-800">
                          <th className="text-left pb-2">Symbol</th>
                          <th className="text-left pb-2">Side</th>
                          <th className="text-right pb-2">Qty</th>
                          <th className="text-right pb-2">Entry</th>
                          <th className="text-right pb-2">Exit</th>
                          <th className="text-right pb-2">P&L</th>
                          <th className="text-left pb-2">Reason</th>
                          <th className="text-right pb-2">Bars</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selected.trades_summary.map((t, i) => (
                          <tr key={i} className="border-b border-gray-800/50">
                            <td className="py-1.5 text-gray-300">{t.symbol}</td>
                            <td
                              className={`py-1.5 ${t.side === "BUY" ? "text-green-400" : "text-red-400"}`}
                            >
                              {t.side}
                            </td>
                            <td className="py-1.5 text-right">{t.quantity}</td>
                            <td className="py-1.5 text-right">{t.entry_price.toFixed(2)}</td>
                            <td className="py-1.5 text-right">{t.exit_price.toFixed(2)}</td>
                            <td
                              className={`py-1.5 text-right ${t.pnl >= 0 ? "text-green-400" : "text-red-400"}`}
                            >
                              {t.pnl >= 0 ? "+" : ""}
                              {t.pnl.toFixed(2)}
                            </td>
                            <td className="py-1.5 text-gray-500 text-xs">{t.exit_reason}</td>
                            <td className="py-1.5 text-right text-gray-500">{t.bars_held}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 text-gray-600 text-sm">
              Select a backtest or run a new one to see results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: boolean;
}) {
  const textColor =
    color === undefined
      ? "text-gray-300"
      : color
        ? "text-green-400"
        : "text-red-400";

  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-sm font-medium ${textColor}`}>{value}</div>
    </div>
  );
}
