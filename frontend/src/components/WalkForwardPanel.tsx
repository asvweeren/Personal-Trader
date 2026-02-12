import { useState } from "react";
import { fetchApi } from "../api/client";

interface WindowResult {
  window_num: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  train_metrics: Record<string, unknown>;
  test_metrics: Record<string, unknown>;
  train_trades: number;
  test_trades: number;
}

interface WFResult {
  windows: WindowResult[];
  aggregate_metrics: Record<string, unknown>;
  degradation_pct: number;
  robustness_score: number;
  num_windows: number;
}

export function WalkForwardPanel() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<WFResult | null>(null);
  const [symbol, setSymbol] = useState("SPY");
  const [error, setError] = useState<string | null>(null);

  const runWalkForward = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchApi("/api/backtest/walk-forward", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy_name: "ml_xgboost",
          symbol,
          train_days: 180,
          test_days: 30,
          step_days: 30,
        }),
      });
      const data = await resp.json();
      // Poll for result
      if (data.id) {
        const poll = async () => {
          const r = await fetchApi(`/api/backtest/${data.id}`);
          const bt = await r.json();
          if (bt.metrics?.status === "running") {
            setTimeout(poll, 3000);
          } else if (bt.metrics?.windows) {
            setResult(bt.metrics as WFResult);
            setLoading(false);
          } else {
            setError(bt.metrics?.error || "Walk-forward failed");
            setLoading(false);
          }
        };
        setTimeout(poll, 3000);
      }
    } catch (e) {
      setError(String(e));
      setLoading(false);
    }
  };

  const robustnessColor = (score: number) => {
    if (score >= 0.7) return "text-green-400";
    if (score >= 0.4) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">Walk-Forward Validation</h2>

      <div className="flex gap-3 mb-4">
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white w-24"
          placeholder="Symbol"
        />
        <button
          onClick={runWalkForward}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white text-sm px-4 py-1.5 rounded"
        >
          {loading ? "Running..." : "Run Analysis"}
        </button>
      </div>

      {error && <p className="text-xs text-red-400 mb-3">{error}</p>}

      {result && (
        <>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-xs text-gray-500">Robustness</div>
              <div className={`text-lg font-mono ${robustnessColor(result.robustness_score)}`}>
                {(result.robustness_score * 100).toFixed(0)}%
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-gray-500">Degradation</div>
              <div className="text-lg font-mono text-gray-300">
                {result.degradation_pct.toFixed(1)}%
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-gray-500">Windows</div>
              <div className="text-lg font-mono text-gray-300">{result.num_windows}</div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-1">#</th>
                  <th className="text-left py-1">Test Period</th>
                  <th className="text-right py-1">Train Return</th>
                  <th className="text-right py-1">Test Return</th>
                  <th className="text-right py-1">Trades</th>
                </tr>
              </thead>
              <tbody>
                {result.windows.map((w) => (
                  <tr key={w.window_num} className="border-b border-gray-800/50">
                    <td className="py-1 text-gray-400">{w.window_num}</td>
                    <td className="py-1 text-gray-400">{w.test_start}</td>
                    <td className="py-1 text-right text-gray-300">
                      {Number(w.train_metrics.total_return_pct ?? 0).toFixed(1)}%
                    </td>
                    <td className={`py-1 text-right ${Number(w.test_metrics.total_return_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {Number(w.test_metrics.total_return_pct ?? 0).toFixed(1)}%
                    </td>
                    <td className="py-1 text-right text-gray-400">{w.test_trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
