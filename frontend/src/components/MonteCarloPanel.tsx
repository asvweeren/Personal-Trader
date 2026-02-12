import { useState } from "react";
import { api } from "../api/client";

interface MCResult {
  percentiles: Record<string, number>;
  max_drawdown_dist: Record<string, number>;
  ruin_probability: number;
  median_return_pct: number;
  equity_bands: { trade_num: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
}

interface Props {
  backtestId?: number;
}

export function MonteCarloPanel({ backtestId }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MCResult | null>(null);
  const [btId, setBtId] = useState(backtestId?.toString() || "");
  const [error, setError] = useState<string | null>(null);

  const runMonteCarlo = async () => {
    const id = parseInt(btId);
    if (!id) {
      setError("Enter a valid backtest ID");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.runMonteCarlo({ backtest_id: id, num_simulations: 1000 });
      if (data.result) {
        setResult(data.result as unknown as MCResult);
      } else {
        setError("Monte Carlo failed");
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">Monte Carlo Simulation</h2>

      <div className="flex gap-3 mb-4">
        <input
          type="number"
          value={btId}
          onChange={(e) => setBtId(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white w-28"
          placeholder="Backtest ID"
        />
        <button
          onClick={runMonteCarlo}
          disabled={loading}
          className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 text-white text-sm px-4 py-1.5 rounded"
        >
          {loading ? "Simulating..." : "Run 1000 Sims"}
        </button>
      </div>

      {error && <p className="text-xs text-red-400 mb-3">{error}</p>}

      {result && (
        <>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <div className="text-xs text-gray-500 mb-1">Median Return</div>
              <div className={`text-xl font-mono ${result.median_return_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                {result.median_return_pct.toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Ruin Probability</div>
              <div className={`text-xl font-mono ${result.ruin_probability < 0.05 ? "text-green-400" : result.ruin_probability < 0.15 ? "text-yellow-400" : "text-red-400"}`}>
                {(result.ruin_probability * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="mb-4">
            <div className="text-xs text-gray-500 mb-2">Final Equity Distribution</div>
            <div className="space-y-1">
              {Object.entries(result.percentiles)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([pct, value]) => (
                  <div key={pct} className="flex justify-between text-xs">
                    <span className="text-gray-400">P{(Number(pct) * 100).toFixed(0)}</span>
                    <span className="text-gray-300 font-mono">${Number(value).toFixed(0)}</span>
                  </div>
                ))}
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-500 mb-2">Max Drawdown Distribution</div>
            <div className="space-y-1">
              {Object.entries(result.max_drawdown_dist)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([pct, value]) => (
                  <div key={pct} className="flex justify-between text-xs">
                    <span className="text-gray-400">P{(Number(pct) * 100).toFixed(0)}</span>
                    <span className="text-red-400 font-mono">{Number(value).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </div>

          {result.equity_bands.length > 0 && (
            <div className="mt-4">
              <div className="text-xs text-gray-500 mb-2">Equity Bands (P5-P95)</div>
              <div className="h-24 flex items-end gap-px">
                {result.equity_bands.map((band, i) => {
                  const maxVal = Math.max(...result.equity_bands.map(b => b.p95));
                  const minVal = Math.min(...result.equity_bands.map(b => b.p5));
                  const range = maxVal - minVal || 1;
                  const p5h = ((band.p5 - minVal) / range) * 100;
                  const p50h = ((band.p50 - minVal) / range) * 100;
                  const p95h = ((band.p95 - minVal) / range) * 100;
                  return (
                    <div key={i} className="flex-1 relative" style={{ height: "100%" }}>
                      <div
                        className="absolute bottom-0 left-0 right-0 bg-purple-900/30 rounded-t"
                        style={{ bottom: `${p5h}%`, height: `${p95h - p5h}%` }}
                      />
                      <div
                        className="absolute left-0 right-0 bg-purple-500 rounded"
                        style={{ bottom: `${p50h}%`, height: "2px" }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
