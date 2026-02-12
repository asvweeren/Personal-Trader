import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { ScreeningResult, ScreeningCandidate } from "../types";

type SortKey = keyof Pick<
  ScreeningCandidate,
  "score" | "momentum_score" | "volume_score" | "volatility_score" | "price" | "change_5d_pct" | "avg_volume"
>;

export function ScreenerPanel() {
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    api
      .getScreenerLatest()
      .then(setResult)
      .catch(() => setError("Failed to load screening data"))
      .finally(() => setLoading(false));
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const data = await api.runScreener();
      setResult(data);
    } catch {
      setError("Screening failed. Check backend logs.");
    } finally {
      setRunning(false);
    }
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sorted = [...(result?.candidates ?? [])].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    return sortAsc ? av - bv : bv - av;
  });

  if (loading) {
    return (
      <div className="animate-pulse text-gray-400 py-8 text-center">
        Loading screener...
      </div>
    );
  }

  const SortHeader = ({ label, field }: { label: string; field: SortKey }) => (
    <th
      className="px-3 py-3 text-left text-xs font-medium text-gray-400 uppercase cursor-pointer hover:text-gray-200 select-none"
      onClick={() => handleSort(field)}
    >
      {label} {sortKey === field ? (sortAsc ? "\u25B2" : "\u25BC") : ""}
    </th>
  );

  const ScoreBar = ({ value, color }: { value: number; color: string }) => (
    <div className="flex items-center gap-2">
      <div className="w-16 bg-gray-700 rounded-full h-1.5">
        <div
          className={`h-1.5 rounded-full ${color}`}
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-10 text-right">
        {(value * 100).toFixed(0)}
      </span>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Stock Screener</h2>
          {result?.screening_date && (
            <p className="text-sm text-gray-500">
              Last run: {result.screening_date} &middot; {result.total_scanned} scanned &middot;{" "}
              {result.candidates.length} selected
            </p>
          )}
          {!result?.screening_date && (
            <p className="text-sm text-gray-500">No screening data yet. Run a scan to get started.</p>
          )}
        </div>
        <button
          onClick={handleRun}
          disabled={running}
          className="px-4 py-2 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {running ? "Screening..." : "Run Screener"}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-800 text-red-300 px-4 py-2 rounded text-sm">
          {error}
        </div>
      )}

      {/* Table */}
      {sorted.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-800/50">
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                    #
                  </th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                    Symbol
                  </th>
                  <SortHeader label="Score" field="score" />
                  <SortHeader label="Momentum" field="momentum_score" />
                  <SortHeader label="Volume" field="volume_score" />
                  <SortHeader label="Volatility" field="volatility_score" />
                  <SortHeader label="Price" field="price" />
                  <SortHeader label="5d Chg%" field="change_5d_pct" />
                  <SortHeader label="Avg Vol" field="avg_volume" />
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                    Sector
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {sorted.map((c, i) => (
                  <tr key={c.symbol} className="hover:bg-gray-800/30">
                    <td className="px-3 py-2 text-gray-500">{i + 1}</td>
                    <td className="px-3 py-2 font-medium text-white">{c.symbol}</td>
                    <td className="px-3 py-2">
                      <span
                        className={`font-mono text-sm ${
                          c.score >= 0.6
                            ? "text-green-400"
                            : c.score >= 0.4
                              ? "text-yellow-400"
                              : "text-red-400"
                        }`}
                      >
                        {(c.score * 100).toFixed(1)}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <ScoreBar value={c.momentum_score} color="bg-blue-500" />
                    </td>
                    <td className="px-3 py-2">
                      <ScoreBar value={c.volume_score} color="bg-purple-500" />
                    </td>
                    <td className="px-3 py-2">
                      <ScoreBar value={c.volatility_score} color="bg-orange-500" />
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-300">
                      ${c.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`font-mono ${
                          c.change_5d_pct >= 0 ? "text-green-400" : "text-red-400"
                        }`}
                      >
                        {c.change_5d_pct >= 0 ? "+" : ""}
                        {c.change_5d_pct.toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-400 font-mono">
                      {c.avg_volume >= 1_000_000
                        ? `${(c.avg_volume / 1_000_000).toFixed(1)}M`
                        : `${(c.avg_volume / 1_000).toFixed(0)}K`}
                    </td>
                    <td className="px-3 py-2">
                      <span className="inline-block px-2 py-0.5 text-xs rounded bg-gray-800 text-gray-400">
                        {c.sector}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!sorted.length && !loading && !error && (
        <div className="text-center py-12 text-gray-500">
          No candidates found. Run the screener to scan the market.
        </div>
      )}
    </div>
  );
}
