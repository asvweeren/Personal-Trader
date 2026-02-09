import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { EngineStatus } from "../types";

const STATE_COLORS: Record<string, string> = {
  RUNNING: "bg-green-900/50 text-green-400",
  STOPPED: "bg-gray-800 text-gray-400",
  STARTING: "bg-blue-900/50 text-blue-400",
  STOPPING: "bg-yellow-900/50 text-yellow-400",
  ERROR: "bg-red-900/50 text-red-400",
  NOT_INITIALIZED: "bg-gray-800 text-gray-500",
};

export function EngineControl() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [toggling, setToggling] = useState(false);

  const fetchStatus = () => {
    api.getEngineStatus().then(setStatus).catch(() => {});
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleToggle = async () => {
    if (!status) return;
    setToggling(true);
    try {
      const result = await api.toggleTrading(!status.trading_enabled);
      setStatus((prev) =>
        prev ? { ...prev, trading_enabled: result.trading_enabled, state: result.state } : prev,
      );
    } catch {
      // ignore
    } finally {
      setToggling(false);
    }
  };

  if (!status) return null;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-gray-400">Trading Engine</h2>
        <span className={`text-xs px-2 py-1 rounded ${STATE_COLORS[status.state] ?? STATE_COLORS.STOPPED}`}>
          {status.state}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-xs text-gray-500">Cycles</div>
          <div className="text-sm font-medium">{status.cycle_count}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Open Trades</div>
          <div className="text-sm font-medium">{status.open_trades}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Pending Orders</div>
          <div className="text-sm font-medium">{status.pending_orders}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Last Cycle</div>
          <div className="text-sm font-medium text-gray-400">
            {status.last_cycle_at
              ? new Date(status.last_cycle_at).toLocaleTimeString()
              : "-"}
          </div>
        </div>
      </div>

      {status.strategies.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-gray-500 mb-1">Strategies</div>
          <div className="flex flex-wrap gap-1">
            {status.strategies.map((s) => (
              <span key={s} className="text-xs px-2 py-0.5 bg-gray-800 text-gray-400 rounded">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {status.symbols.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-gray-500 mb-1">Symbols</div>
          <div className="flex flex-wrap gap-1">
            {status.symbols.map((s) => (
              <span key={s} className="text-xs px-2 py-0.5 bg-gray-800 text-gray-300 rounded">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={handleToggle}
        disabled={toggling || status.state === "NOT_INITIALIZED"}
        className={`w-full py-2 rounded text-sm font-medium transition-colors ${
          status.trading_enabled
            ? "bg-red-900/50 text-red-400 hover:bg-red-900/70"
            : "bg-green-900/50 text-green-400 hover:bg-green-900/70"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {toggling
          ? "..."
          : status.trading_enabled
            ? "Disable Trading"
            : "Enable Trading"}
      </button>
    </div>
  );
}
