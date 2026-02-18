import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { SystemHealth } from "../types";

interface Props {
  wsConnected: boolean;
}

export function SystemStatus({ wsConnected }: Props) {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  const brokerStatus = health?.components.broker.status ?? "unknown";
  const engineStatus = health?.components.trading_engine?.status ?? "unknown";
  const isPaper = health?.components.paper_trading ?? true;

  useEffect(() => {
    const fetchHealth = () => {
      api.getSystemHealth().then(setHealth).catch(() => {});
    };
    fetchHealth();
    // Poll faster when broker is disconnected or reconnecting
    const intervalMs = brokerStatus !== "connected" && health ? 5000 : 30000;
    const interval = setInterval(fetchHealth, intervalMs);
    return () => clearInterval(interval);
  }, [brokerStatus]);

  return (
    <div className="flex items-center gap-4 text-xs">
      {isPaper && (
        <span className="px-2 py-1 bg-yellow-900/50 text-yellow-400 rounded">
          PAPER
        </span>
      )}

      <div className="flex items-center gap-1.5">
        <div
          className={`w-2 h-2 rounded-full ${
            brokerStatus === "connected" ? "bg-green-400"
            : brokerStatus === "reconnecting" ? "bg-yellow-400 animate-pulse"
            : "bg-red-400"
          }`}
        />
        <span className="text-gray-500">Broker</span>
      </div>

      <div className="flex items-center gap-1.5">
        <div
          className={`w-2 h-2 rounded-full ${
            engineStatus === "RUNNING" ? "bg-green-400" : engineStatus === "STOPPED" ? "bg-gray-400" : "bg-yellow-400"
          }`}
        />
        <span className="text-gray-500">Engine</span>
      </div>

      <div className="flex items-center gap-1.5">
        <div
          className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-400" : "bg-red-400"}`}
        />
        <span className="text-gray-500">WS</span>
      </div>

      <div className="flex items-center gap-1.5">
        <div
          className={`w-2 h-2 rounded-full ${
            health?.status === "healthy" ? "bg-green-400" : "bg-yellow-400"
          }`}
        />
        <span className="text-gray-500">System</span>
      </div>
    </div>
  );
}
