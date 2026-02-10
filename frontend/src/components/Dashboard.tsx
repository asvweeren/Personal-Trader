import { useState } from "react";
import { usePortfolio } from "../hooks/usePortfolio";
import { useWebSocket } from "../hooks/useWebSocket";
import { useAuth } from "../contexts/AuthContext";
import { PortfolioCard } from "./PortfolioCard";
import { PositionList } from "./PositionList";
import { TradeTable } from "./TradeTable";
import { PnLChart } from "./PnLChart";
import { RiskMetrics } from "./RiskMetrics";
import { SignalFeed } from "./SignalFeed";
import { SystemStatus } from "./SystemStatus";
import { EngineControl } from "./EngineControl";
import { BacktestPanel } from "./BacktestPanel";
import { ValidationDashboard } from "./ValidationDashboard";
import { SettingsPage } from "./SettingsPage";

type Tab = "overview" | "trades" | "backtest" | "validation" | "settings";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "trades", label: "Trades" },
  { id: "backtest", label: "Backtest" },
  { id: "validation", label: "Validation" },
  { id: "settings", label: "Settings" },
];

export function Dashboard() {
  const [tab, setTab] = useState<Tab>("overview");
  const { logout } = useAuth();
  const { portfolio, performance, risk, loading, error } = usePortfolio();
  const ws = useWebSocket(
    `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/live`,
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="animate-pulse text-gray-400 text-lg">Loading dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-400 text-lg">Error: {error}</div>
          <div className="text-gray-500 text-sm mt-2">
            Make sure the backend is running on port 8000
          </div>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div>
            <h1 className="text-xl font-bold text-white">AI Trader Dashboard</h1>
            <p className="text-sm text-gray-500">Automated Day Trading System</p>
          </div>
          <div className="flex items-center gap-4">
            <SystemStatus wsConnected={ws.connected} />
            <button
              onClick={logout}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors px-2 py-1 rounded border border-gray-800 hover:border-gray-700"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="border-b border-gray-800 px-6">
        <div className="max-w-7xl mx-auto flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === t.id
                  ? "border-blue-500 text-blue-400"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {tab === "overview" && (
          <>
            {/* Top row: Portfolio + Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <PortfolioCard portfolio={portfolio} performance={performance} />
              <div className="lg:col-span-2">
                <PnLChart />
              </div>
            </div>

            {/* Middle row: Risk + Engine + Positions */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <RiskMetrics risk={risk} />
              <EngineControl />
              <PositionList positions={portfolio?.positions ?? []} />
            </div>

            {/* Bottom row: Signal Feed */}
            <SignalFeed lastMessage={ws.lastMessage} />
          </>
        )}

        {tab === "trades" && <TradeTable />}

        {tab === "backtest" && <BacktestPanel />}

        {tab === "validation" && <ValidationDashboard />}

        {tab === "settings" && <SettingsPage />}
      </main>
    </div>
  );
}
