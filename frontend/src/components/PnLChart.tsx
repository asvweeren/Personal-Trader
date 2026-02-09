import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import { api } from "../api/client";

interface ChartPoint {
  date: string;
  value: number;
  pnl: number;
}

export function PnLChart() {
  const [data, setData] = useState<ChartPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<"value" | "pnl">("value");

  useEffect(() => {
    api
      .getPortfolioSnapshots(200)
      .then((snapshots) => {
        const points: ChartPoint[] = snapshots.map((s) => ({
          date: s.timestamp
            ? new Date(s.timestamp).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
              })
            : "",
          value: s.total_value,
          pnl: s.daily_pnl,
        }));
        setData(points);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const hasData = data.length > 0;
  const startValue = hasData ? data[0]!.value : 0;
  const endValue = hasData ? data[data.length - 1]!.value : 0;
  const totalReturn = startValue > 0 ? ((endValue - startValue) / startValue) * 100 : 0;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-medium text-gray-400">Portfolio Value</h2>
          {hasData && (
            <span
              className={`text-xs ${totalReturn >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {totalReturn >= 0 ? "+" : ""}
              {totalReturn.toFixed(2)}%
            </span>
          )}
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setView("value")}
            className={`text-xs px-2 py-1 rounded ${
              view === "value"
                ? "bg-blue-900/50 text-blue-400"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            Value
          </button>
          <button
            onClick={() => setView("pnl")}
            className={`text-xs px-2 py-1 rounded ${
              view === "pnl"
                ? "bg-blue-900/50 text-blue-400"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            Daily P&L
          </button>
        </div>
      </div>

      {loading ? (
        <div className="h-[250px] flex items-center justify-center text-gray-600 text-sm">
          Loading chart data...
        </div>
      ) : !hasData ? (
        <div className="h-[250px] flex items-center justify-center text-gray-600 text-sm">
          No snapshot data yet. Start the trading engine to collect data.
        </div>
      ) : view === "value" ? (
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#6b7280", fontSize: 11 }}
              axisLine={{ stroke: "#374151" }}
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
              formatter={(value: number) => [`€${value.toFixed(2)}`, "Value"]}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#colorValue)"
              dot={false}
              activeDot={{ r: 4, fill: "#3b82f6" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#6b7280", fontSize: 11 }}
              axisLine={{ stroke: "#374151" }}
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
              formatter={(value: number) => [
                `${value >= 0 ? "+" : ""}€${value.toFixed(2)}`,
                "Daily P&L",
              ]}
            />
            <Line
              type="monotone"
              dataKey="pnl"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#8b5cf6" }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
