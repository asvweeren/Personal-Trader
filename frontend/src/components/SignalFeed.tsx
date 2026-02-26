import { useState, useEffect } from "react";
import type { WSMessage } from "../types";

interface Props {
  lastMessage: WSMessage | null;
}

interface FeedItem {
  id: number;
  type: string;
  message: string;
  timestamp: string;
  severity: "info" | "success" | "warning" | "error";
}

let nextId = 0;

export function SignalFeed({ lastMessage }: Props) {
  const [items, setItems] = useState<FeedItem[]>([]);

  useEffect(() => {
    if (!lastMessage) return;

    const { message, severity } = formatMessage(lastMessage);
    const item: FeedItem = {
      id: nextId++,
      type: lastMessage.type,
      message,
      timestamp: new Date(lastMessage.timestamp).toLocaleTimeString(),
      severity,
    };

    setItems((prev) => [item, ...prev].slice(0, 50));
  }, [lastMessage]);

  const severityColors = {
    info: "text-gray-300",
    success: "text-green-400",
    warning: "text-yellow-400",
    error: "text-red-400",
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">Signal Feed</h2>

      {items.length === 0 ? (
        <p className="text-gray-600 text-sm">
          Waiting for signals... Connect to the WebSocket for real-time updates.
        </p>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {items.map((item) => (
            <div
              key={item.id}
              className="text-xs py-1.5 border-b border-gray-800/50 last:border-0"
            >
              <span className="text-gray-500">{item.timestamp}</span>
              <span className={`ml-2 ${severityColors[item.severity]}`}>
                {item.message}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatMessage(msg: WSMessage): { message: string; severity: FeedItem["severity"] } {
  const data = msg.data;
  switch (msg.type) {
    case "signal.generated": {
      const aiTag = data.ai_modifier && (data.ai_modifier as number) !== 1.0
        ? ` [AI: ${(data.ai_modifier as number).toFixed(1)}x]`
        : "";
      return {
        message: `Signal: ${data.action} ${data.symbol} (${((data.confidence as number) * 100).toFixed(0)}% conf, ${data.strategy})${aiTag}`,
        severity: "info",
      };
    }
    case "order.placed":
      return {
        message: `Order placed: ${data.side} ${data.quantity}x ${data.symbol}`,
        severity: "info",
      };
    case "order.filled":
      return {
        message: `Order filled: ${data.symbol} @ ${data.filled_price}`,
        severity: "success",
      };
    case "order.cancelled":
      return {
        message: `Order cancelled: ${data.symbol} (${data.reason ?? ""})`,
        severity: "warning",
      };
    case "position.closed":
      return {
        message: `Position closed: ${data.symbol} P&L: ${data.realized_pnl}`,
        severity: Number(data.realized_pnl ?? 0) >= 0 ? "success" : "error",
      };
    case "portfolio.updated":
      return {
        message: `Portfolio updated: ${data.positions} positions, value: ${data.total_value}`,
        severity: "info",
      };
    case "risk.daily_stop":
      return {
        message: "DAILY LOSS LIMIT HIT - Trading halted",
        severity: "error",
      };
    case "risk.warning":
      return {
        message: `Risk warning: ${data.message ?? data.description ?? ""}`,
        severity: "warning",
      };
    case "engine.state_change":
      return {
        message: `Engine: ${data.old_state} -> ${data.new_state}`,
        severity: "info",
      };
    case "engine.cycle":
      return {
        message: `Trading cycle #${data.cycle_count} completed`,
        severity: "info",
      };
    case "system.heartbeat":
      return {
        message: `System: broker ${data.broker_connected ? "connected" : "disconnected"}, engine ${data.engine_state}`,
        severity: data.broker_connected ? "info" : "warning",
      };
    default:
      return {
        message: `${msg.type}: ${JSON.stringify(data)}`,
        severity: "info",
      };
  }
}
