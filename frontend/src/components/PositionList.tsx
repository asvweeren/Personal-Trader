import type { Position } from "../types";

interface Props {
  positions: Position[];
}

export function PositionList({ positions }: Props) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
      <h2 className="text-sm font-medium text-gray-400 mb-4">
        Open Positions ({positions.length})
      </h2>

      {positions.length === 0 ? (
        <p className="text-gray-600 text-sm">No open positions</p>
      ) : (
        <div className="space-y-3">
          {positions.map((pos) => (
            <div
              key={pos.symbol}
              className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0"
            >
              <div>
                <div className="font-medium text-white">{pos.symbol}</div>
                <div className="text-xs text-gray-500">
                  {pos.quantity} shares @ {pos.avg_cost.toFixed(2)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium">
                  {pos.market_value.toFixed(2)}
                </div>
                <div
                  className={`text-xs ${pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
                >
                  {pos.unrealized_pnl >= 0 ? "+" : ""}
                  {pos.unrealized_pnl.toFixed(2)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
