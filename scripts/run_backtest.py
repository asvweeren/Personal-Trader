"""Run a backtest using downloaded historical data and a simple SMA crossover strategy.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbol SPY
    python scripts/run_backtest.py --symbol AAPL --fast-period 10 --slow-period 30
    python scripts/run_backtest.py --data-dir ml/data --results-dir ml/data/backtest_results
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow importing from the backend package
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.data.indicators import compute_features, sma
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal

# ── SMA Crossover Strategy ──────────────────────────────────────


class SMACrossoverStrategy(Strategy):
    """Simple moving average crossover strategy used as a baseline.

    Generates a BUY signal when the fast SMA crosses above the slow SMA,
    and a SELL signal when the fast SMA crosses below the slow SMA.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self._fast_period = fast_period
        self._slow_period = slow_period

    @property
    def name(self) -> str:
        return f"sma_crossover_{self._fast_period}_{self._slow_period}"

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals: list[TradingSignal] = []

        for symbol, df in market_data.ohlcv.items():
            if df.empty or len(df) < self._slow_period + 2:
                continue

            close = df["close"]
            fast = sma(close, self._fast_period)
            slow = sma(close, self._slow_period)

            # Look at the last two bars for a crossover
            prev_fast = fast.iloc[-2]
            prev_slow = slow.iloc[-2]
            curr_fast = fast.iloc[-1]
            curr_slow = slow.iloc[-1]

            if pd.isna(prev_fast) or pd.isna(prev_slow):
                continue

            # Bullish crossover: fast crosses above slow
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        action=SignalAction.BUY,
                        confidence=0.7,
                        strategy_name=self.name,
                        metadata={
                            "fast_sma": round(float(curr_fast), 4),
                            "slow_sma": round(float(curr_slow), 4),
                        },
                    )
                )

            # Bearish crossover: fast crosses below slow
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        action=SignalAction.SELL,
                        confidence=0.7,
                        strategy_name=self.name,
                        metadata={
                            "fast_sma": round(float(curr_fast), 4),
                            "slow_sma": round(float(curr_slow), 4),
                        },
                    )
                )

        return signals


# ── Helpers ─────────────────────────────────────────────────────


def load_historical_data(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Load a previously downloaded CSV for *symbol* from *data_dir*.

    Tries common naming patterns produced by the download script.
    """
    candidates = [
        data_dir / f"{symbol}_1d.csv",
        data_dir / f"{symbol}_1h.csv",
        data_dir / f"{symbol}_daily.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            return df

    # Fall back to any CSV containing the symbol name
    for path in sorted(data_dir.glob(f"{symbol}*.csv")):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    raise FileNotFoundError(
        f"No data file found for {symbol} in {data_dir}. "
        "Run download_historical_data.py first."
    )


def format_metrics(metrics: dict, indent: int = 2) -> str:
    """Pretty-print a metrics dict."""
    lines: list[str] = []
    pad = " " * indent
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            lines.append(f"{pad}{label}: {value:>10.2f}")
        else:
            lines.append(f"{pad}{label}: {value!s:>10}")
    return "\n".join(lines)


def save_results(result, output_dir: Path) -> Path:
    """Serialize backtest results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{result.strategy_name}_{result.symbol}_{timestamp}.json"
    path = output_dir / filename

    payload = {
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "config": result.config,
        "metrics": result.metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "total_bars": result.total_bars,
        "duration_days": result.duration_days,
        "trade_count": len(result.trades),
        "trades": [
            {
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "commission": t.commission,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
            for t in result.trades
        ],
        "equity_curve": result.equity_curve,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return path


# ── CLI ─────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an SMA crossover backtest on downloaded historical data.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Ticker symbol to backtest (default: SPY).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "ml" / "data"),
        help="Directory containing CSV data files (default: ml/data/).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "ml" / "data" / "backtest_results"),
        help="Directory to save JSON results (default: ml/data/backtest_results/).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=5000.0,
        help="Starting capital in USD (default: 5000).",
    )
    parser.add_argument(
        "--fast-period",
        type=int,
        default=20,
        help="Fast SMA period (default: 20).",
    )
    parser.add_argument(
        "--slow-period",
        type=int,
        default=50,
        help="Slow SMA period (default: 50).",
    )
    parser.add_argument(
        "--commission-pct",
        type=float,
        default=0.1,
        help="Commission as percentage of trade value (default: 0.1%%).",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.05,
        help="Slippage as percentage of price (default: 0.05%%).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=3.0,
        help="Stop loss percentage below entry price (default: 3.0%%).",
    )
    return parser.parse_args(argv)


async def run_backtest(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    # 1. Load data
    print(f"Loading data for {args.symbol} from {data_dir} ...")
    df = load_historical_data(data_dir, args.symbol)
    print(f"  Loaded {len(df)} bars  [{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}]")

    # Ensure indicators are present (they may already be in the CSV)
    if "sma_20" not in df.columns and len(df) >= 50:
        print("  Computing technical indicators ...")
        df = compute_features(df)

    # 2. Build strategy and config
    strategy = SMACrossoverStrategy(
        fast_period=args.fast_period,
        slow_period=args.slow_period,
    )

    config = BacktestConfig(
        strategy=strategy,
        symbol=args.symbol,
        start_date=str(df["timestamp"].iloc[0]),
        end_date=str(df["timestamp"].iloc[-1]),
        initial_capital=args.initial_capital,
        commission_pct=args.commission_pct,
        slippage_pct=args.slippage_pct,
        stop_loss_pct=args.stop_loss_pct,
        min_bars=args.slow_period + 5,  # Ensure enough lookback for SMA calculation
    )

    # 3. Run backtest
    print(f"\nRunning backtest: {strategy.name}")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Commission: {config.commission_pct}%  |  Slippage: {config.slippage_pct}%")
    print(f"  Stop loss: {config.stop_loss_pct}%")
    print("-" * 60)

    engine = BacktestEngine()
    result = await engine.run(config, df)

    # 4. Print results
    print("\n=== PERFORMANCE METRICS ===")
    print(format_metrics(result.metrics))

    print("\n=== BENCHMARK COMPARISON (Buy & Hold) ===")
    print(format_metrics(result.benchmark_metrics))

    print(f"\n=== TRADE SUMMARY ===")
    print(f"  Total trades: {len(result.trades)}")
    print(f"  Duration: {result.duration_days} days  |  Bars processed: {result.total_bars}")

    if result.trades:
        winning = [t for t in result.trades if t.pnl > 0]
        losing = [t for t in result.trades if t.pnl <= 0]
        print(f"  Winning: {len(winning)}  |  Losing: {len(losing)}")

        # Show last 5 trades
        print("\n  Last 5 trades:")
        for trade in result.trades[-5:]:
            direction = "+" if trade.pnl > 0 else ""
            print(
                f"    {trade.exit_time}  {trade.side:4s}  "
                f"qty={trade.quantity}  entry={trade.entry_price:.2f}  "
                f"exit={trade.exit_price:.2f}  "
                f"pnl={direction}{trade.pnl:.2f}  ({trade.exit_reason})"
            )

    # 5. Save results
    path = save_results(result, results_dir)
    print(f"\nResults saved to {path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
