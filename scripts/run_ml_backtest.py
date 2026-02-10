"""Run a backtest using the trained XGBoost ML model on historical data.

Usage:
    python scripts/run_ml_backtest.py
    python scripts/run_ml_backtest.py --symbol AAPL
    python scripts/run_ml_backtest.py --symbol SPY --confidence 0.5
    python scripts/run_ml_backtest.py --all-symbols  # Run on all available symbols
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.data.indicators import compute_features
from app.strategy.ml_strategy import MLStrategy

DATA_DIR = Path(__file__).parent.parent / "ml" / "data"
MODEL_PATH = Path(__file__).parent.parent / "ml" / "models" / "xgboost_model.pkl"
RESULTS_DIR = Path(__file__).parent.parent / "ml" / "data" / "backtest_results"


def load_data(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Load CSV data for a symbol."""
    path = data_dir / f"{symbol}_1d.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "sma_10" not in df.columns and len(df) >= 50:
        df = compute_features(df)
    return df


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


def save_results(result, symbol: str, output_dir: Path) -> Path:
    """Save backtest results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"ml_xgboost_{symbol}_{timestamp}.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XGBoost ML model backtest on historical data."
    )
    parser.add_argument(
        "--symbol", type=str, default="SPY",
        help="Ticker symbol to backtest (default: SPY).",
    )
    parser.add_argument(
        "--all-symbols", action="store_true",
        help="Run backtest on all available symbols in data dir.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence threshold for ML signals (default: 0.5).",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=5000.0,
        help="Starting capital in USD (default: 5000).",
    )
    parser.add_argument(
        "--stop-loss-pct", type=float, default=3.0,
        help="Stop loss percentage (default: 3.0%%).",
    )
    parser.add_argument(
        "--commission-pct", type=float, default=0.1,
        help="Commission percentage (default: 0.1%%).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Directory containing CSV data files.",
    )
    parser.add_argument(
        "--model-path", type=str, default=str(MODEL_PATH),
        help="Path to trained XGBoost model .pkl file.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help="Directory to save JSON results.",
    )
    return parser.parse_args()


async def run_single_backtest(
    symbol: str, args: argparse.Namespace, strategy: MLStrategy
) -> dict | None:
    """Run backtest for a single symbol, return summary dict or None on error."""
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    try:
        df = load_data(data_dir, symbol)
    except FileNotFoundError as e:
        print(f"  SKIP {symbol}: {e}")
        return None

    if len(df) < 60:
        print(f"  SKIP {symbol}: only {len(df)} bars (need 60+)")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} ({len(df)} bars)")
    print(f"  {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"{'='*60}")

    config = BacktestConfig(
        strategy=strategy,
        symbol=symbol,
        start_date=str(df["timestamp"].iloc[0]),
        end_date=str(df["timestamp"].iloc[-1]),
        initial_capital=args.initial_capital,
        commission_pct=args.commission_pct,
        slippage_pct=0.05,
        stop_loss_pct=args.stop_loss_pct,
        min_bars=55,
    )

    engine = BacktestEngine()
    result = await engine.run(config, df)

    # Print results
    print("\n  PERFORMANCE METRICS:")
    print(format_metrics(result.metrics, indent=4))

    print("\n  BENCHMARK (Buy & Hold):")
    print(format_metrics(result.benchmark_metrics, indent=4))

    print(f"\n  TRADES: {len(result.trades)}")
    print(f"  Duration: {result.duration_days} days | Bars: {result.total_bars}")

    if result.trades:
        winning = [t for t in result.trades if t.pnl > 0]
        losing = [t for t in result.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in result.trades)
        print(f"  Winning: {len(winning)} | Losing: {len(losing)}")
        print(f"  Total P&L: ${total_pnl:.2f}")

        # Show last 5 trades
        print("\n  Last 5 trades:")
        for trade in result.trades[-5:]:
            sign = "+" if trade.pnl > 0 else ""
            print(
                f"    {trade.exit_time}  {trade.side:4s}  "
                f"qty={trade.quantity}  entry={trade.entry_price:.2f}  "
                f"exit={trade.exit_price:.2f}  "
                f"pnl={sign}{trade.pnl:.2f}  ({trade.exit_reason})"
            )

    # Save results
    path = save_results(result, symbol, results_dir)
    print(f"\n  Saved: {path.name}")

    return {
        "symbol": symbol,
        "trades": len(result.trades),
        "total_return": result.metrics.get("total_return_pct", 0),
        "sharpe": result.metrics.get("sharpe_ratio", 0),
        "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
        "win_rate": result.metrics.get("win_rate", 0),
        "benchmark_return": result.benchmark_metrics.get("benchmark_return_pct", 0),
    }


async def main():
    args = parse_args()
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run `python scripts/train_model.py --combine` first.")
        sys.exit(1)

    print("ML STRATEGY BACKTEST")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  Capital: ${args.initial_capital:,.2f}")
    print(f"  Stop loss: {args.stop_loss_pct}%")

    strategy = MLStrategy(
        confidence_threshold=args.confidence,
        model_path=str(model_path),
    )

    if strategy._model is None:
        print("ERROR: Failed to load model.")
        sys.exit(1)

    print(f"  Features: {len(strategy._feature_columns)}")
    print(f"  Model info: {strategy._model_metadata.get('trained_at', 'unknown')}")

    # Determine symbols to test
    if args.all_symbols:
        csvs = sorted(data_dir.glob("*_1d.csv"))
        symbols = [p.stem.replace("_1d", "") for p in csvs]
    else:
        symbols = [args.symbol]

    print(f"\n  Symbols: {', '.join(symbols)}")

    # Run backtests
    summaries = []
    for symbol in symbols:
        summary = await run_single_backtest(symbol, args, strategy)
        if summary:
            summaries.append(summary)

    # Print comparison table
    if len(summaries) > 1:
        print("\n\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        print(
            f"  {'Symbol':<8} {'Trades':>6} {'Return':>8} {'Sharpe':>8} "
            f"{'MaxDD':>8} {'WinRate':>8} {'B&H Ret':>8}"
        )
        print("  " + "-" * 56)
        for s in summaries:
            print(
                f"  {s['symbol']:<8} {s['trades']:>6} "
                f"{s['total_return']:>7.2f}% {s['sharpe']:>8.2f} "
                f"{s['max_drawdown']:>7.2f}% {s['win_rate']:>7.1f}% "
                f"{s['benchmark_return']:>7.2f}%"
            )

        # Averages
        avg_ret = sum(s["total_return"] for s in summaries) / len(summaries)
        avg_sharpe = sum(s["sharpe"] for s in summaries) / len(summaries)
        avg_bh = sum(s["benchmark_return"] for s in summaries) / len(summaries)
        print("  " + "-" * 56)
        print(
            f"  {'AVG':<8} {'':>6} {avg_ret:>7.2f}% {avg_sharpe:>8.2f} "
            f"{'':>8} {'':>8} {avg_bh:>7.2f}%"
        )

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
