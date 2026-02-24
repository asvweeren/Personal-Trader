"""Walk-Forward Backtest for the ML Trading Strategy.

Trains the model on rolling windows and tests on out-of-sample data
to validate real-world performance. No data leakage: each test window
is strictly after the training window.

Usage:
    python scripts/walk_forward_backtest.py
    python scripts/walk_forward_backtest.py --test-symbols SPY QQQ AAPL MSFT
    python scripts/walk_forward_backtest.py --train-months 18 --test-months 3
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Support running both locally (scripts/ dir) and inside Docker (/app)
_backend = Path(__file__).parent.parent / "backend"
if _backend.exists():
    sys.path.insert(0, str(_backend))
else:
    sys.path.insert(0, "/app")

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.data.indicators import compute_features
from app.strategy.feature_pipeline import create_binary_target
from app.strategy.ml_strategy import MLStrategy

# Broad training universe
TRAIN_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "AMD",
    "NFLX", "CRM", "AVGO",
    "JPM", "BAC", "GS", "JNJ", "UNH", "LLY", "XOM", "CVX", "WMT",
    "HD", "KO", "COST", "PG",
    "SPY", "QQQ", "IWM", "DIA", "EFA", "VGK", "XLF", "XLE",
]

DEFAULT_TEST_SYMBOLS = ["SPY", "QQQ", "AAPL"]


def download_data(symbols: list[str], period: str = "3y") -> dict[str, pd.DataFrame]:
    """Download OHLCV data for symbols via yfinance."""
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, interval="1d", progress=False)
            if df.empty:
                print(f"  {sym}: NO DATA")
                continue
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            keep = ["open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            df.index.name = "timestamp"
            df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            data[sym] = df
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    return data


def create_windows(
    total_bars: int,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    lookback: int = 100,
) -> list[tuple[int, int, int, int]]:
    """Generate (train_start, train_end, test_data_start, test_end) tuples.

    test_data_start includes lookback bars before the actual test period
    so that the backtest engine can warm up indicators without data leakage.
    """
    windows = []
    start = 0
    while start + train_bars + test_bars <= total_bars:
        train_start = start
        train_end = start + train_bars
        test_data_start = max(train_end - lookback, 0)
        test_end = train_end + test_bars
        windows.append((train_start, train_end, test_data_start, test_end))
        start += step_bars
    return windows


async def train_on_window(
    train_datasets: dict[str, pd.DataFrame],
) -> MLStrategy | None:
    """Train a fresh MLStrategy on combined symbol data for one window."""
    feature_dfs = []
    for sym, df in train_datasets.items():
        try:
            feat_df = compute_features(df)
            feat_df["target"] = create_binary_target(
                feat_df, forward_periods=5, buy_threshold=0.015,
            )
            feat_df = feat_df.dropna()
            if len(feat_df) >= 60:
                feature_dfs.append(feat_df)
        except Exception:
            pass

    if not feature_dfs:
        return None

    combined = pd.concat(feature_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    strategy = MLStrategy()
    result = await strategy.train(combined, features_precomputed=True)
    if not result.success:
        return None
    return strategy


async def run_walk_forward(
    test_symbols: list[str],
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    initial_capital: float = 5000.0,
):
    all_symbols = sorted(set(TRAIN_SYMBOLS + test_symbols))
    print(f"Downloading 3Y daily data for {len(all_symbols)} symbols...")
    print("-" * 65)
    all_data = download_data(all_symbols, period="3y")

    if not all_data:
        print("No data downloaded!")
        return

    # Use SPY as date reference
    ref_sym = "SPY" if "SPY" in all_data else list(all_data.keys())[0]
    ref_df = all_data[ref_sym]
    total_bars = len(ref_df)

    train_bars = train_months * 21
    test_bars = test_months * 21
    step_bars = step_months * 21
    lookback = 100

    windows = create_windows(total_bars, train_bars, test_bars, step_bars, lookback)
    print(f"\nTotal bars: {total_bars} ({ref_sym} as reference)")
    print(f"Config: train={train_months}mo, test={test_months}mo, step={step_months}mo")
    print(f"Windows: {len(windows)}")
    print(f"Lookback for indicator warmup: {lookback} bars")

    all_results: dict[str, list[dict]] = {sym: [] for sym in test_symbols}

    for w_idx, (train_start, train_end, test_data_start, test_end) in enumerate(windows):
        train_date_start = ref_df.iloc[train_start]["timestamp"].strftime("%Y-%m-%d")
        train_date_end = ref_df.iloc[min(train_end - 1, total_bars - 1)]["timestamp"].strftime("%Y-%m-%d")
        test_date_start = ref_df.iloc[train_end]["timestamp"].strftime("%Y-%m-%d")
        test_date_end = ref_df.iloc[min(test_end - 1, total_bars - 1)]["timestamp"].strftime("%Y-%m-%d")

        print(f"\n{'=' * 65}")
        print(f"  WINDOW {w_idx + 1}/{len(windows)}")
        print(f"  Train: {train_date_start} -> {train_date_end} ({train_end - train_start} bars)")
        print(f"  Test:  {test_date_start} -> {test_date_end} ({test_end - train_end} bars)")
        print(f"{'=' * 65}")

        # Slice training data per symbol for this window
        train_datasets = {}
        ref_start_ts = ref_df.iloc[train_start]["timestamp"]
        ref_end_ts = ref_df.iloc[min(train_end - 1, total_bars - 1)]["timestamp"]
        for sym in TRAIN_SYMBOLS:
            if sym not in all_data:
                continue
            sym_df = all_data[sym]
            mask = (sym_df["timestamp"] >= ref_start_ts) & (sym_df["timestamp"] <= ref_end_ts)
            window_df = sym_df[mask].copy().reset_index(drop=True)
            if len(window_df) >= 100:
                train_datasets[sym] = window_df

        print(f"  Training on {len(train_datasets)} symbols...")
        strategy = await train_on_window(train_datasets)

        if strategy is None:
            print("  Training FAILED, skipping window")
            continue

        meta = strategy._model_metadata or {}
        precision = meta.get("test_buy_precision", "?")
        pscore = meta.get("test_profit_score", "?")
        samples = meta.get("train_samples", "?")
        print(f"  Model: precision={precision}, profit_score={pscore}, samples={samples}")

        # Backtest each test symbol
        ref_test_start_ts = ref_df.iloc[test_data_start]["timestamp"]
        ref_test_end_ts = ref_df.iloc[min(test_end - 1, total_bars - 1)]["timestamp"]

        for sym in test_symbols:
            if sym not in all_data:
                print(f"  {sym}: no data, skipping")
                continue

            sym_df = all_data[sym]
            mask = (sym_df["timestamp"] >= ref_test_start_ts) & (sym_df["timestamp"] <= ref_test_end_ts)
            test_df = sym_df[mask].copy().reset_index(drop=True)

            if len(test_df) < lookback + 10:
                print(f"  {sym}: insufficient test data ({len(test_df)} bars)")
                continue

            try:
                config = BacktestConfig(
                    strategy=strategy,
                    symbol=sym,
                    start_date=test_date_start,
                    end_date=test_date_end,
                    initial_capital=initial_capital,
                    commission_pct=0.1,
                    slippage_pct=0.05,
                    max_position_pct=20.0,
                    stop_loss_pct=3.0,
                    take_profit_pct=6.0,
                    enable_eod_close=False,
                    trailing_stop_tiers="4.0:1.5,6.0:2.0,8.0:2.5,10.0:3.0",
                )

                engine = BacktestEngine()
                result = await engine.run(config, test_df)
                m = result.metrics

                trades = m.get("total_trades", 0)
                win_rate = m.get("win_rate", 0)
                ret = m.get("total_return_pct", 0)
                sharpe = m.get("sharpe_ratio", 0)
                dd = m.get("max_drawdown_pct", 0)
                pf = m.get("profit_factor", 0)
                bm_ret = (result.benchmark_metrics or {}).get("total_return_pct", 0)

                print(
                    f"  {sym}: {trades} trades, "
                    f"ret={ret:+.2f}%, WR={win_rate:.0f}%, "
                    f"Sharpe={sharpe:.2f}, DD={dd:.2f}%, PF={pf}"
                )

                all_results[sym].append({
                    "window": w_idx + 1,
                    "test_period": f"{test_date_start} -> {test_date_end}",
                    "total_trades": trades,
                    "win_rate": win_rate,
                    "total_return_pct": ret,
                    "sharpe_ratio": sharpe,
                    "max_drawdown_pct": dd,
                    "profit_factor": pf,
                    "final_value": m.get("final_value", initial_capital),
                    "avg_win_pct": m.get("avg_win_pct", 0),
                    "avg_loss_pct": m.get("avg_loss_pct", 0),
                    "benchmark_return": bm_ret,
                })
            except Exception as e:
                print(f"  {sym}: ERROR ({e})")
                import traceback
                traceback.print_exc()

    # ─── Aggregate Report ─────────────────────────────────────────
    print(f"\n\n{'=' * 65}")
    print(f"  WALK-FORWARD RESULTS (Out-of-Sample)")
    print(f"  {len(windows)} windows x {len(test_symbols)} symbols")
    print(f"{'=' * 65}")

    for sym in test_symbols:
        results = all_results[sym]
        if not results:
            print(f"\n  {sym}: No results")
            continue

        n = len(results)
        total_trades = sum(r["total_trades"] for r in results)

        compound = 1.0
        for r in results:
            compound *= 1 + r["total_return_pct"] / 100
        compound_pct = (compound - 1) * 100

        compound_bm = 1.0
        for r in results:
            compound_bm *= 1 + r["benchmark_return"] / 100
        compound_bm_pct = (compound_bm - 1) * 100

        trading_windows = [r for r in results if r["total_trades"] > 0]
        avg_ret = sum(r["total_return_pct"] for r in results) / n
        avg_sharpe = sum(r["sharpe_ratio"] for r in results) / n
        avg_wr = (
            sum(r["win_rate"] for r in trading_windows) / len(trading_windows)
            if trading_windows
            else 0
        )
        worst_dd = min(r["max_drawdown_pct"] for r in results)
        win_windows = sum(1 for r in results if r["total_return_pct"] > 0)

        print(f"\n  -- {sym} {'─' * (55 - len(sym))}")
        print(f"  Windows:           {n} ({win_windows} profitable)")
        print(f"  Total trades:      {total_trades}")
        print(f"  Compound return:   {compound_pct:+.2f}%")
        print(f"  Benchmark (B&H):   {compound_bm_pct:+.2f}%")
        print(f"  Alpha:             {compound_pct - compound_bm_pct:+.2f}%")
        print(f"  Avg return/window: {avg_ret:+.2f}%")
        print(f"  Avg Sharpe:        {avg_sharpe:.2f}")
        print(f"  Avg win rate:      {avg_wr:.1f}%")
        print(f"  Worst drawdown:    {worst_dd:.2f}%")

        hdr = f"  {'Win':<5} {'Test Period':<27} {'#Tr':>4} {'Return':>8} {'B&H':>8} {'WR':>6} {'Sharpe':>7} {'DD':>7}"
        print(f"\n{hdr}")
        print(f"  {'-' * (len(hdr) - 2)}")
        for r in results:
            print(
                f"  W{r['window']:<4} {r['test_period']:<27} "
                f"{r['total_trades']:>4} "
                f"{r['total_return_pct']:>+7.2f}% "
                f"{r['benchmark_return']:>+7.2f}% "
                f"{r['win_rate']:>5.0f}% "
                f"{r['sharpe_ratio']:>7.2f} "
                f"{r['max_drawdown_pct']:>6.2f}%"
            )

    print(f"\n{'=' * 65}")


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--test-symbols", nargs="+", default=DEFAULT_TEST_SYMBOLS)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--initial-capital", type=float, default=5000.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_walk_forward(
            test_symbols=args.test_symbols,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            initial_capital=args.initial_capital,
        )
    )
