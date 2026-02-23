"""Train the XGBoost ML strategy model on historical data.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --symbols SPY QQQ AAPL
    python scripts/train_model.py --combine  # Combine all symbol data for training
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.data.indicators import compute_features
from app.strategy.feature_pipeline import create_target
from app.strategy.ml_strategy import MLStrategy

DATA_DIR = Path(__file__).parent.parent / "ml" / "data"
DEFAULT_SYMBOLS = [
    # Large-cap tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
    # Large-cap value
    "JNJ", "JPM", "XOM", "BAC", "PG",
    # ETFs (broad, tech, intl)
    "SPY", "QQQ", "IWM", "EFA", "VGK",
    # EU blue chips
    "ASML", "SAP", "SIE.DE", "MC.PA", "AZN.L",
]


def load_data(data_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load CSV data for the given symbols."""
    loaded = {}
    for symbol in symbols:
        path = data_dir / f"{symbol}_1d.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping {symbol}")
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        loaded[symbol] = df
        print(f"  {symbol}: {len(df)} bars")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost trading model")
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Symbols to train on (default: SPY QQQ AAPL MSFT GOOGL NVDA)",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Combine all symbol data into one training set",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Directory containing CSV data files",
    )
    parser.add_argument(
        "--forward-periods", type=int, default=5,
        help="Forward return look-ahead in bars (default: 5)",
    )
    parser.add_argument(
        "--buy-threshold", type=float, default=0.02,
        help="Forward return threshold for BUY label (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--sell-threshold", type=float, default=-0.02,
        help="Forward return threshold for SELL label (default: -0.02 = -2%%)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"Loading data for {len(args.symbols)} symbols from {data_dir}")
    print("-" * 60)
    datasets = load_data(data_dir, args.symbols)

    if not datasets:
        print("No data files found. Run download_historical_data.py first.")
        return

    if args.combine:
        # Compute features PER SYMBOL to avoid rolling indicators crossing
        # symbol boundaries, then concatenate feature DataFrames.
        fwd = args.forward_periods
        buy_thr = args.buy_threshold
        sell_thr = args.sell_threshold
        print(f"\nComputing features per symbol for {len(datasets)} symbols...")
        print(f"Target: forward_periods={fwd}, buy_threshold={buy_thr}, sell_threshold={sell_thr}")
        feature_dfs = []
        for sym, df in datasets.items():
            feat_df = compute_features(df)
            feat_df["target"] = create_target(
                feat_df, forward_periods=fwd,
                buy_threshold=buy_thr, sell_threshold=sell_thr,
            )
            feat_df = feat_df.dropna()
            feature_dfs.append(feat_df)
            print(f"  {sym}: {len(df)} bars → {len(feat_df)} samples")

        combined = pd.concat(feature_dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        print(f"Combined dataset: {len(combined)} samples")

        strategy = MLStrategy()
        result = await strategy.train(combined, features_precomputed=True)
        print(f"\nResult: {result.message}")
        if result.metrics:
            for key, value in result.metrics.items():
                if key != "feature_importance":
                    print(f"  {key}: {value}")
            if "feature_importance" in result.metrics:
                print("\n  Top features:")
                for feat, imp in result.metrics["feature_importance"].items():
                    print(f"    {feat}: {imp:.4f}")
    else:
        # Train on each symbol individually (last one's model is saved)
        strategy = MLStrategy()
        results = {}

        for symbol, df in datasets.items():
            print(f"\nTraining on {symbol} ({len(df)} bars)...")
            result = await strategy.train(df)
            results[symbol] = result
            print(f"  {result.message}")

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for symbol, result in results.items():
            status = "OK" if result.success else "FAIL"
            metrics = result.metrics
            cv = metrics.get("cv_accuracy", 0)
            val = metrics.get("val_accuracy", 0)
            test = metrics.get("test_accuracy", 0)
            print(f"  {symbol}: [{status}] CV={cv:.4f} Val={val:.4f} Test={test:.4f}")

    print("\nTraining complete! Model saved to ml/models/xgboost_model.pkl")


if __name__ == "__main__":
    asyncio.run(main())
