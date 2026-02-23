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
        # Combine all data for a single model
        print(f"\nCombining {len(datasets)} symbols into one dataset...")
        combined = pd.concat(datasets.values(), ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        print(f"Combined dataset: {len(combined)} bars")

        strategy = MLStrategy()
        result = await strategy.train(combined)
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
