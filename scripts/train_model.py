"""Train the ML strategy model on historical data."""

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.strategy.ml_strategy import MLStrategy

DATA_DIR = Path("ml/data")


async def main():
    strategy = MLStrategy()

    # Combine all available data files
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        print("No data files found in ml/data/. Run download_historical_data.py first.")
        return

    for file in parquet_files:
        symbol = file.stem.split("_")[0]
        print(f"Training on {symbol} data from {file}...")

        df = pd.read_parquet(file)
        print(f"  {len(df)} bars loaded")

        result = await strategy.train(df)
        print(f"  Result: {result.message}")
        if result.metrics:
            print(f"  Metrics: {result.metrics}")

    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
