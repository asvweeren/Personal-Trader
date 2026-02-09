"""Download historical data from IBKR for model training."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.broker.ibkr_adapter import IBKRAdapter
from app.config import settings

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ", "IWM"]
OUTPUT_DIR = Path("ml/data")


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    broker = IBKRAdapter(
        host=settings.ibkr_host,
        port=settings.ibkr_port,
        client_id=settings.ibkr_client_id,
    )

    try:
        await broker.connect()
        print(f"Connected to IBKR at {settings.ibkr_host}:{settings.ibkr_port}")

        for symbol in SYMBOLS:
            print(f"Downloading {symbol}...")
            df = await broker.get_historical_data(
                symbol=symbol,
                duration="1 Y",
                bar_size="1 hour",
            )
            if not df.empty:
                output_path = OUTPUT_DIR / f"{symbol}_1h.parquet"
                df.to_parquet(output_path, index=False)
                print(f"  Saved {len(df)} bars to {output_path}")
            else:
                print(f"  No data received for {symbol}")

    finally:
        await broker.disconnect()

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
