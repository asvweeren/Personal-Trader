"""Download historical OHLCV data using yfinance and compute technical indicators.

Usage:
    python scripts/download_historical_data.py
    python scripts/download_historical_data.py --symbols SPY QQQ AAPL
    python scripts/download_historical_data.py --start 2022-01-01 --end 2024-01-01
    python scripts/download_historical_data.py --symbols SPY --interval 1h --period 60d
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Allow importing from the backend package
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.data.indicators import compute_features

# ── Defaults ────────────────────────────────────────────────────

DEFAULT_SYMBOLS = [
    # US ETFs
    "SPY",
    "QQQ",
    "IWM",
    # EU ETFs
    "EFA",
    "VGK",
    # US large-cap tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
]

OUTPUT_DIR = Path(__file__).parent.parent / "ml" / "data"

# ── Helpers ─────────────────────────────────────────────────────


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for a single symbol via yfinance.

    Returns a DataFrame with lowercase column names and a 'timestamp' column,
    matching the convention used throughout the codebase.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

    if df.empty:
        return df

    # Normalise column names to match project convention (lowercase OHLCV)
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Keep only the columns the rest of the codebase expects
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Convert the DatetimeIndex into a regular column
    df.index.name = "timestamp"
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop rows with missing prices (holidays / halted days)
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators using the shared indicator library.

    The project's ``compute_features`` function expects lowercase OHLCV columns
    which we already guarantee in ``download_symbol``.
    """
    if df.empty or len(df) < 50:
        return df

    # compute_features returns a copy with indicator columns appended
    return compute_features(df)


def save_data(
    df: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    interval: str = "1d",
) -> Path:
    """Save a DataFrame to CSV inside *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    interval_tag = interval.replace(" ", "")
    filename = f"{symbol}_{interval_tag}.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


# ── CLI ─────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical market data and compute technical indicators.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="List of ticker symbols to download (default: popular ETFs + tech stocks).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format (default: 2 years ago).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"],
        help="Bar interval (default: 1d). Note: intraday intervals are limited to ~60 days.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save CSV files (default: ml/data/).",
    )
    parser.add_argument(
        "--no-indicators",
        action="store_true",
        help="Skip computing technical indicators.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    print(f"Downloading {len(args.symbols)} symbols  [{args.start} -> {args.end}]")
    print(f"Interval: {args.interval}  |  Output: {output_dir}")
    print("-" * 60)

    success = 0
    failed: list[str] = []

    for symbol in args.symbols:
        print(f"  {symbol} ... ", end="", flush=True)
        try:
            df = download_symbol(symbol, start=args.start, end=args.end, interval=args.interval)
            if df.empty:
                print("no data returned")
                failed.append(symbol)
                continue

            if not args.no_indicators:
                df = add_indicators(df)

            path = save_data(df, symbol, output_dir, interval=args.interval)
            print(f"{len(df)} bars -> {path.name}")
            success += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed.append(symbol)

    print("-" * 60)
    print(f"Done. {success} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed symbols: {', '.join(failed)}")


if __name__ == "__main__":
    main()
