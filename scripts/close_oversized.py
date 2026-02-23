"""Close oversized positions to bring portfolio within risk limits.

Automatically detects positions exceeding max_position_pct and sells the
excess shares. Dry-run by default — use --execute to actually place orders.

Usage:
    # From backend dir (dry-run, shows what would happen):
    uv run python -m scripts.close_oversized

    # Actually execute the sells:
    uv run python -m scripts.close_oversized --execute

    # Custom max position size (default from settings: 10%):
    uv run python -m scripts.close_oversized --max-pct 15 --execute

    # Close specific symbols completely:
    uv run python -m scripts.close_oversized --close SHOP APH --execute
"""
import argparse
import asyncio
import math
import sys

from app.config import settings
from app.broker.ibkr_adapter import IBKRAdapter
from app.broker.base import OrderRequest, OrderSide, OrderType, Position


def build_trim_orders(
    positions: list[Position],
    total_value: float,
    max_pct: float,
    close_symbols: list[str] | None = None,
) -> list[tuple[Position, int, str]]:
    """Determine which positions to trim and by how much.

    Returns list of (position, sell_qty, reason) tuples.
    """
    actions: list[tuple[Position, int, str]] = []

    for pos in positions:
        if pos.quantity <= 0:
            continue

        # Full close if explicitly requested
        if close_symbols and pos.symbol in close_symbols:
            actions.append((pos, pos.quantity, "explicit close"))
            continue

        # Skip if within limits
        if total_value <= 0:
            continue
        pct = (abs(pos.market_value) / total_value) * 100
        if pct <= max_pct:
            continue

        # Calculate how many shares to sell to get to max_pct
        target_value = total_value * (max_pct / 100)
        excess_value = abs(pos.market_value) - target_value
        price = pos.market_price if pos.market_price > 0 else pos.avg_cost
        if price <= 0:
            continue

        sell_qty = math.ceil(excess_value / price)
        sell_qty = min(sell_qty, pos.quantity)  # Never sell more than we have

        if sell_qty > 0:
            actions.append((pos, sell_qty, f"{pct:.1f}% > {max_pct}%"))

    # Sort by excess size (largest first)
    actions.sort(key=lambda x: x[1] * (x[0].market_price or x[0].avg_cost), reverse=True)
    return actions


async def main(execute: bool, max_pct: float, close_symbols: list[str] | None):
    print(f"Connecting to IBKR at {settings.ibkr_host}:{settings.ibkr_port}...")
    broker = IBKRAdapter(
        host=settings.ibkr_host,
        port=settings.ibkr_port,
        client_id=99,  # Separate client ID to not interfere with trading engine
    )
    await broker.connect()

    try:
        portfolio = await broker.get_portfolio()
        summary = portfolio.account_summary
        positions = portfolio.positions

        print(f"\n{'='*65}")
        print(f"  Account Summary")
        print(f"{'='*65}")
        print(f"  Total value:  {summary.total_value:>12,.2f}")
        print(f"  Cash:         {summary.cash:>12,.2f}")
        print(f"  Buying power: {summary.buying_power:>12,.2f}")
        print(f"  Positions:    {len(positions):>12d}")
        print(f"  Max per pos:  {max_pct:>11.1f}%")
        print(f"{'='*65}\n")

        # Show all positions sorted by size
        sorted_positions = sorted(positions, key=lambda p: abs(p.market_value), reverse=True)
        print(f"{'Symbol':<12s} {'Qty':>8s} {'Price':>10s} {'Value':>12s} {'%Port':>7s} {'P&L':>10s}  Status")
        print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*7} {'-'*10}  {'-'*10}")

        for pos in sorted_positions:
            if pos.quantity == 0:
                continue
            pct = (abs(pos.market_value) / summary.total_value * 100) if summary.total_value > 0 else 0
            oversized = pct > max_pct
            forced = close_symbols and pos.symbol in close_symbols
            flag = " << CLOSE" if forced else (" << OVERSIZED" if oversized else "")
            print(
                f"{pos.symbol:<12s} {pos.quantity:>8d} {pos.market_price:>10.2f} "
                f"{pos.market_value:>12,.2f} {pct:>6.1f}% {pos.unrealized_pnl:>+10,.2f}{flag}"
            )

        # Build trim orders
        actions = build_trim_orders(sorted_positions, summary.total_value, max_pct, close_symbols)

        if not actions:
            print("\nNo oversized positions found. Nothing to do.")
            return

        print(f"\n{'='*65}")
        print(f"  Planned sells ({len(actions)} orders)")
        print(f"{'='*65}")
        total_sell_value = 0.0
        for pos, qty, reason in actions:
            price = pos.market_price if pos.market_price > 0 else pos.avg_cost
            value = qty * price
            total_sell_value += value
            remaining = pos.quantity - qty
            print(f"  SELL {qty:>8d} {pos.symbol:<12s}  ~{value:>10,.2f}  ({reason})  remaining: {remaining}")

        print(f"\n  Total sell value: ~{total_sell_value:>,.2f}")

        if not execute:
            print(f"\n  DRY RUN — no orders placed. Use --execute to proceed.")
            return

        # Execute
        print(f"\n  Executing {len(actions)} orders...\n")
        results = []
        for pos, qty, reason in actions:
            order = OrderRequest(
                symbol=pos.symbol,
                side=OrderSide.SELL,
                quantity=qty,
                order_type=OrderType.MARKET,
            )
            try:
                result = await broker.place_order(order)
                status = f"order_id={result.order_id} status={result.status}"
                results.append((pos.symbol, qty, True, status))
                print(f"  SELL {qty:>8d} {pos.symbol:<12s} -> {status}")
            except Exception as e:
                results.append((pos.symbol, qty, False, str(e)))
                print(f"  SELL {qty:>8d} {pos.symbol:<12s} -> ERROR: {e}")

        # Wait for fills
        print("\n  Waiting 10s for fills...")
        await asyncio.sleep(10)

        # Show updated account
        summary = await broker.get_account_summary()
        new_positions = await broker.get_positions()

        print(f"\n{'='*65}")
        print(f"  After execution")
        print(f"{'='*65}")
        print(f"  Total value:  {summary.total_value:>12,.2f}")
        print(f"  Cash:         {summary.cash:>12,.2f}")
        print(f"  Buying power: {summary.buying_power:>12,.2f}")
        print(f"  Positions:    {len(new_positions):>12d}")

        # Summary of results
        success = sum(1 for _, _, ok, _ in results if ok)
        failed = len(results) - success
        print(f"\n  Orders: {success} succeeded, {failed} failed")

        if new_positions:
            print(f"\n{'Symbol':<12s} {'Qty':>8s} {'Value':>12s} {'%Port':>7s}")
            print(f"{'-'*12} {'-'*8} {'-'*12} {'-'*7}")
            for pos in sorted(new_positions, key=lambda p: abs(p.market_value), reverse=True):
                if pos.quantity == 0:
                    continue
                pct = (abs(pos.market_value) / summary.total_value * 100) if summary.total_value > 0 else 0
                flag = " !" if pct > max_pct else ""
                print(f"  {pos.symbol:<12s} {pos.quantity:>8d} {pos.market_value:>12,.2f} {pct:>6.1f}%{flag}")

    finally:
        await broker.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Close oversized positions to bring portfolio within risk limits.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually place sell orders (default is dry-run)",
    )
    parser.add_argument(
        "--max-pct",
        type=float,
        default=settings.max_position_pct,
        help=f"Max position size %% (default: {settings.max_position_pct}%%)",
    )
    parser.add_argument(
        "--close",
        nargs="+",
        metavar="SYMBOL",
        help="Fully close these specific positions",
    )
    args = parser.parse_args()

    asyncio.run(main(
        execute=args.execute,
        max_pct=args.max_pct,
        close_symbols=args.close,
    ))
