"""Market simulator for backtesting with realistic execution modeling."""

from dataclasses import dataclass
from enum import Enum



class FillStatus(str, Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


@dataclass
class FillResult:
    status: FillStatus
    fill_price: float
    filled_quantity: int
    commission: float
    slippage: float
    message: str = ""


@dataclass
class SimulatedPosition:
    """Tracks a single position in the simulation."""

    symbol: str
    quantity: int
    avg_cost: float
    stop_loss: float | None = None
    take_profit: float | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


class MarketSimulator:
    """Simulates market execution with realistic slippage, spread, and commissions.

    Slippage model:
        - Base slippage from configured percentage
        - Volume-based impact: larger orders relative to bar volume get more slippage
        - Spread: half the spread is added to buy / subtracted from sell

    Commission model:
        - Percentage-based (default 0.1% = 10 bps, typical for IBKR)
        - Minimum commission per order
    """

    def __init__(
        self,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
        spread_pct: float = 0.02,
        min_commission: float = 1.0,
        market_impact_factor: float = 0.1,
    ):
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct
        self.min_commission = min_commission
        self.market_impact_factor = market_impact_factor

    def simulate_fill(
        self,
        price: float,
        side: str,
        quantity: int,
        bar_volume: int | None = None,
    ) -> FillResult:
        """Simulate order fill with slippage, spread, and commission.

        Args:
            price: Reference price (typically the close of the bar).
            side: "BUY" or "SELL".
            quantity: Number of shares.
            bar_volume: Volume of the current bar (for market impact calculation).

        Returns:
            FillResult with fill details.
        """
        if quantity <= 0:
            return FillResult(
                status=FillStatus.REJECTED,
                fill_price=0.0,
                filled_quantity=0,
                commission=0.0,
                slippage=0.0,
                message="Invalid quantity",
            )

        # Base slippage
        base_slippage = price * (self.slippage_pct / 100)

        # Market impact: larger orders move the price more
        impact = 0.0
        if bar_volume and bar_volume > 0:
            volume_ratio = quantity / bar_volume
            impact = price * volume_ratio * self.market_impact_factor

        # Spread cost (half spread per side)
        half_spread = price * (self.spread_pct / 100) / 2

        total_slippage = base_slippage + impact + half_spread

        if side == "BUY":
            fill_price = price + total_slippage
        else:
            fill_price = price - total_slippage

        fill_price = round(fill_price, 4)

        # Commission
        commission = max(
            fill_price * quantity * (self.commission_pct / 100),
            self.min_commission,
        )
        commission = round(commission, 2)

        return FillResult(
            status=FillStatus.FILLED,
            fill_price=fill_price,
            filled_quantity=quantity,
            commission=commission,
            slippage=round(total_slippage, 4),
        )

    def simulate_stop_check(
        self,
        stop_price: float,
        bar_low: float,
        bar_high: float,
        side: str = "SELL",
    ) -> bool:
        """Check if a stop order would have been triggered during a bar.

        For SELL stops (long position): triggered if bar_low <= stop_price.
        For BUY stops (short position): triggered if bar_high >= stop_price.
        """
        if side == "SELL":
            return bar_low <= stop_price
        return bar_high >= stop_price

    def simulate_stop_fill(
        self,
        stop_price: float,
        bar_open: float,
        bar_low: float,
        quantity: int,
    ) -> FillResult:
        """Simulate a stop-loss fill. Price may gap through the stop level."""
        # If bar opened below stop (gap down), fill at open price (worse)
        fill_ref = min(stop_price, bar_open) if bar_open < stop_price else stop_price

        return self.simulate_fill(fill_ref, "SELL", quantity)

    def simulate_take_profit_check(
        self,
        take_profit_price: float,
        bar_high: float,
    ) -> bool:
        """Check if take-profit would have been triggered during a bar."""
        return bar_high >= take_profit_price

    def simulate_take_profit_fill(
        self,
        take_profit_price: float,
        bar_open: float,
        bar_high: float,
        quantity: int,
    ) -> FillResult:
        """Simulate a take-profit fill. Price may gap through the target."""
        # If bar opened above target (gap up), fill at open price (better)
        fill_ref = (
            max(take_profit_price, bar_open)
            if bar_open > take_profit_price
            else take_profit_price
        )

        return self.simulate_fill(fill_ref, "SELL", quantity)
