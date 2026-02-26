"""VWAP/TWAP smart execution algorithms for larger orders."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum

import structlog

from app.broker.base import BrokerAdapter, OrderResult, OrderType
from app.config import settings
from app.execution.order_manager import OrderManager

logger = structlog.get_logger()


class ExecutionAlgo(StrEnum):
    MARKET = "market"
    VWAP = "vwap"
    TWAP = "twap"


@dataclass
class SliceResult:
    order_result: OrderResult
    slice_qty: int
    slice_num: int


class SmartExecutor:
    """Selects and executes optimal order routing based on order size vs volume."""

    def __init__(self, broker: BrokerAdapter, order_manager: OrderManager):
        self._broker = broker
        self._order_manager = order_manager

    def select_algo(
        self,
        quantity: int,
        avg_volume: float,
        urgency: float = 0.5,
    ) -> ExecutionAlgo:
        """Select execution algorithm based on order size relative to volume.

        Args:
            quantity: Number of shares to trade
            avg_volume: Average daily volume of the symbol
            urgency: 0.0 = patient, 1.0 = urgent (higher urgency favors MARKET)
        """
        if not settings.smart_execution_enabled:
            return ExecutionAlgo.MARKET

        if avg_volume <= 0:
            return ExecutionAlgo.MARKET

        participation_rate = quantity / avg_volume

        # High urgency shifts thresholds
        market_threshold = 0.001 * (1 + urgency)
        twap_threshold = 0.01 * (1 + urgency)

        if participation_rate < market_threshold:
            return ExecutionAlgo.MARKET
        elif participation_rate < twap_threshold:
            return ExecutionAlgo.TWAP
        else:
            return ExecutionAlgo.VWAP

    async def execute_vwap(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        total_qty: int,
        duration_minutes: int | None = None,
    ) -> list[OrderResult]:
        """Execute using VWAP: volume-weighted slices over a duration.

        Splits order into slices with front-loading (more volume early).
        """
        if duration_minutes is None:
            duration_minutes = settings.vwap_duration_minutes

        num_slices = max(3, duration_minutes // 3)
        results = []

        # Front-loaded weights: more shares in early slices
        weights = [1.0 / (i + 1) for i in range(num_slices)]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        remaining = total_qty
        interval = (duration_minutes * 60) / num_slices

        for i, weight in enumerate(weights):
            if remaining <= 0:
                break

            slice_qty = max(1, round(total_qty * weight))
            slice_qty = min(slice_qty, remaining)

            try:
                result = await self._order_manager.submit_order(
                    trade_id=trade_id,
                    symbol=symbol,
                    side=side,
                    quantity=slice_qty,
                    order_type=OrderType.MARKET,
                )
                results.append(result)
                remaining -= slice_qty

                logger.debug(
                    "smart_exec.vwap_slice",
                    symbol=symbol,
                    slice=i + 1,
                    qty=slice_qty,
                    remaining=remaining,
                )
            except Exception:
                logger.exception("smart_exec.vwap_slice_error", slice=i + 1)
                break

            if remaining > 0 and i < len(weights) - 1:
                await asyncio.sleep(interval)

        return results

    async def execute_twap(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        total_qty: int,
        num_slices: int | None = None,
        interval_seconds: int | None = None,
    ) -> list[OrderResult]:
        """Execute using TWAP: equal slices at regular intervals."""
        if num_slices is None:
            num_slices = settings.twap_slices
        if interval_seconds is None:
            interval_seconds = 60  # 1 minute between slices

        results = []
        slice_qty = max(1, total_qty // num_slices)
        remaining = total_qty

        for i in range(num_slices):
            if remaining <= 0:
                break

            # Last slice gets the remainder
            qty = min(slice_qty, remaining) if i < num_slices - 1 else remaining
            qty = min(qty, remaining)

            try:
                result = await self._order_manager.submit_order(
                    trade_id=trade_id,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                )
                results.append(result)
                remaining -= qty

                logger.debug(
                    "smart_exec.twap_slice",
                    symbol=symbol,
                    slice=i + 1,
                    qty=qty,
                    remaining=remaining,
                )
            except Exception:
                logger.exception("smart_exec.twap_slice_error", slice=i + 1)
                break

            if remaining > 0 and i < num_slices - 1:
                await asyncio.sleep(interval_seconds)

        return results
