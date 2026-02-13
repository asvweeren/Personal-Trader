"""Tests for VWAP/TWAP smart execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.execution.smart_execution import SmartExecutor, ExecutionAlgo
from app.broker.base import OrderResult


class TestExecutionAlgo:
    def test_enum_values(self):
        assert ExecutionAlgo.MARKET == "market"
        assert ExecutionAlgo.VWAP == "vwap"
        assert ExecutionAlgo.TWAP == "twap"


class TestSmartExecutor:
    def _make_executor(self):
        broker = MagicMock()
        order_manager = MagicMock()
        return SmartExecutor(broker, order_manager)

    def test_select_algo_small_order(self):
        executor = self._make_executor()
        algo = executor.select_algo(quantity=10, avg_volume=1_000_000)
        assert algo == ExecutionAlgo.MARKET

    def test_select_algo_medium_order(self):
        executor = self._make_executor()
        # 0.5% participation → TWAP
        algo = executor.select_algo(quantity=5_000, avg_volume=1_000_000)
        assert algo == ExecutionAlgo.TWAP

    def test_select_algo_large_order(self):
        executor = self._make_executor()
        # 2% participation → VWAP
        algo = executor.select_algo(quantity=20_000, avg_volume=1_000_000)
        assert algo == ExecutionAlgo.VWAP

    def test_select_algo_zero_volume(self):
        executor = self._make_executor()
        algo = executor.select_algo(quantity=100, avg_volume=0)
        assert algo == ExecutionAlgo.MARKET

    def test_select_algo_high_urgency(self):
        executor = self._make_executor()
        # Higher urgency shifts thresholds up, so TWAP-range order becomes MARKET
        algo = executor.select_algo(quantity=1_500, avg_volume=1_000_000, urgency=1.0)
        assert algo == ExecutionAlgo.MARKET

    @patch("app.execution.smart_execution.settings")
    def test_select_algo_disabled(self, mock_settings):
        mock_settings.smart_execution_enabled = False
        executor = self._make_executor()
        algo = executor.select_algo(quantity=50_000, avg_volume=1_000_000)
        assert algo == ExecutionAlgo.MARKET

    @pytest.mark.asyncio
    async def test_execute_twap(self):
        broker = MagicMock()
        order_manager = AsyncMock()
        order_manager.submit_order = AsyncMock(return_value=OrderResult(
            order_id="1", status="filled", filled_price=100.0, filled_quantity=25,
        ))

        executor = SmartExecutor(broker, order_manager)
        results = await executor.execute_twap(
            trade_id=1,
            symbol="AAPL",
            side="BUY",
            total_qty=100,
            num_slices=4,
            interval_seconds=0,  # No delay for test
        )
        assert len(results) == 4
        assert order_manager.submit_order.call_count == 4

    @pytest.mark.asyncio
    @patch("app.execution.smart_execution.asyncio.sleep", new_callable=AsyncMock)
    async def test_execute_vwap(self, mock_sleep):
        broker = MagicMock()
        order_manager = AsyncMock()
        order_manager.submit_order = AsyncMock(return_value=OrderResult(
            order_id="1", status="filled", filled_price=100.0, filled_quantity=10,
        ))

        executor = SmartExecutor(broker, order_manager)
        results = await executor.execute_vwap(
            trade_id=1,
            symbol="AAPL",
            side="BUY",
            total_qty=100,
            duration_minutes=9,  # Will create 3 slices (9//3)
        )
        assert len(results) >= 1
        assert order_manager.submit_order.call_count >= 1

    @pytest.mark.asyncio
    async def test_execute_twap_handles_error(self):
        broker = MagicMock()
        order_manager = AsyncMock()
        order_manager.submit_order = AsyncMock(side_effect=Exception("Network error"))

        executor = SmartExecutor(broker, order_manager)
        results = await executor.execute_twap(
            trade_id=1,
            symbol="AAPL",
            side="BUY",
            total_qty=100,
            num_slices=4,
            interval_seconds=0,
        )
        # Should return empty list since all slices failed
        assert len(results) == 0
