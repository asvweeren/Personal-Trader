"""Tests for VaR calculator."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
import pandas as pd

from app.broker.base import Portfolio, AccountSummary, Position
from app.risk.var_calculator import VaRCalculator, VaRResult, STRESS_SCENARIOS


def _make_portfolio(positions: list[tuple[str, int, float]]) -> Portfolio:
    """Create portfolio with positions: (symbol, qty, price)."""
    total_value = 10000.0
    pos_list = []
    for sym, qty, price in positions:
        pos_list.append(Position(
            symbol=sym,
            quantity=qty,
            avg_cost=price,
            market_price=price,
            market_value=price * qty,
            unrealized_pnl=0.0,
        ))
    cash = total_value - sum(p.market_value for p in pos_list)
    return Portfolio(
        account_summary=AccountSummary(
            total_value=total_value,
            cash=cash,
            buying_power=cash,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        ),
        positions=pos_list,
    )


class TestVaRResult:
    def test_to_dict(self):
        result = VaRResult(
            var_95=150.0,
            var_99=250.0,
            cvar_95=200.0,
            stress_scenarios={"Crash": -1000.0},
        )
        d = result.to_dict()
        assert d["var_95"] == 150.0
        assert d["var_99"] == 250.0
        assert d["cvar_95"] == 200.0
        assert d["stress_scenarios"]["Crash"] == -1000.0

    def test_default_values(self):
        result = VaRResult()
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0
        assert result.cvar_95 == 0.0
        assert result.stress_scenarios == {}


class TestVaRCalculator:
    @pytest.mark.asyncio
    async def test_empty_portfolio(self):
        calc = VaRCalculator()
        portfolio = Portfolio(
            account_summary=AccountSummary(
                total_value=10000.0, cash=10000.0, buying_power=10000.0,
                unrealized_pnl=0.0, realized_pnl=0.0,
            ),
            positions=[],
        )
        market_data = AsyncMock()
        result = await calc.calculate_portfolio_var(portfolio, market_data)
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0

    @pytest.mark.asyncio
    async def test_zero_total_value(self):
        calc = VaRCalculator()
        portfolio = Portfolio(
            account_summary=AccountSummary(
                total_value=0.0, cash=0.0, buying_power=0.0,
                unrealized_pnl=0.0, realized_pnl=0.0,
            ),
            positions=[],
        )
        market_data = AsyncMock()
        result = await calc.calculate_portfolio_var(portfolio, market_data)
        assert result.var_95 == 0.0

    @pytest.mark.asyncio
    async def test_with_historical_data(self):
        calc = VaRCalculator()
        portfolio = _make_portfolio([("AAPL", 50, 150.0)])

        # Create mock historical data with some returns
        np.random.seed(42)
        n = 60
        closes = 150.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
        df = pd.DataFrame({
            "close": closes,
            "open": closes * 1.001,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "volume": np.random.randint(1000000, 5000000, n),
        })

        market_data = AsyncMock()
        market_data.get_historical_data = AsyncMock(return_value=df)

        result = await calc.calculate_portfolio_var(portfolio, market_data)
        assert result.var_95 >= 0
        assert result.var_99 >= result.var_95  # 99% VaR should be >= 95% VaR
        assert result.cvar_95 >= result.var_95  # CVaR >= VaR

    @pytest.mark.asyncio
    async def test_stress_scenarios(self):
        calc = VaRCalculator()
        portfolio = _make_portfolio([("AAPL", 50, 150.0)])

        market_data = AsyncMock()
        market_data.get_historical_data = AsyncMock(return_value=None)

        result = await calc.calculate_portfolio_var(portfolio, market_data)
        # Should have stress scenarios from fallback
        assert len(result.stress_scenarios) == len(STRESS_SCENARIOS)
        for name in STRESS_SCENARIOS:
            assert name in result.stress_scenarios
            assert result.stress_scenarios[name] < 0  # All are negative impacts

    @pytest.mark.asyncio
    async def test_estimate_var_fallback(self):
        calc = VaRCalculator()
        portfolio = _make_portfolio([("AAPL", 50, 150.0)])
        result = calc._estimate_var(10000.0, portfolio)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 > result.var_95

    @pytest.mark.asyncio
    async def test_data_fetch_error_uses_fallback(self):
        calc = VaRCalculator()
        portfolio = _make_portfolio([("AAPL", 50, 150.0)])

        market_data = AsyncMock()
        market_data.get_historical_data = AsyncMock(side_effect=Exception("Connection error"))

        result = await calc.calculate_portfolio_var(portfolio, market_data)
        # Should still return a result (fallback)
        assert result.var_95 > 0
