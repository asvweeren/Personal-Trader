"""Tests for Monte Carlo simulation."""

import pytest

from app.backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig, MonteCarloResult


class TestMonteCarloResult:
    def test_to_dict(self):
        result = MonteCarloResult(
            percentiles={0.05: 4000.0, 0.5: 5500.0, 0.95: 7000.0},
            max_drawdown_dist={0.05: 2.0, 0.5: 10.0, 0.95: 25.0},
            ruin_probability=0.03,
            median_return_pct=10.0,
            equity_bands=[{"trade_num": 1, "p5": 4500, "p50": 5000, "p95": 5500}],
        )
        d = result.to_dict()
        assert d["ruin_probability"] == 0.03
        assert d["median_return_pct"] == 10.0
        assert "0.5" in d["percentiles"]

    def test_default_result(self):
        result = MonteCarloResult()
        assert result.percentiles == {}
        assert result.ruin_probability == 0.0
        assert result.equity_bands == []


class TestMonteCarloSimulator:
    def test_empty_pnls(self):
        sim = MonteCarloSimulator()
        result = sim.run([], 5000.0)
        assert result.percentiles == {}
        assert result.ruin_probability == 0.0

    def test_basic_simulation(self):
        sim = MonteCarloSimulator()
        # 50 profitable trades
        pnls = [10.0] * 30 + [-5.0] * 20
        result = sim.run(pnls, 5000.0)

        assert len(result.percentiles) == 5  # Default confidence levels
        assert result.median_return_pct > 0
        assert 0 <= result.ruin_probability <= 1
        assert len(result.equity_bands) > 0

    def test_all_winning_trades(self):
        sim = MonteCarloSimulator()
        pnls = [50.0] * 20
        result = sim.run(pnls, 5000.0)

        assert result.median_return_pct > 0
        assert result.ruin_probability == 0.0
        # All percentiles should be above initial capital
        for level, value in result.percentiles.items():
            assert value > 5000.0

    def test_all_losing_trades(self):
        sim = MonteCarloSimulator()
        pnls = [-100.0] * 30
        result = sim.run(pnls, 5000.0)

        assert result.median_return_pct < 0
        assert result.ruin_probability > 0

    def test_custom_config(self):
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(
            num_simulations=100,
            confidence_levels=[0.1, 0.5, 0.9],
        )
        pnls = [10.0, -5.0, 15.0, -3.0, 8.0] * 10
        result = sim.run(pnls, 5000.0, config=config)

        assert len(result.percentiles) == 3
        assert 0.1 in result.percentiles
        assert 0.5 in result.percentiles
        assert 0.9 in result.percentiles

    def test_deterministic_with_seed(self):
        """Monte Carlo uses seed=42 so should be deterministic."""
        sim = MonteCarloSimulator()
        pnls = [10.0, -5.0, 20.0, -8.0, 15.0] * 4
        result1 = sim.run(pnls, 5000.0)
        result2 = sim.run(pnls, 5000.0)

        assert result1.median_return_pct == result2.median_return_pct
        assert result1.ruin_probability == result2.ruin_probability

    def test_equity_bands_structure(self):
        sim = MonteCarloSimulator()
        pnls = [10.0, -5.0, 15.0, -3.0] * 20
        result = sim.run(pnls, 5000.0)

        for band in result.equity_bands:
            assert "trade_num" in band
            assert "p5" in band
            assert "p25" in band
            assert "p50" in band
            assert "p75" in band
            assert "p95" in band
            # P5 <= P25 <= P50 <= P75 <= P95
            assert band["p5"] <= band["p25"]
            assert band["p25"] <= band["p50"]
            assert band["p50"] <= band["p75"]
            assert band["p75"] <= band["p95"]

    def test_drawdown_distribution(self):
        sim = MonteCarloSimulator()
        pnls = [10.0, -20.0, 15.0, -10.0] * 20
        result = sim.run(pnls, 5000.0)

        for level, dd in result.max_drawdown_dist.items():
            assert dd >= 0  # Drawdowns are positive percentages

    def test_single_trade(self):
        sim = MonteCarloSimulator()
        pnls = [100.0]
        result = sim.run(pnls, 5000.0)
        # Should work even with single trade
        assert result.median_return_pct > 0
        assert len(result.equity_bands) >= 1
