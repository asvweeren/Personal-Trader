import numpy as np
import pytest

from app.backtest.metrics import (
    _annualized_return,
    _max_consecutive,
    _max_drawdown_with_duration,
    _sharpe_ratio,
    _sortino_ratio,
    calculate_benchmark_comparison,
    calculate_metrics,
)

# ── calculate_metrics tests ──────────────────────────────────


def test_empty_equity_curve():
    result = calculate_metrics([], [], 5000.0)
    assert result == {}


def test_basic_metrics():
    pnls = [100, -50, 200, -30, 150]
    equity = [5000, 5100, 5050, 5250, 5220, 5370]
    metrics = calculate_metrics(pnls, equity, 5000.0)

    assert metrics["total_trades"] == 5
    assert metrics["winning_trades"] == 3
    assert metrics["losing_trades"] == 2
    assert metrics["win_rate"] == 60.0
    assert metrics["total_return_pct"] == pytest.approx(7.4, abs=0.1)
    assert metrics["final_equity"] == 5370.0


def test_all_winning_trades():
    pnls = [100, 200, 150]
    equity = [5000, 5100, 5300, 5450]
    metrics = calculate_metrics(pnls, equity, 5000.0)

    assert metrics["win_rate"] == 100.0
    assert metrics["profit_factor"] == "inf"
    assert metrics["max_consecutive_wins"] == 3
    assert metrics["max_consecutive_losses"] == 0


def test_all_losing_trades():
    pnls = [-100, -200, -150]
    equity = [5000, 4900, 4700, 4550]
    metrics = calculate_metrics(pnls, equity, 5000.0)

    assert metrics["win_rate"] == 0.0
    assert metrics["max_consecutive_wins"] == 0
    assert metrics["max_consecutive_losses"] == 3


def test_no_trades():
    equity = [5000, 5000, 5000]
    metrics = calculate_metrics([], equity, 5000.0)

    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0


def test_profit_factor():
    pnls = [200, -100, 300, -50]
    equity = [5000, 5200, 5100, 5400, 5350]
    metrics = calculate_metrics(pnls, equity, 5000.0)

    # Gross profit = 500, Gross loss = 150
    assert metrics["profit_factor"] == pytest.approx(3.333, abs=0.01)


def test_expectancy():
    pnls = [100, -50, 100, -50]
    equity = [5000, 5100, 5050, 5150, 5100]
    metrics = calculate_metrics(pnls, equity, 5000.0)

    # win_rate = 0.5, avg_win = 100, avg_loss = -50
    # expectancy = 0.5 * 100 + 0.5 * (-50) = 25
    assert metrics["expectancy"] == 25.0


# ── Max drawdown tests ───────────────────────────────────────


def test_max_drawdown_simple():
    equity = np.array([100, 110, 105, 115, 100, 120])
    dd, duration = _max_drawdown_with_duration(equity)
    # Peak 115, trough 100 -> 13.04%
    assert dd == pytest.approx(0.1304, abs=0.01)


def test_max_drawdown_no_drawdown():
    equity = np.array([100, 110, 120, 130])
    dd, duration = _max_drawdown_with_duration(equity)
    assert dd == 0.0


def test_max_drawdown_duration():
    equity = np.array([100, 110, 105, 95, 100, 105, 115])
    dd, duration = _max_drawdown_with_duration(equity)
    assert duration > 0


def test_max_drawdown_single_value():
    dd, duration = _max_drawdown_with_duration(np.array([100]))
    assert dd == 0.0
    assert duration == 0


# ── Consecutive wins/losses tests ────────────────────────────


def test_consecutive_mixed():
    pnls = [10, 20, -5, -10, -15, 30]
    wins, losses = _max_consecutive(pnls)
    assert wins == 2
    assert losses == 3


def test_consecutive_empty():
    wins, losses = _max_consecutive([])
    assert wins == 0
    assert losses == 0


def test_consecutive_all_zeros():
    wins, losses = _max_consecutive([0, 0, 0])
    assert wins == 0
    assert losses == 0


# ── Sharpe ratio tests ───────────────────────────────────────


def test_sharpe_positive_returns():
    returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
    sharpe = _sharpe_ratio(returns, 0.02, 252)
    assert sharpe > 0


def test_sharpe_flat_returns():
    returns = np.array([0.0, 0.0, 0.0, 0.0])
    sharpe = _sharpe_ratio(returns, 0.02, 252)
    assert sharpe <= 0


def test_sharpe_insufficient_data():
    sharpe = _sharpe_ratio(np.array([0.01]), 0.02, 252)
    assert sharpe == 0.0


# ── Sortino ratio tests ─────────────────────────────────────


def test_sortino_no_downside():
    returns = np.array([0.01, 0.02, 0.015])
    sortino = _sortino_ratio(returns, 0.02, 252)
    assert sortino == 0.0  # No negative returns


def test_sortino_with_downside():
    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    sortino = _sortino_ratio(returns, 0.02, 252)
    assert sortino != 0.0


# ── Annualized return tests ──────────────────────────────────


def test_annualized_return_positive():
    equity = np.array([1000, 1100])  # 10% in 1 period
    ann = _annualized_return(equity, 252)
    assert ann > 0


def test_annualized_return_zero_start():
    equity = np.array([0, 100])
    ann = _annualized_return(equity, 252)
    assert ann == 0.0


# ── Benchmark comparison tests ───────────────────────────────


def test_benchmark_comparison_basic():
    equity = [5000, 5100, 5200, 5300, 5400]
    prices = np.array([100, 102, 104, 106, 108])
    result = calculate_benchmark_comparison(equity, prices, 5000.0)

    assert "strategy_return_pct" in result
    assert "benchmark_return_pct" in result
    assert "excess_return_pct" in result
    assert "alpha" in result
    assert "beta" in result


def test_benchmark_comparison_outperformance():
    # Strategy doubles, benchmark flat
    equity = [5000, 6000, 7000, 8000, 9000, 10000]
    prices = np.array([100, 100, 100, 100, 100])
    result = calculate_benchmark_comparison(equity, prices, 5000.0)

    assert result["strategy_return_pct"] > result["benchmark_return_pct"]
    assert result["excess_return_pct"] > 0


def test_benchmark_comparison_insufficient_data():
    result = calculate_benchmark_comparison([5000], np.array([100]), 5000.0)
    assert result == {}
