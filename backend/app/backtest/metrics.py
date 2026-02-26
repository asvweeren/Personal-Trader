"""Performance metrics calculation for backtests and live trading."""

import numpy as np
from scipy import stats as scipy_stats


def calculate_metrics(
    trade_pnls: list[float],
    equity_curve: list[float],
    initial_capital: float,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        trade_pnls: List of P&L per closed trade.
        equity_curve: List of equity values over time.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading periods per year (252 for daily, ~1764 for hourly).
    """
    if not equity_curve:
        return {}

    equity = np.array(equity_curve, dtype=float)
    total_trades = len(trade_pnls)
    winning = [p for p in trade_pnls if p > 0]
    losing = [p for p in trade_pnls if p < 0]

    total_return = (equity[-1] - initial_capital) / initial_capital if initial_capital > 0 else 0
    win_rate = len(winning) / total_trades if total_trades > 0 else 0

    # Returns series
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])

    # Sharpe ratio (annualized)
    sharpe = _sharpe_ratio(returns, risk_free_rate, periods_per_year)

    # Sortino ratio
    sortino = _sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Calmar ratio (annualized return / max drawdown)
    max_dd, max_dd_duration = _max_drawdown_with_duration(equity)
    annualized_return = _annualized_return(equity, periods_per_year)
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    # Profit factor
    gross_profit = sum(winning) if winning else 0
    gross_loss = abs(sum(losing)) if losing else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade stats
    avg_win = float(np.mean(winning)) if winning else 0
    avg_loss = float(np.mean(losing)) if losing else 0

    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Consecutive wins/losses
    max_consec_wins, max_consec_losses = _max_consecutive(trade_pnls)

    # Trade duration stats (if bars_held info is available, caller handles that)
    avg_trade_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0

    # Recovery factor: net profit / max drawdown
    net_profit = float(equity[-1] - initial_capital)
    if max_dd > 0 and initial_capital > 0:
        recovery_factor = net_profit / (max_dd * initial_capital)
    else:
        recovery_factor = 0.0

    # Return distribution stats
    return_skew = float(scipy_stats.skew(returns)) if len(returns) > 2 else 0.0
    return_kurtosis = float(scipy_stats.kurtosis(returns)) if len(returns) > 2 else 0.0

    # Drawdown recovery time (bars from trough back to peak)
    dd_recovery_bars = _drawdown_recovery_bars(equity)

    result = {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round(win_rate * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration": max_dd_duration,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "expectancy": round(expectancy, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "final_equity": round(float(equity[-1]), 2),
        "recovery_factor": round(recovery_factor, 3),
        "return_skew": round(return_skew, 3),
        "return_kurtosis": round(return_kurtosis, 3),
        "drawdown_recovery_bars": dd_recovery_bars,
    }

    return result


def calculate_benchmark_comparison(
    equity_curve: list[float],
    price_series: np.ndarray,
    initial_capital: float,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict:
    """Compare strategy performance against buy-and-hold benchmark.

    Args:
        equity_curve: Strategy equity values.
        price_series: Asset price series (same length as equity minus initial).
        initial_capital: Starting capital.
    """
    if len(equity_curve) < 2 or len(price_series) < 2:
        return {}

    # Buy-and-hold: invest all capital at first price
    first_price = float(price_series[0])
    shares = initial_capital / first_price if first_price > 0 else 0
    benchmark_equity = shares * np.array(price_series, dtype=float)

    benchmark_return = (
        (benchmark_equity[-1] - initial_capital) / initial_capital
        if initial_capital > 0 else 0
    )
    strategy_equity = np.array(equity_curve, dtype=float)
    strategy_return = (
        (strategy_equity[-1] - initial_capital) / initial_capital
        if initial_capital > 0 else 0
    )

    # Strategy returns (align lengths)
    strategy_vals = strategy_equity[1:]  # Skip initial capital
    min_len = min(len(strategy_vals), len(benchmark_equity))
    strategy_vals = strategy_vals[:min_len]
    benchmark_vals = benchmark_equity[:min_len]

    # Returns
    strat_returns = (
        np.diff(strategy_vals) / strategy_vals[:-1]
        if len(strategy_vals) > 1 else np.array([])
    )
    bench_returns = (
        np.diff(benchmark_vals) / benchmark_vals[:-1]
        if len(benchmark_vals) > 1 else np.array([])
    )

    # Alpha and Beta (CAPM)
    alpha, beta = 0.0, 1.0
    if len(strat_returns) > 1 and len(bench_returns) > 1:
        min_r = min(len(strat_returns), len(bench_returns))
        sr = strat_returns[:min_r]
        br = bench_returns[:min_r]
        if np.std(br) > 0:
            beta = float(np.cov(sr, br)[0, 1] / np.var(br))
            alpha = float(np.mean(sr) - beta * np.mean(br)) * periods_per_year

    # Information ratio
    info_ratio = 0.0
    if len(strat_returns) > 1 and len(bench_returns) > 1:
        min_r = min(len(strat_returns), len(bench_returns))
        tracking_error = np.std(strat_returns[:min_r] - bench_returns[:min_r])
        if tracking_error > 0:
            excess = strat_returns[:min_r] - bench_returns[:min_r]
            info_ratio = (
                float(np.mean(excess))
                / tracking_error
                * np.sqrt(periods_per_year)
            )

    # Benchmark max drawdown
    bm_dd, _ = _max_drawdown_with_duration(benchmark_vals)

    return {
        "strategy_return_pct": round(strategy_return * 100, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2),
        "excess_return_pct": round((strategy_return - benchmark_return) * 100, 2),
        "alpha": round(alpha * 100, 3),
        "beta": round(beta, 3),
        "information_ratio": round(float(info_ratio), 3),
        "benchmark_max_drawdown_pct": round(bm_dd * 100, 2),
        "benchmark_final_equity": round(float(benchmark_equity[-1]), 2),
    }


def calculate_monthly_returns(
    equity_curve: list[float],
    timestamps: list,
) -> dict[str, float]:
    """Calculate monthly returns from equity curve with timestamps."""
    if len(equity_curve) < 2 or len(timestamps) < 2:
        return {}

    import pandas as pd

    df = pd.DataFrame({"equity": equity_curve, "time": pd.to_datetime(timestamps)})
    df["month"] = df["time"].dt.to_period("M")
    monthly = df.groupby("month")["equity"].agg(["first", "last"])
    monthly["return_pct"] = ((monthly["last"] - monthly["first"]) / monthly["first"] * 100).round(2)

    return {str(k): v for k, v in monthly["return_pct"].items()}


# ── Internal helpers ─────────────────────────────────────────


def _sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float, periods_per_year: int
) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def _sortino_ratio(
    returns: np.ndarray, risk_free_rate: float, periods_per_year: int
) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess_mean = float(np.mean(returns)) - rf_per_period
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return 0.0
    return float(excess_mean / np.std(downside, ddof=1) * np.sqrt(periods_per_year))


def _annualized_return(equity: np.ndarray, periods_per_year: int) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    total_return = equity[-1] / equity[0]
    n_periods = len(equity) - 1
    if n_periods == 0:
        return 0.0
    return float(total_return ** (periods_per_year / n_periods) - 1)


def _max_drawdown_with_duration(equity: np.ndarray) -> tuple[float, int]:
    """Calculate maximum drawdown and its duration in periods."""
    if len(equity) < 2:
        return 0.0, 0

    peak = equity[0]
    max_dd = 0.0
    _dd_start = 0
    max_dd_duration = 0
    current_dd_start = 0
    in_drawdown = False

    for i, value in enumerate(equity):
        if value > peak:
            if in_drawdown:
                duration = i - current_dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                in_drawdown = False
            peak = value
        else:
            dd = (peak - value) / peak
            if dd > 0 and not in_drawdown:
                current_dd_start = i
                in_drawdown = True
            if dd > max_dd:
                max_dd = dd

    # If still in drawdown at end
    if in_drawdown:
        duration = len(equity) - 1 - current_dd_start
        if duration > max_dd_duration:
            max_dd_duration = duration

    return float(max_dd), max_dd_duration


def _drawdown_recovery_bars(equity: np.ndarray) -> int:
    """Bars from maximum drawdown trough back to previous peak."""
    if len(equity) < 2:
        return 0
    peak = equity[0]
    max_dd = 0.0
    trough_idx = 0
    peak_at_trough = peak
    for i, value in enumerate(equity):
        if value > peak:
            peak = value
        else:
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                trough_idx = i
                peak_at_trough = peak
    # Count bars from trough to recovery (back to peak level)
    for i in range(trough_idx, len(equity)):
        if equity[i] >= peak_at_trough:
            return i - trough_idx
    # Never recovered
    return len(equity) - 1 - trough_idx


def _max_consecutive(trade_pnls: list[float]) -> tuple[int, int]:
    """Calculate maximum consecutive wins and losses."""
    if not trade_pnls:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in trade_pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses
