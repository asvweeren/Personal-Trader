from app.broker.base import AccountSummary, Portfolio, Position
from app.risk.position_sizer import (
    calculate_position_size,
    calculate_kelly_fraction,
    calculate_correlation_factor,
    calculate_sector_factor,
    calculate_trailing_stop,
    get_sector,
)


def make_portfolio(total_value=5000, cash=3000, positions=None):
    return Portfolio(
        account_summary=AccountSummary(
            total_value=total_value,
            cash=cash,
            buying_power=cash,
            unrealized_pnl=0,
            realized_pnl=0,
        ),
        positions=positions or [],
    )


# ── Basic position sizing ─────────────────────────────────────


def test_basic_position_size():
    portfolio = make_portfolio(total_value=5000, cash=3000)
    qty = calculate_position_size(portfolio, price=100, max_position_pct=20, confidence=0.8)
    assert qty > 0
    assert qty * 100 <= 5000 * 0.20


def test_low_confidence_smaller_position():
    portfolio = make_portfolio(total_value=5000, cash=3000)
    qty_high = calculate_position_size(portfolio, price=100, max_position_pct=20, confidence=0.9)
    qty_low = calculate_position_size(portfolio, price=100, max_position_pct=20, confidence=0.3)
    assert qty_high > qty_low


def test_zero_price():
    portfolio = make_portfolio()
    qty = calculate_position_size(portfolio, price=0, max_position_pct=20, confidence=0.8)
    assert qty == 0


def test_no_cash():
    portfolio = make_portfolio(total_value=5000, cash=0)
    qty = calculate_position_size(portfolio, price=100, max_position_pct=20, confidence=0.8)
    assert qty == 0


def test_volatility_reduces_size():
    portfolio = make_portfolio(total_value=5000, cash=3000)
    qty_low_vol = calculate_position_size(
        portfolio, price=100, max_position_pct=20, confidence=0.8, volatility=0.1
    )
    qty_high_vol = calculate_position_size(
        portfolio, price=100, max_position_pct=20, confidence=0.8, volatility=0.6
    )
    assert qty_low_vol >= qty_high_vol


# ── Kelly criterion ───────────────────────────────────────────


def test_kelly_positive_edge():
    # Win rate 60%, 1.5:1 win/loss → positive Kelly
    k = calculate_kelly_fraction(win_rate=0.60, avg_win=1.5, avg_loss=1.0)
    assert k > 0


def test_kelly_no_edge():
    # Win rate 40%, 1:1 ratio → negative Kelly → clamped to 0
    k = calculate_kelly_fraction(win_rate=0.40, avg_win=1.0, avg_loss=1.0)
    assert k == 0.0


def test_kelly_capped_at_quarter():
    # Even with huge edge, half-Kelly capped at 0.25
    k = calculate_kelly_fraction(win_rate=0.90, avg_win=5.0, avg_loss=1.0)
    assert k <= 0.25


def test_kelly_with_zero_loss():
    k = calculate_kelly_fraction(win_rate=0.55, avg_win=1.5, avg_loss=0.0)
    assert k == 0.0


def test_position_size_with_kelly():
    portfolio = make_portfolio(total_value=10000, cash=7000)
    qty = calculate_position_size(
        portfolio, price=100, max_position_pct=20, confidence=0.8,
        win_rate=0.6, avg_win_loss_ratio=1.5,
    )
    assert qty > 0


# ── Correlation factor ────────────────────────────────────────


def test_correlation_no_positions():
    factor = calculate_correlation_factor("AAPL", [])
    assert factor == 1.0


def test_correlation_same_sector_reduces():
    # AAPL and MSFT are both technology
    factor = calculate_correlation_factor("AAPL", ["MSFT"])
    assert factor < 1.0


def test_correlation_different_sector_no_reduction():
    # AAPL (tech) vs JPM (finance)
    factor = calculate_correlation_factor("AAPL", ["JPM"])
    assert factor == 1.0


def test_correlation_multiple_same_sector():
    # More positions in same sector → stronger reduction
    factor_one = calculate_correlation_factor("AAPL", ["MSFT"])
    factor_two = calculate_correlation_factor("AAPL", ["MSFT", "GOOGL"])
    assert factor_two < factor_one


def test_correlation_with_matrix():
    matrix = {("AAPL", "MSFT"): 0.85}
    factor = calculate_correlation_factor("AAPL", ["MSFT"], matrix)
    assert factor < 1.0
    # High correlation → significant reduction
    assert factor < 0.5


def test_correlation_minimum_factor():
    # Even with many correlated positions, factor doesn't go below 0.3
    factor = calculate_correlation_factor("AAPL", ["MSFT", "GOOGL", "AMZN", "META", "NVDA"])
    assert factor >= 0.3


# ── Sector factor ─────────────────────────────────────────────


def test_sector_no_concentration():
    portfolio = make_portfolio(total_value=5000)
    factor = calculate_sector_factor("AAPL", portfolio)
    assert factor == 1.0  # No positions, no concentration


def test_sector_high_concentration_reduces():
    positions = [
        Position("MSFT", 10, 380, 380, 1500, 0),  # tech
        Position("GOOGL", 5, 150, 150, 750, 0),    # tech
    ]
    portfolio = make_portfolio(total_value=5000, positions=positions)
    # Tech at 2250/5000 = 45% > 40% limit
    factor = calculate_sector_factor("AAPL", portfolio)
    assert factor == 0.0  # Sector is full


def test_sector_approaching_limit_reduces():
    positions = [
        Position("MSFT", 5, 300, 300, 1500, 0),  # tech, 30% of 5000
    ]
    portfolio = make_portfolio(total_value=5000, positions=positions)
    # Tech at 30%, limit 40% → approaching but not there
    factor = calculate_sector_factor("AAPL", portfolio)
    assert 0.0 < factor <= 1.0


def test_sector_different_sector_full_factor():
    positions = [
        Position("JPM", 20, 150, 150, 1500, 0),  # finance
    ]
    portfolio = make_portfolio(total_value=5000, positions=positions)
    # Adding tech, finance is concentrated but not tech
    factor = calculate_sector_factor("AAPL", portfolio)
    assert factor == 1.0


# ── Trailing stop ─────────────────────────────────────────────


def test_trailing_stop_fixed_pct():
    stop = calculate_trailing_stop(entry_price=100, current_price=110, trail_pct=3.0)
    assert stop == round(110 * 0.97, 2)


def test_trailing_stop_atr_based():
    stop = calculate_trailing_stop(entry_price=100, current_price=110, atr=2.5, trail_pct=3.0)
    # ATR stop: 110 - 2*2.5 = 105
    # Pct stop: 110 * 0.97 = 106.7
    # max(105, 106.7) = 106.7
    assert stop >= 105


def test_trailing_stop_atr_takes_precedence_when_wider():
    stop = calculate_trailing_stop(entry_price=100, current_price=110, atr=5.0, trail_pct=3.0)
    # ATR stop: 110 - 2*5 = 100
    # Pct stop: 110 * 0.97 = 106.7
    # max(100, 106.7) = 106.7 (pct is the floor)
    pct_stop = 110 * 0.97
    assert stop >= pct_stop


# ── get_sector ────────────────────────────────────────────────


def test_get_sector_known():
    assert get_sector("AAPL") == "technology"
    assert get_sector("JPM") == "finance"
    assert get_sector("JNJ") == "healthcare"


def test_get_sector_unknown():
    assert get_sector("RANDOM_TICKER") == "unknown"
