import pytest

from app.backtest.simulator import FillStatus, MarketSimulator, SimulatedPosition


# ── Basic fill tests ─────────────────────────────────────────


def test_buy_fill_adds_slippage():
    sim = MarketSimulator(commission_pct=0.0, slippage_pct=0.1, spread_pct=0.0)
    result = sim.simulate_fill(100.0, "BUY", 10)
    assert result.status == FillStatus.FILLED
    assert result.fill_price > 100.0
    assert result.filled_quantity == 10


def test_sell_fill_subtracts_slippage():
    sim = MarketSimulator(commission_pct=0.0, slippage_pct=0.1, spread_pct=0.0)
    result = sim.simulate_fill(100.0, "SELL", 10)
    assert result.status == FillStatus.FILLED
    assert result.fill_price < 100.0


def test_zero_quantity_rejected():
    sim = MarketSimulator()
    result = sim.simulate_fill(100.0, "BUY", 0)
    assert result.status == FillStatus.REJECTED
    assert result.filled_quantity == 0


def test_negative_quantity_rejected():
    sim = MarketSimulator()
    result = sim.simulate_fill(100.0, "BUY", -5)
    assert result.status == FillStatus.REJECTED


# ── Commission tests ─────────────────────────────────────────


def test_commission_percentage():
    sim = MarketSimulator(commission_pct=0.1, slippage_pct=0.0, spread_pct=0.0)
    result = sim.simulate_fill(100.0, "BUY", 100)
    # 0.1% of 100 * 100 = 10.0
    assert result.commission == 10.0


def test_minimum_commission():
    sim = MarketSimulator(commission_pct=0.1, slippage_pct=0.0, spread_pct=0.0, min_commission=5.0)
    result = sim.simulate_fill(10.0, "BUY", 1)
    # 0.1% of 10 * 1 = 0.01, but min is 5.0
    assert result.commission == 5.0


def test_zero_commission():
    sim = MarketSimulator(commission_pct=0.0, slippage_pct=0.0, spread_pct=0.0, min_commission=0.0)
    result = sim.simulate_fill(100.0, "BUY", 10)
    assert result.commission == 0.0


# ── Spread tests ─────────────────────────────────────────────


def test_spread_increases_buy_price():
    sim = MarketSimulator(commission_pct=0.0, slippage_pct=0.0, spread_pct=0.1, min_commission=0.0)
    result = sim.simulate_fill(100.0, "BUY", 10)
    # Half spread = 100 * 0.1% / 2 = 0.05
    assert result.fill_price == pytest.approx(100.05, abs=0.01)


def test_spread_decreases_sell_price():
    sim = MarketSimulator(commission_pct=0.0, slippage_pct=0.0, spread_pct=0.1, min_commission=0.0)
    result = sim.simulate_fill(100.0, "SELL", 10)
    assert result.fill_price == pytest.approx(99.95, abs=0.01)


# ── Market impact tests ─────────────────────────────────────


def test_market_impact_with_volume():
    sim = MarketSimulator(
        commission_pct=0.0, slippage_pct=0.0, spread_pct=0.0,
        market_impact_factor=0.1, min_commission=0.0,
    )
    # Large order: 1000 shares with 10000 bar volume = 10% ratio
    result = sim.simulate_fill(100.0, "BUY", 1000, bar_volume=10000)
    # Impact = 100 * (1000/10000) * 0.1 = 1.0
    assert result.fill_price == pytest.approx(101.0, abs=0.01)


def test_no_market_impact_without_volume():
    sim = MarketSimulator(
        commission_pct=0.0, slippage_pct=0.0, spread_pct=0.0,
        market_impact_factor=0.1, min_commission=0.0,
    )
    result = sim.simulate_fill(100.0, "BUY", 1000, bar_volume=None)
    assert result.fill_price == 100.0


def test_small_order_low_impact():
    sim = MarketSimulator(
        commission_pct=0.0, slippage_pct=0.0, spread_pct=0.0,
        market_impact_factor=0.1, min_commission=0.0,
    )
    # 10 shares in 100000 volume = 0.01% ratio
    result = sim.simulate_fill(100.0, "BUY", 10, bar_volume=100000)
    assert result.fill_price == pytest.approx(100.001, abs=0.01)


# ── Combined slippage test ───────────────────────────────────


def test_total_slippage_reported():
    sim = MarketSimulator(
        commission_pct=0.0, slippage_pct=0.05, spread_pct=0.02,
        min_commission=0.0,
    )
    result = sim.simulate_fill(100.0, "BUY", 10)
    assert result.slippage > 0
    assert result.fill_price == pytest.approx(100.0 + result.slippage, abs=0.01)


# ── Stop check tests ────────────────────────────────────────


def test_stop_triggered_when_low_hits_stop():
    sim = MarketSimulator()
    assert sim.simulate_stop_check(stop_price=95.0, bar_low=94.0, bar_high=101.0) is True


def test_stop_not_triggered_above_stop():
    sim = MarketSimulator()
    assert sim.simulate_stop_check(stop_price=95.0, bar_low=96.0, bar_high=101.0) is False


def test_stop_triggered_at_exact_price():
    sim = MarketSimulator()
    assert sim.simulate_stop_check(stop_price=95.0, bar_low=95.0, bar_high=100.0) is True


# ── Stop fill tests ──────────────────────────────────────────


def test_stop_fill_at_stop_price():
    sim = MarketSimulator(slippage_pct=0.0, spread_pct=0.0, commission_pct=0.0, min_commission=0.0)
    result = sim.simulate_stop_fill(stop_price=95.0, bar_open=98.0, bar_low=94.0, quantity=10)
    assert result.status == FillStatus.FILLED
    assert result.fill_price == 95.0


def test_stop_fill_gap_down():
    sim = MarketSimulator(slippage_pct=0.0, spread_pct=0.0, commission_pct=0.0, min_commission=0.0)
    # Bar opens at 90, below stop of 95 -> fill at 90 (worse)
    result = sim.simulate_stop_fill(stop_price=95.0, bar_open=90.0, bar_low=88.0, quantity=10)
    assert result.fill_price == 90.0


# ── SimulatedPosition tests ─────────────────────────────────


def test_position_is_open():
    pos = SimulatedPosition(symbol="AAPL", quantity=10, avg_cost=150.0)
    assert pos.is_open is True


def test_position_is_closed():
    pos = SimulatedPosition(symbol="AAPL", quantity=0, avg_cost=150.0)
    assert pos.is_open is False
