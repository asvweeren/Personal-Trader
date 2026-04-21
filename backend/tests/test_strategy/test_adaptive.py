"""Tests for the adaptive learning system."""

from app.strategy.adaptive import AdaptiveManager, RegimeProfile, SymbolProfile


def test_symbol_profile_adjustment():
    """Symbol with poor win rate should raise threshold."""
    p = SymbolProfile()
    # Record 15 losing trades
    for _ in range(12):
        p.record(-50.0, 0.6)
    for _ in range(3):
        p.record(100.0, 0.7)
    # 3/15 = 20% win rate → should get +0.10 adjustment
    assert p.trades == 15
    assert p.win_rate < 0.25
    assert p.confidence_adjustment == 0.10


def test_symbol_profile_good_performance():
    """Symbol with good win rate should lower threshold."""
    p = SymbolProfile()
    for _ in range(8):
        p.record(100.0, 0.7)
    for _ in range(4):
        p.record(-50.0, 0.5)
    # 8/12 = 67% win rate → should get -0.03 adjustment
    assert p.win_rate > 0.50
    assert p.confidence_adjustment == -0.03


def test_symbol_profile_insufficient_data():
    """Below 10 trades, no adjustment."""
    p = SymbolProfile()
    for _ in range(5):
        p.record(-100.0, 0.6)
    assert p.confidence_adjustment == 0.0


def test_regime_profile_update():
    """Regime profile tracks wins and hold times."""
    r = RegimeProfile(regime_name="trending_up")
    r.record(100.0, hold_minutes=30.0)
    r.record(50.0, hold_minutes=45.0)
    r.record(-20.0, hold_minutes=10.0)
    assert r.trades == 3
    assert r.wins == 2
    assert r.optimal_hold_minutes == 37.5  # avg of 30 and 45


def test_adaptive_manager_threshold():
    """Adaptive manager combines symbol and regime adjustments."""
    mgr = AdaptiveManager()
    # No data → no adjustment
    assert mgr.get_adjusted_threshold(0.50, "AAPL") == 0.50

    # Add symbol data with bad performance
    mgr._symbol_profiles["AAPL"] = SymbolProfile(
        trades=20, wins=4, losses=16, confidence_adjustment=0.10,
    )
    assert mgr.get_adjusted_threshold(0.50, "AAPL") == 0.60

    # Add regime adjustment
    mgr._regime_profiles["high_vol"] = RegimeProfile(
        regime_name="high_vol", trades=20, wins=3, confidence_adjustment=0.10,
    )
    assert mgr.get_adjusted_threshold(0.50, "AAPL", "high_vol") == 0.70


def test_adaptive_manager_clamped():
    """Threshold should be clamped to [0.30, 0.80]."""
    mgr = AdaptiveManager()
    mgr._symbol_profiles["SPY"] = SymbolProfile(
        trades=20, wins=18, confidence_adjustment=-0.03,
    )
    mgr._regime_profiles["trending"] = RegimeProfile(
        regime_name="trending", trades=20, wins=18, confidence_adjustment=-0.03,
    )
    # 0.35 - 0.03 - 0.03 = 0.29, clamped to 0.30
    assert mgr.get_adjusted_threshold(0.35, "SPY", "trending") == 0.30


def test_adaptive_skip_symbol():
    """Should skip symbols with very poor performance."""
    mgr = AdaptiveManager()
    # Few trades with okay win rate → don't skip
    mgr._symbol_profiles["TSLA"] = SymbolProfile(trades=2, wins=1, losses=1, total_pnl=-50)
    assert not mgr.should_skip_symbol("TSLA")

    # 5 trades, <25% win rate → skip
    mgr._symbol_profiles["TSLA"] = SymbolProfile(trades=5, wins=1, losses=4, total_pnl=-200)
    assert mgr.should_skip_symbol("TSLA")

    # 3 trades with big loss → skip
    mgr._symbol_profiles["TSLA"] = SymbolProfile(trades=3, wins=0, losses=3, total_pnl=-600)
    assert mgr.should_skip_symbol("TSLA")

    # 3 consecutive losses → skip
    mgr._symbol_profiles["TSLA"] = SymbolProfile(trades=4, wins=2, losses=2, total_pnl=50, consecutive_losses=3)
    assert mgr.should_skip_symbol("TSLA")


def test_adaptive_feature_tracking():
    """Feature effectiveness tracking works."""
    mgr = AdaptiveManager()
    # Record 40 outcomes where feature 'sma_10' is in top features
    for i in range(40):
        pnl = 100.0 if i % 5 == 0 else -50.0  # 20% win rate for this feature
        mgr.record_trade_outcome(
            symbol="SPY", pnl=pnl, confidence=0.6,
            strategy_name="ml", regime="unknown",
            top_features=["sma_10", "rsi_14"],
        )
    # sma_10 should be declining (< 40% hit rate)
    declining = mgr.get_declining_features(min_samples=30)
    assert "sma_10" in declining


def test_adaptive_record_and_persist(tmp_path):
    """Recording outcomes and saving/loading state."""
    import app.strategy.adaptive as adaptive_module
    original_path = adaptive_module.ADAPTIVE_STATE_PATH
    adaptive_module.ADAPTIVE_STATE_PATH = tmp_path / "adaptive_state.json"

    try:
        mgr = AdaptiveManager()
        mgr.record_trade_outcome(
            symbol="AAPL", pnl=150.0, confidence=0.7,
            strategy_name="ml", regime="trending_up",
            hold_minutes=30.0, top_features=["sma_10"],
        )
        mgr.save_state()

        # Load into new manager
        mgr2 = AdaptiveManager()
        assert "AAPL" in mgr2._symbol_profiles
        assert mgr2._symbol_profiles["AAPL"].trades == 1
        assert "trending_up" in mgr2._regime_profiles
    finally:
        adaptive_module.ADAPTIVE_STATE_PATH = original_path
