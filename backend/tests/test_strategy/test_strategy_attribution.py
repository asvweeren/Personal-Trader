"""Tests for per-strategy performance attribution."""


from app.monitoring.performance import PerformanceTracker, StrategyMetrics


class TestStrategyMetrics:
    def test_win_rate_no_trades(self):
        sm = StrategyMetrics()
        assert sm.win_rate == 0.0

    def test_win_rate_with_trades(self):
        sm = StrategyMetrics(trades=10, wins=7, losses=3)
        assert sm.win_rate == 70.0

    def test_profit_factor_no_losses(self):
        sm = StrategyMetrics()
        sm.trade_pnls = [10.0, 20.0, 30.0]
        assert sm.profit_factor == float("inf")

    def test_profit_factor_no_profits(self):
        sm = StrategyMetrics()
        sm.trade_pnls = [-10.0, -20.0]
        assert sm.profit_factor == 0.0

    def test_profit_factor_mixed(self):
        sm = StrategyMetrics()
        sm.trade_pnls = [30.0, -10.0]
        assert sm.profit_factor == 3.0

    def test_avg_hold_time_empty(self):
        sm = StrategyMetrics()
        assert sm.avg_hold_time == 0.0

    def test_avg_hold_time(self):
        sm = StrategyMetrics()
        sm.hold_durations = [60.0, 120.0, 180.0]
        assert sm.avg_hold_time == 120.0

    def test_to_dict(self):
        sm = StrategyMetrics(trades=5, wins=3, losses=2, pnl=150.0)
        sm.trade_pnls = [50.0, 30.0, 70.0, -20.0, -30.0]
        d = sm.to_dict()
        assert d["trades"] == 5
        assert d["wins"] == 3
        assert d["win_rate"] == 60.0
        assert d["pnl"] == 150.0
        assert "profit_factor" in d
        assert "avg_hold_time_minutes" in d


class TestPerformanceTrackerAttribution:
    def test_record_trade_with_strategy(self):
        tracker = PerformanceTracker()
        tracker.record_trade(pnl=50.0, strategy_name="ml_xgboost", confidence=0.7)

        assert "ml_xgboost" in tracker.strategy_metrics
        sm = tracker.strategy_metrics["ml_xgboost"]
        assert sm.trades == 1
        assert sm.wins == 1
        assert sm.pnl == 50.0
        assert sm.avg_confidence == 0.7

    def test_record_trade_without_strategy(self):
        tracker = PerformanceTracker()
        tracker.record_trade(pnl=50.0)
        assert len(tracker.strategy_metrics) == 0

    def test_multiple_strategies(self):
        tracker = PerformanceTracker()
        tracker.record_trade(pnl=50.0, strategy_name="ml_xgboost", confidence=0.8)
        tracker.record_trade(pnl=-20.0, strategy_name="sentiment", confidence=0.6)
        tracker.record_trade(pnl=30.0, strategy_name="ml_xgboost", confidence=0.7)

        assert len(tracker.strategy_metrics) == 2
        ml = tracker.strategy_metrics["ml_xgboost"]
        assert ml.trades == 2
        assert ml.pnl == 80.0
        assert ml.wins == 2

        sent = tracker.strategy_metrics["sentiment"]
        assert sent.trades == 1
        assert sent.losses == 1

    def test_hold_duration_recorded(self):
        tracker = PerformanceTracker()
        tracker.record_trade(
            pnl=50.0,
            strategy_name="ml_xgboost",
            hold_duration_minutes=120.0,
        )
        sm = tracker.strategy_metrics["ml_xgboost"]
        assert len(sm.hold_durations) == 1
        assert sm.hold_durations[0] == 120.0

    def test_hold_duration_zero_not_recorded(self):
        tracker = PerformanceTracker()
        tracker.record_trade(
            pnl=50.0,
            strategy_name="ml_xgboost",
            hold_duration_minutes=0.0,
        )
        sm = tracker.strategy_metrics["ml_xgboost"]
        assert len(sm.hold_durations) == 0

    def test_get_strategy_breakdown(self):
        tracker = PerformanceTracker()
        tracker.record_trade(pnl=50.0, strategy_name="ml_xgboost")
        tracker.record_trade(pnl=-10.0, strategy_name="sentiment")

        breakdown = tracker.get_strategy_breakdown()
        assert "ml_xgboost" in breakdown
        assert "sentiment" in breakdown
        assert breakdown["ml_xgboost"]["pnl"] == 50.0
        assert breakdown["sentiment"]["pnl"] == -10.0

    def test_to_dict_includes_breakdown(self):
        tracker = PerformanceTracker()
        tracker.record_trade(pnl=50.0, strategy_name="ml_xgboost")

        d = tracker.to_dict()
        assert "strategy_breakdown" in d
        assert "ml_xgboost" in d["strategy_breakdown"]

    def test_avg_confidence_updates(self):
        tracker = PerformanceTracker()
        tracker.record_trade(pnl=50.0, strategy_name="ml", confidence=0.8)
        tracker.record_trade(pnl=30.0, strategy_name="ml", confidence=0.6)

        sm = tracker.strategy_metrics["ml"]
        assert abs(sm.avg_confidence - 0.7) < 0.01
