from app.monitoring.performance import PerformanceTracker


def test_record_api_call():
    tracker = PerformanceTracker()
    assert tracker.api_calls_today == 0
    assert tracker.api_cost_today_usd == 0.0

    tracker.record_api_call(0.0003)
    assert tracker.api_calls_today == 1
    assert tracker.api_cost_today_usd == 0.0003

    tracker.record_api_call(0.0003)
    assert tracker.api_calls_today == 2
    assert tracker.api_cost_today_usd == 0.0006


def test_reset_daily_clears_api_costs():
    tracker = PerformanceTracker()
    tracker.record_api_call(0.0003)
    tracker.record_api_call(0.0003)
    assert tracker.api_calls_today == 2

    tracker.reset_daily()
    assert tracker.api_calls_today == 0
    assert tracker.api_cost_today_usd == 0.0


def test_to_dict_includes_api_costs():
    tracker = PerformanceTracker()
    tracker.record_api_call(0.0003)
    d = tracker.to_dict()
    assert "api_calls_today" in d
    assert "api_cost_today_usd" in d
    assert d["api_calls_today"] == 1
    assert d["api_cost_today_usd"] == 0.0003
