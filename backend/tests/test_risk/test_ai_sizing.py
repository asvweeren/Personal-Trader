from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.risk.ai_sizing import AISizingAdvisor, AISizingResult, _default_result

# ── AISizingResult ────────────────────────────────────────────


def test_result_to_dict():
    result = AISizingResult(modifier=1.2, reasoning="Strong momentum", risk_factors=["earnings"])
    d = result.to_dict()
    assert d["modifier"] == 1.2
    assert d["reasoning"] == "Strong momentum"
    assert d["risk_factors"] == ["earnings"]


def test_result_from_dict():
    data = {"modifier": 0.7, "reasoning": "High vol", "risk_factors": ["macro", "earnings"]}
    result = AISizingResult.from_dict(data)
    assert result.modifier == 0.7
    assert result.reasoning == "High vol"
    assert len(result.risk_factors) == 2


def test_result_from_dict_defaults():
    result = AISizingResult.from_dict({})
    assert result.modifier == 1.0
    assert result.reasoning == ""
    assert result.risk_factors == []


def test_default_result():
    result = _default_result()
    assert result.modifier == 1.0
    assert result.risk_factors == []


# ── AISizingAdvisor ───────────────────────────────────────────


@pytest.fixture
def advisor_no_key():
    """Advisor with no API key configured."""
    with patch("app.risk.ai_sizing.settings") as mock_settings:
        mock_settings.anthropic_api_key = ""
        mock_settings.redis_url = "redis://localhost:6379"
        mock_settings.ai_sizing_cache_ttl = 900
        advisor = AISizingAdvisor()
    return advisor


@pytest.fixture
def advisor_with_client():
    """Advisor with a mocked API client."""
    with patch("app.risk.ai_sizing.settings") as mock_settings:
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.redis_url = "redis://localhost:6379"
        mock_settings.ai_sizing_cache_ttl = 900
        advisor = AISizingAdvisor()
    # Replace with a mock client and disable Redis to avoid cross-test cache
    advisor._client = MagicMock()
    advisor._redis = None
    advisor._get_redis = AsyncMock(return_value=None)
    return advisor


@pytest.mark.asyncio
async def test_no_api_key_returns_default(advisor_no_key):
    result = await advisor_no_key.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ensemble",
        portfolio_summary={"total_value": 5000, "cash": 3000, "positions": 2},
    )
    assert result.modifier == 1.0


@pytest.mark.asyncio
async def test_valid_api_response(advisor_with_client):
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text='{"modifier": 1.3, "reasoning": "Strong alignment",'
            ' "risk_factors": ["earnings"]}'
        )
    ]
    advisor_with_client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.85,
        strategy_name="ensemble",
        portfolio_summary={"total_value": 5000, "cash": 3000, "positions": 1},
        features={"rsi": 55, "macd_hist": 0.5},
        sentiment={"sentiment_score": 0.6},
    )
    assert result.modifier == 1.3
    assert result.reasoning == "Strong alignment"
    assert "earnings" in result.risk_factors


@pytest.mark.asyncio
async def test_api_failure_returns_default(advisor_with_client):
    advisor_with_client._client.messages.create = AsyncMock(side_effect=Exception("API down"))

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert result.modifier == 1.0


@pytest.mark.asyncio
async def test_parse_error_returns_default(advisor_with_client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="not valid json")]
    advisor_with_client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert result.modifier == 1.0


@pytest.mark.asyncio
async def test_modifier_clamped_high(advisor_with_client):
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"modifier": 3.0, "reasoning": "Over the top", "risk_factors": []}')
    ]
    advisor_with_client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert result.modifier == 1.5


@pytest.mark.asyncio
async def test_modifier_clamped_low(advisor_with_client):
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"modifier": 0.1, "reasoning": "Too risky", "risk_factors": []}')
    ]
    advisor_with_client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert result.modifier == 0.5


@pytest.mark.asyncio
async def test_cache_hit_skips_api(advisor_with_client):
    cached = AISizingResult(modifier=1.1, reasoning="cached", risk_factors=[])
    advisor_with_client._memory_cache["AAPL"] = (cached, float("inf"))

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert result.modifier == 1.1
    assert result.reasoning == "cached"
    # API should not have been called
    advisor_with_client._client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_markdown_wrapped_response(advisor_with_client):
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text='```json\n{"modifier": 0.8, "reasoning": "Caution",'
            ' "risk_factors": ["vol"]}\n```'
        )
    ]
    advisor_with_client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert result.modifier == 0.8
    assert result.reasoning == "Caution"


# ── Call counter tracking ─────────────────────────────────────


@pytest.mark.asyncio
async def test_call_counter_increments(advisor_with_client):
    assert advisor_with_client.call_count == 0
    assert advisor_with_client.estimated_cost_usd == 0.0

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"modifier": 1.0, "reasoning": "ok", "risk_factors": []}')
    ]
    advisor_with_client._client.messages.create = AsyncMock(return_value=mock_response)

    await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert advisor_with_client.call_count == 1
    assert advisor_with_client.estimated_cost_usd == pytest.approx(0.0003)

    # Clear cache so second call hits the API
    advisor_with_client._memory_cache.clear()

    await advisor_with_client.get_modifier(
        symbol="MSFT",
        signal_confidence=0.7,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert advisor_with_client.call_count == 2
    assert advisor_with_client.estimated_cost_usd == pytest.approx(0.0006)


@pytest.mark.asyncio
async def test_call_counter_not_incremented_on_error(advisor_with_client):
    advisor_with_client._client.messages.create = AsyncMock(side_effect=Exception("down"))

    await advisor_with_client.get_modifier(
        symbol="AAPL",
        signal_confidence=0.8,
        strategy_name="ml",
        portfolio_summary={"total_value": 5000},
    )
    assert advisor_with_client.call_count == 0
