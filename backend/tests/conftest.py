import os

import pytest

# Set test environment before importing app modules
os.environ["APP_ENV"] = "test"
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_DB"] = "trader_test"

from app.broker.mock_adapter import MockBrokerAdapter


@pytest.fixture
def mock_broker():
    """Provide a fresh mock broker for each test."""
    broker = MockBrokerAdapter(initial_cash=5000.0)
    return broker


@pytest.fixture
def mock_broker_with_positions(mock_broker):
    """Mock broker with some existing positions."""
    mock_broker.set_price("AAPL", 150.0)
    mock_broker.set_price("MSFT", 380.0)
    return mock_broker
