"""Tests for model experiment tracking model."""

import pytest
from datetime import datetime

from app.models.model_experiment import ModelExperiment


class TestModelExperiment:
    def test_table_name(self):
        assert ModelExperiment.__tablename__ == "model_experiments"

    def test_create_instance(self):
        exp = ModelExperiment(
            strategy_name="ml_xgboost",
            version="v1",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
            train_metrics={"accuracy": 0.85, "f1": 0.82},
            val_metrics={"accuracy": 0.80},
            test_metrics={},
            feature_importance={"rsi_14": 0.15, "macd": 0.12},
            is_active=False,
            notes="Initial training run",
        )
        assert exp.strategy_name == "ml_xgboost"
        assert exp.version == "v1"
        assert exp.hyperparameters["n_estimators"] == 100
        assert exp.is_active is False

    def test_defaults(self):
        exp = ModelExperiment(
            strategy_name="test",
            version="v1",
            is_active=False,
        )
        assert exp.is_active is False
        assert exp.notes is None

    def test_jsonb_fields(self):
        """Verify JSONB fields accept nested dicts."""
        exp = ModelExperiment(
            strategy_name="test",
            version="v1",
            hyperparameters={"nested": {"key": "value"}},
            train_metrics={"confusion_matrix": [[10, 2], [3, 15]]},
            feature_importance={},
        )
        assert exp.hyperparameters["nested"]["key"] == "value"
        assert exp.train_metrics["confusion_matrix"][0][0] == 10
