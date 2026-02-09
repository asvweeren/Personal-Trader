"""XGBoost-based trading strategy with hyperparameter tuning and model versioning."""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score

from app.data.indicators import compute_features
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TrainResult, TradingSignal
from app.strategy.feature_pipeline import (
    FeaturePipelineConfig,
    prepare_ml_data,
    get_feature_importance,
)

logger = structlog.get_logger()

MODEL_DIR = Path("ml/models")


class MLStrategy(Strategy):
    """XGBoost-based trading strategy using technical indicators.

    Features:
    - Automated feature engineering via feature_pipeline
    - Hyperparameter tuning with TimeSeriesSplit CV
    - Model versioning with metadata (date, metrics, features)
    - Class weight balancing for imbalanced targets
    - Feature importance tracking
    """

    @property
    def name(self) -> str:
        return "ml_xgboost"

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        model_path: str | None = None,
    ):
        self._model = None
        self._confidence_threshold = confidence_threshold
        self._feature_columns: list[str] = []
        self._model_metadata: dict = {}
        self._model_path = Path(model_path) if model_path else MODEL_DIR / "xgboost_model.pkl"
        self._load_model()

    def _load_model(self) -> None:
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    saved = pickle.load(f)  # noqa: S301
                    self._model = saved["model"]
                    self._feature_columns = saved["feature_columns"]
                    self._model_metadata = saved.get("metadata", {})
                logger.info(
                    "ml_strategy.model_loaded",
                    features=len(self._feature_columns),
                    trained_at=self._model_metadata.get("trained_at", "unknown"),
                )
            except Exception:
                logger.warning("ml_strategy.model_load_error")

    def _save_model(self, model, feature_columns: list[str], metadata: dict) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "metadata": metadata,
                },
                f,
            )
        # Also save metadata as JSON for easy inspection
        meta_path = self._model_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("ml_strategy.model_saved", path=str(self._model_path))

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []
        if self._model is None:
            return signals

        for symbol, df in market_data.ohlcv.items():
            if df.empty or len(df) < 50:
                continue

            try:
                features_df = compute_features(df)
                # Ensure all required columns exist
                missing = set(self._feature_columns) - set(features_df.columns)
                if missing:
                    logger.warning(
                        "ml_strategy.missing_features",
                        symbol=symbol,
                        missing=list(missing),
                    )
                    continue

                latest = features_df[self._feature_columns].iloc[-1:]

                if latest.isnull().any(axis=1).iloc[0]:
                    continue

                # Model predicts: 0=SELL, 1=HOLD, 2=BUY
                proba = self._model.predict_proba(latest)[0]
                pred_class = int(np.argmax(proba))
                confidence = float(proba[pred_class])

                if confidence < self._confidence_threshold:
                    action = SignalAction.HOLD
                elif pred_class == 2:
                    action = SignalAction.BUY
                elif pred_class == 0:
                    action = SignalAction.SELL
                else:
                    action = SignalAction.HOLD

                # Also add pre-computed features from the pipeline if available
                enriched_features = market_data.features.get(symbol, {})
                feature_snapshot = {
                    **latest.iloc[0].to_dict(),
                    **enriched_features,
                }

                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        strategy_name=self.name,
                        features_snapshot=feature_snapshot,
                        metadata={
                            "probabilities": {
                                "sell": float(proba[0]),
                                "hold": float(proba[1]),
                                "buy": float(proba[2]),
                            },
                        },
                    )
                )
            except Exception:
                logger.exception("ml_strategy.signal_error", symbol=symbol)

        return signals

    async def train(self, historical_data: pd.DataFrame) -> TrainResult:
        """Train the XGBoost model with full feature pipeline."""
        try:
            import xgboost as xgb

            # Prepare data through feature pipeline
            config = FeaturePipelineConfig(
                forward_periods=5,
                buy_threshold=0.005,
                sell_threshold=-0.005,
                correlation_threshold=0.95,
                train_pct=0.70,
                val_pct=0.15,
            )
            data = prepare_ml_data(historical_data, config, normalize=False)

            # Calculate class weights for imbalanced data
            class_counts = data.y_train.value_counts()
            total = len(data.y_train)
            n_classes = len(class_counts)
            sample_weights = data.y_train.map(
                lambda c: total / (n_classes * class_counts.get(c, 1))
            )

            # Hyperparameter candidates
            param_grid = [
                {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
                {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
                {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08},
            ]

            best_score = -1
            best_model = None
            best_params = {}
            cv_results = []

            for params in param_grid:
                model = xgb.XGBClassifier(
                    **params,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_weight=3,
                    use_label_encoder=False,
                )

                # Time-series cross-validation on training set
                tscv = TimeSeriesSplit(n_splits=3)
                fold_scores = []

                for train_idx, val_idx in tscv.split(data.X_train):
                    X_tr = data.X_train.iloc[train_idx]
                    y_tr = data.y_train.iloc[train_idx]
                    X_va = data.X_train.iloc[val_idx]
                    y_va = data.y_train.iloc[val_idx]
                    sw_tr = sample_weights.iloc[train_idx]

                    model.fit(
                        X_tr, y_tr,
                        sample_weight=sw_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                    )
                    score = accuracy_score(y_va, model.predict(X_va))
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)
                cv_results.append({
                    "params": params,
                    "cv_mean": float(avg_score),
                    "cv_scores": [float(s) for s in fold_scores],
                })

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    best_model = model

            # Retrain best model on full training set
            best_model = xgb.XGBClassifier(
                **best_params,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                use_label_encoder=False,
            )
            best_model.fit(
                data.X_train, data.y_train,
                sample_weight=sample_weights,
                eval_set=[(data.X_val, data.y_val)],
                verbose=False,
            )

            # Evaluate on validation and test sets
            val_preds = best_model.predict(data.X_val)
            test_preds = best_model.predict(data.X_test)
            val_accuracy = accuracy_score(data.y_val, val_preds)
            test_accuracy = accuracy_score(data.y_test, test_preds)

            # Feature importance
            importance = get_feature_importance(
                best_model, data.feature_columns, top_n=15
            )

            # Save model with metadata
            metadata = {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "best_params": best_params,
                "cv_mean_accuracy": float(best_score),
                "val_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
                "feature_count": len(data.feature_columns),
                "train_samples": len(data.X_train),
                "val_samples": len(data.X_val),
                "test_samples": len(data.X_test),
                "class_distribution": data.y_train.value_counts().to_dict(),
                "feature_importance": importance,
                "cv_results": cv_results,
            }

            self._model = best_model
            self._feature_columns = data.feature_columns
            self._model_metadata = metadata
            self._save_model(best_model, data.feature_columns, metadata)

            logger.info(
                "ml_strategy.trained",
                cv_accuracy=round(best_score, 4),
                val_accuracy=round(val_accuracy, 4),
                test_accuracy=round(test_accuracy, 4),
                features=len(data.feature_columns),
            )

            return TrainResult(
                success=True,
                metrics={
                    "cv_accuracy": float(best_score),
                    "val_accuracy": float(val_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "best_params": best_params,
                    "feature_importance": importance,
                },
                message=(
                    f"Model trained: CV={best_score:.4f}, "
                    f"Val={val_accuracy:.4f}, Test={test_accuracy:.4f}"
                ),
            )
        except Exception as e:
            logger.exception("ml_strategy.train_error")
            return TrainResult(success=False, metrics={}, message=str(e))

    def get_confidence(self) -> float:
        if self._model_metadata:
            return self._model_metadata.get("val_accuracy", 0.5)
        return 0.5

    def get_model_info(self) -> dict:
        """Return model metadata for API/dashboard."""
        return {
            "name": self.name,
            "model_loaded": self._model is not None,
            "feature_count": len(self._feature_columns),
            "confidence_threshold": self._confidence_threshold,
            **self._model_metadata,
        }
