"""XGBoost-based trading strategy with hyperparameter tuning and model versioning."""

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        confidence_threshold: float = 0.75,
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

    def get_regime_threshold(self) -> float:
        """Get confidence threshold adjusted for current market regime."""
        try:
            from app.strategy.regime import MarketRegime
            from app.risk.reconciliation import get_last_result  # noqa: F401
            # Access regime from engine's detector via module-level reference
            regime = getattr(self, "_current_regime", None)
            if regime is not None:
                if regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                    return 0.65  # Slightly relaxed in clear trends
                elif regime.regime == MarketRegime.RANGING:
                    return 0.80  # Conservative in ranges
                elif regime.regime == MarketRegime.HIGH_VOLATILITY:
                    return 0.85  # Very conservative in high vol
        except Exception:
            pass
        return self._confidence_threshold

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []
        if self._model is None:
            return signals

        confidence_threshold = self.get_regime_threshold()

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

                if confidence < confidence_threshold:
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
                forward_periods=10,
                buy_threshold=0.015,
                sell_threshold=-0.015,
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

            # Hyperparameter candidates (12 configs with varying regularization)
            param_grid = [
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.08, "reg_alpha": 0.5, "reg_lambda": 1.5},
                {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "reg_alpha": 0.01, "reg_lambda": 0.5},
                {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 2.0},
                {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.5, "reg_lambda": 1.5},
                {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "reg_alpha": 1.0, "reg_lambda": 3.0},
                {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.02, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.02, "reg_alpha": 0.5, "reg_lambda": 2.0},
            ]

            best_score = -1
            best_model = None
            best_params = {}
            cv_results = []

            for params in param_grid:
                model = xgb.XGBClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 1.0),
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
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                learning_rate=best_params["learning_rate"],
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=best_params.get("reg_alpha", 0.1),
                reg_lambda=best_params.get("reg_lambda", 1.0),
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

            # Per-class metrics (confusion matrix + classification report)
            target_names = ["SELL", "HOLD", "BUY"]
            test_report = classification_report(
                data.y_test, test_preds, target_names=target_names, output_dict=True
            )
            test_confusion = confusion_matrix(data.y_test, test_preds).tolist()
            logger.info(
                "ml_strategy.per_class_metrics",
                sell_f1=round(test_report["SELL"]["f1-score"], 3),
                hold_f1=round(test_report["HOLD"]["f1-score"], 3),
                buy_f1=round(test_report["BUY"]["f1-score"], 3),
                confusion_matrix=test_confusion,
            )

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
                "classification_report": test_report,
                "confusion_matrix": test_confusion,
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

    async def train_candidate(self, historical_data: pd.DataFrame) -> TrainResult:
        """Train a candidate model without affecting the live model.

        Saves to xgboost_model_candidate.pkl so the live model stays untouched.
        """
        candidate_path = self._model_path.parent / "xgboost_model_candidate.pkl"

        # Temporarily swap model path, train, then restore
        original_path = self._model_path
        original_model = self._model
        original_features = self._feature_columns
        original_metadata = self._model_metadata

        try:
            self._model_path = candidate_path
            result = await self.train(historical_data)
        finally:
            # Restore live model state regardless of outcome
            self._model_path = original_path
            self._model = original_model
            self._feature_columns = original_features
            self._model_metadata = original_metadata

        return result

    def hot_swap_model(self, candidate_path: Path | None = None) -> bool:
        """Swap the live model with a trained candidate.

        1. Backup current model to xgboost_model_backup.pkl
        2. Move candidate to the live model path
        3. Reload
        4. On failure: rollback from backup
        """
        if candidate_path is None:
            candidate_path = self._model_path.parent / "xgboost_model_candidate.pkl"

        if not candidate_path.exists():
            logger.error("ml_strategy.hot_swap_no_candidate", path=str(candidate_path))
            return False

        backup_path = self._model_path.parent / "xgboost_model_backup.pkl"

        try:
            # Backup current model
            if self._model_path.exists():
                shutil.copy2(self._model_path, backup_path)
                logger.info("ml_strategy.model_backed_up", path=str(backup_path))

            # Move candidate to live
            shutil.move(str(candidate_path), str(self._model_path))

            # Also move candidate metadata JSON if it exists
            candidate_meta = candidate_path.with_suffix(".json")
            if candidate_meta.exists():
                shutil.move(str(candidate_meta), str(self._model_path.with_suffix(".json")))

            # Reload
            self._load_model()

            if self._model is None:
                raise RuntimeError("Model failed to load after swap")

            logger.info("ml_strategy.hot_swap_success")
            return True

        except Exception:
            logger.exception("ml_strategy.hot_swap_error")
            # Rollback
            if backup_path.exists():
                try:
                    shutil.copy2(backup_path, self._model_path)
                    self._load_model()
                    logger.info("ml_strategy.rollback_success")
                except Exception:
                    logger.exception("ml_strategy.rollback_error")
            return False

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
