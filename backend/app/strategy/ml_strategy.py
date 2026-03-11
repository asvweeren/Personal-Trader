"""XGBoost-based trading strategy with hyperparameter tuning and model versioning."""

import json
import pickle
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.model_selection import TimeSeriesSplit

from app.data.indicators import compute_features
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal, TrainResult
from app.strategy.feature_pipeline import (
    FeaturePipelineConfig,
    get_feature_importance,
    prepare_ml_data,
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
        confidence_threshold: float = 0.50,
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

    def is_model_stale(self, days_threshold: int = 30) -> bool:
        """Check if model hasn't been retrained recently."""
        if not self._model_metadata:
            return True
        trained_at = self._model_metadata.get("trained_at")
        if not trained_at:
            return True
        trained_dt = datetime.fromisoformat(str(trained_at))
        days_old = (datetime.now(UTC) - trained_dt).days
        return days_old > days_threshold

    def get_robustness_score(self) -> float:
        """Get walk-forward robustness score (0=overfit, 1=robust).

        Returns 0.5 (neutral) if no walk-forward validation has been run.
        """
        return self._model_metadata.get("walk_forward_robustness", 0.5)

    def _check_feature_drift(self, latest: pd.DataFrame, symbol: str) -> bool:
        """Check if live features deviate significantly from training distribution.

        Returns True if drift is severe enough to skip this signal.
        """
        feature_stats = self._model_metadata.get("feature_stats")
        if not feature_stats:
            return False
        drift_count = 0
        for col in latest.columns:
            stats = feature_stats.get(col)
            if not stats:
                continue
            val = float(latest[col].iloc[0])
            mean = stats["mean"]
            std = stats["std"]
            if std > 0 and abs(val - mean) > 3 * std:
                drift_count += 1
                logger.warning(
                    "ml_strategy.feature_drift",
                    symbol=symbol,
                    feature=col,
                    value=round(val, 4),
                    training_mean=round(mean, 4),
                    training_std=round(std, 4),
                )
        # If >20% of features are drifting, signal is unreliable
        if drift_count > 0 and drift_count / len(latest.columns) > 0.2:
            logger.warning(
                "ml_strategy.drift_halt",
                symbol=symbol,
                drifted_features=drift_count,
                total_features=len(latest.columns),
            )
            return True
        return False

    def _is_binary_model(self) -> bool:
        """Check if the loaded model is a binary classifier."""
        return self._model_metadata.get("model_type") == "binary"

    def _get_probability_gate(self) -> float:
        """Get minimum probability gate based on training class distribution.

        For binary models, random chance is ~base_rate for BUY class.
        Gate must exceed base_rate + margin to ensure the model adds value.
        """
        class_dist = self._model_metadata.get("class_distribution", {})
        if not class_dist:
            return 0.55
        # class_dist keys may be strings (from JSON) or ints
        total = sum(class_dist.values())
        if total == 0:
            return 0.55
        buy_rate = class_dist.get(1, class_dist.get("1", 0)) / total
        return max(0.48, buy_rate + 0.05)

    def get_regime_threshold(self) -> float:
        """Get confidence threshold adjusted for current market regime.

        For binary models, random chance = ~50%, so thresholds are higher.
        For 3-class models, random chance = 33%.
        """
        base = self._confidence_threshold
        try:
            from app.strategy.regime import MarketRegime

            regime = getattr(self, "_current_regime", None)
            if regime is not None:
                if self._is_binary_model():
                    if regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                        base = 0.48
                    elif regime.regime == MarketRegime.RANGING:
                        base = 0.52
                    elif regime.regime == MarketRegime.HIGH_VOLATILITY:
                        base = 0.55
                else:
                    if regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                        return 0.38
                    elif regime.regime == MarketRegime.RANGING:
                        return 0.45
                    elif regime.regime == MarketRegime.HIGH_VOLATILITY:
                        return 0.50
        except Exception:
            pass

        # For binary models, also enforce the probability gate
        if self._is_binary_model():
            gate = self._get_probability_gate()
            return max(base, gate)
        return base

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []
        if self._model is None:
            return signals

        confidence_threshold = self.get_regime_threshold()

        # Increase threshold for stale models (add 15% penalty)
        if self.is_model_stale():
            confidence_threshold = min(confidence_threshold + 0.15, 0.70)
            logger.warning(
                "ml_strategy.stale_model",
                trained_at=self._model_metadata.get("trained_at"),
                adjusted_threshold=confidence_threshold,
            )

        # Increase threshold if walk-forward shows poor robustness
        robustness = self.get_robustness_score()
        if robustness < 0.3:
            confidence_threshold = min(confidence_threshold + 0.10, 0.75)
            logger.warning(
                "ml_strategy.low_robustness",
                robustness=robustness,
                adjusted_threshold=confidence_threshold,
            )

        regime_name = getattr(getattr(self, "_current_regime", None), "regime", None)
        logger.info(
            "ml_strategy.threshold",
            threshold=round(confidence_threshold, 3),
            regime=regime_name.value if regime_name else "none",
            symbols=len(market_data.ohlcv),
        )

        for symbol, df in market_data.ohlcv.items():
            if df.empty or len(df) < 50:
                continue

            try:
                # Use pre-computed features from snapshot if available
                features_df = market_data.computed_features_df.get(symbol)
                if features_df is None:
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

                # Check for feature drift — skip signal if too many features drifted
                if self._check_feature_drift(latest, symbol):
                    continue

                proba = self._model.predict_proba(latest)[0]

                if self._is_binary_model():
                    # Binary model: 0=NOT_BUY, 1=BUY
                    buy_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    confidence = buy_prob
                    if buy_prob >= confidence_threshold:
                        action = SignalAction.BUY
                    else:
                        action = SignalAction.HOLD
                    prob_metadata = {
                        "not_buy": float(proba[0]),
                        "buy": buy_prob,
                    }
                else:
                    # Legacy 3-class model: 0=SELL, 1=HOLD, 2=BUY
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
                    prob_metadata = {
                        "sell": float(proba[0]),
                        "hold": float(proba[1]),
                        "buy": float(proba[2]),
                    }

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
                            "probabilities": prob_metadata,
                        },
                    )
                )
            except Exception:
                logger.exception("ml_strategy.signal_error", symbol=symbol)

        return signals

    @staticmethod
    def _profit_score(y_true, y_pred) -> float:
        """CV scoring: 70% precision + 30% F1 on BUY class.

        Penalizes models that never predict BUY (signal_rate < 5%).
        """
        buy_mask = y_pred == 1
        if buy_mask.sum() == 0:
            return 0.0
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        signal_rate = buy_mask.sum() / len(y_pred)
        if signal_rate < 0.05:
            return prec * 0.5  # Too few signals
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        return 0.7 * prec + 0.3 * f1

    async def train(
        self, historical_data: pd.DataFrame, features_precomputed: bool = False,
    ) -> TrainResult:
        """Train the XGBoost model with full feature pipeline.

        Uses binary classification (BUY vs NOT_BUY) by default.
        Backward compatible: set binary_mode=False in config for legacy 3-class.
        """
        try:
            import xgboost as xgb

            binary_mode = True  # Default to binary

            # Prepare data through feature pipeline
            config = FeaturePipelineConfig(
                forward_periods=5,
                buy_threshold=0.015,
                sell_threshold=-0.015,
                correlation_threshold=0.85,
                train_pct=0.70,
                val_pct=0.15,
                binary_mode=binary_mode,
            )
            data = prepare_ml_data(
                historical_data, config, normalize=False,
                features_precomputed=features_precomputed,
            )

            if binary_mode:
                # Binary: scale_pos_weight handles class imbalance
                n_negative = int((data.y_train == 0).sum())
                n_positive = int((data.y_train == 1).sum())
                scale_pos_weight = n_negative / max(n_positive, 1)
                logger.info(
                    "ml_strategy.binary_class_balance",
                    n_positive=n_positive,
                    n_negative=n_negative,
                    scale_pos_weight=round(scale_pos_weight, 2),
                )
                sample_weights = None
                xgb_objective = "binary:logistic"
                xgb_eval_metric = "logloss"
                xgb_extra = {"scale_pos_weight": scale_pos_weight}
            else:
                # Legacy 3-class
                class_counts = data.y_train.value_counts()
                total = len(data.y_train)
                n_classes = len(class_counts)
                sample_weights = data.y_train.map(
                    lambda c: total / (n_classes * class_counts.get(c, 1))
                )
                xgb_objective = "multi:softprob"
                xgb_eval_metric = "mlogloss"
                xgb_extra = {"num_class": 3}
                scale_pos_weight = 1.0

            # Hyperparameter candidates (12 configs with varying regularization)
            # fmt: off
            param_grid = [
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10, "reg_alpha": 0.01, "reg_lambda": 1.0},  # noqa: E501
                {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "reg_alpha": 0.10, "reg_lambda": 1.0},  # noqa: E501
                {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.08, "reg_alpha": 0.50, "reg_lambda": 1.5},  # noqa: E501
                {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08, "reg_alpha": 0.10, "reg_lambda": 1.0},  # noqa: E501
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "reg_alpha": 0.01, "reg_lambda": 0.5},  # noqa: E501
                {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "reg_alpha": 0.10, "reg_lambda": 1.0},  # noqa: E501
                {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "reg_alpha": 0.50, "reg_lambda": 2.0},  # noqa: E501
                {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "reg_alpha": 0.10, "reg_lambda": 1.0},  # noqa: E501
                {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.50, "reg_lambda": 1.5},  # noqa: E501
                {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "reg_alpha": 1.00, "reg_lambda": 3.0},  # noqa: E501
                {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.02, "reg_alpha": 0.10, "reg_lambda": 1.0},  # noqa: E501
                {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.02, "reg_alpha": 0.50, "reg_lambda": 2.0},  # noqa: E501
            ]
            # fmt: on

            best_score = -1
            best_model = None
            best_params = {}
            cv_results = []

            for params in param_grid:
                model = xgb.XGBClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    objective=xgb_objective,
                    eval_metric=xgb_eval_metric,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 1.0),
                    min_child_weight=3,
                    use_label_encoder=False,
                    **xgb_extra,
                )

                # Time-series cross-validation on training set
                tscv = TimeSeriesSplit(n_splits=3)
                fold_scores = []

                for train_idx, val_idx in tscv.split(data.X_train):
                    X_tr = data.X_train.iloc[train_idx]
                    y_tr = data.y_train.iloc[train_idx]
                    X_va = data.X_train.iloc[val_idx]
                    y_va = data.y_train.iloc[val_idx]

                    fit_kwargs = {
                        "eval_set": [(X_va, y_va)],
                        "verbose": False,
                    }
                    if sample_weights is not None:
                        fit_kwargs["sample_weight"] = sample_weights.iloc[train_idx]

                    model.fit(X_tr, y_tr, **fit_kwargs)

                    y_pred = model.predict(X_va)
                    if binary_mode:
                        score = self._profit_score(y_va.values, y_pred)
                    else:
                        score = accuracy_score(y_va, y_pred)
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
                objective=xgb_objective,
                eval_metric=xgb_eval_metric,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=best_params.get("reg_alpha", 0.1),
                reg_lambda=best_params.get("reg_lambda", 1.0),
                min_child_weight=3,
                use_label_encoder=False,
                **xgb_extra,
            )
            fit_kwargs = {
                "eval_set": [(data.X_val, data.y_val)],
                "verbose": False,
            }
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights
            best_model.fit(data.X_train, data.y_train, **fit_kwargs)

            # Calibrate probabilities using isotonic regression on validation set
            try:
                calibrated_model = CalibratedClassifierCV(
                    best_model, method="isotonic", cv="prefit"
                )
                calibrated_model.fit(data.X_val, data.y_val)
                best_model = calibrated_model
                logger.info("ml_strategy.calibration_applied")
            except Exception:
                logger.warning("ml_strategy.calibration_failed")

            # Evaluate on validation and test sets
            val_preds = best_model.predict(data.X_val)
            test_preds = best_model.predict(data.X_test)
            val_accuracy = accuracy_score(data.y_val, val_preds)
            test_accuracy = accuracy_score(data.y_test, test_preds)

            # Per-class metrics
            if binary_mode:
                target_names = ["NOT_BUY", "BUY"]
                test_report = classification_report(
                    data.y_test, test_preds, target_names=target_names, output_dict=True
                )
                test_confusion = confusion_matrix(data.y_test, test_preds).tolist()
                buy_precision = test_report.get("BUY", {}).get("precision", 0.0)
                buy_f1 = test_report.get("BUY", {}).get("f1-score", 0.0)
                test_profit_score = self._profit_score(data.y_test.values, test_preds)
                logger.info(
                    "ml_strategy.binary_metrics",
                    buy_precision=round(buy_precision, 3),
                    buy_f1=round(buy_f1, 3),
                    profit_score=round(test_profit_score, 3),
                    confusion_matrix=test_confusion,
                )
            else:
                target_names = ["SELL", "HOLD", "BUY"]
                test_report = classification_report(
                    data.y_test, test_preds, target_names=target_names, output_dict=True
                )
                test_confusion = confusion_matrix(data.y_test, test_preds).tolist()
                test_profit_score = 0.0
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

            # Compute feature statistics for drift detection
            feature_stats = {}
            for col in data.feature_columns:
                feature_stats[col] = {
                    "mean": float(data.X_train[col].mean()),
                    "std": float(data.X_train[col].std()),
                    "min": float(data.X_train[col].min()),
                    "max": float(data.X_train[col].max()),
                }

            # Save model with metadata
            metadata = {
                "trained_at": datetime.now(UTC).isoformat(),
                "model_type": "binary" if binary_mode else "multiclass",
                "best_params": best_params,
                "cv_profit_score": float(best_score) if binary_mode else 0.0,
                "cv_mean_accuracy": (
                    float(best_score) if not binary_mode
                    else float(accuracy_score(data.y_val, val_preds))
                ),
                "val_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
                "test_profit_score": float(test_profit_score) if binary_mode else 0.0,
                "feature_count": len(data.feature_columns),
                "train_samples": len(data.X_train),
                "val_samples": len(data.X_val),
                "test_samples": len(data.X_test),
                "class_distribution": {
                    int(k): int(v)
                    for k, v in data.y_train.value_counts().to_dict().items()
                },
                "feature_importance": importance,
                "cv_results": cv_results,
                "classification_report": test_report,
                "confusion_matrix": test_confusion,
                "feature_stats": feature_stats,
            }

            self._model = best_model
            self._feature_columns = data.feature_columns
            self._model_metadata = metadata
            self._save_model(best_model, data.feature_columns, metadata)

            score_label = "profit_score" if binary_mode else "accuracy"
            logger.info(
                "ml_strategy.trained",
                mode="binary" if binary_mode else "multiclass",
                cv_score=round(best_score, 4),
                val_accuracy=round(val_accuracy, 4),
                test_accuracy=round(test_accuracy, 4),
                features=len(data.feature_columns),
            )

            return TrainResult(
                success=True,
                metrics={
                    f"cv_{score_label}": float(best_score),
                    "val_accuracy": float(val_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "test_profit_score": float(test_profit_score) if binary_mode else 0.0,
                    "best_params": best_params,
                    "feature_importance": importance,
                },
                message=(
                    f"Model trained ({('binary' if binary_mode else '3-class')}): "
                    f"CV {score_label}={best_score:.4f}, "
                    f"Val={val_accuracy:.4f}, Test={test_accuracy:.4f}"
                ),
            )
        except Exception as e:
            logger.exception("ml_strategy.train_error")
            return TrainResult(success=False, metrics={}, message=str(e))

    async def train_candidate(
        self, historical_data: pd.DataFrame, features_precomputed: bool = False,
    ) -> TrainResult:
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
            result = await self.train(historical_data, features_precomputed=features_precomputed)
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
