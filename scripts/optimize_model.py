"""Optimize the XGBoost model with expanded hyperparameter search.

Improvements over default train_model.py:
- Wider target thresholds (1% instead of 0.5%)
- Extended hyperparameter grid (12 combos)
- 5-fold TimeSeriesSplit CV
- Combined training across all symbols
- Additional derived features
- Reports per-class precision/recall

Usage:
    python scripts/optimize_model.py
    python scripts/optimize_model.py --forward-periods 3
    python scripts/optimize_model.py --buy-threshold 0.01 --sell-threshold -0.01
"""

import argparse
import asyncio
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from app.data.indicators import compute_features
from app.strategy.feature_pipeline import (
    FeaturePipelineConfig,
    create_target,
    select_feature_columns,
    remove_low_variance,
    remove_highly_correlated,
    time_based_split,
)

logger = structlog.get_logger()

DATA_DIR = Path(__file__).parent.parent / "ml" / "data"
MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional derived features beyond the standard compute_features."""
    close = df["close"]

    # Price position relative to range
    if "sma_50" in df.columns:
        df["price_vs_sma50"] = close / df["sma_50"] - 1
    if "sma_10" in df.columns:
        df["price_vs_sma10"] = close / df["sma_10"] - 1

    # Momentum
    df["momentum_10d"] = close / close.shift(10) - 1
    df["momentum_20d"] = close / close.shift(20) - 1

    # Volume change
    if "volume" in df.columns:
        df["volume_change_5d"] = df["volume"] / df["volume"].shift(5) - 1
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Volatility ratio
    if "volatility_20d" in df.columns:
        vol_10 = close.pct_change().rolling(10).std()
        df["vol_ratio_10_20"] = vol_10 / df["volatility_20d"]

    # RSI momentum (rate of change of RSI)
    if "rsi_14" in df.columns:
        df["rsi_change_5d"] = df["rsi_14"] - df["rsi_14"].shift(5)

    # MACD strength
    if "macd" in df.columns and "macd_signal" in df.columns:
        df["macd_divergence"] = df["macd"] - df["macd_signal"]

    return df


def load_and_prepare_data(
    data_dir: Path,
    symbols: list[str],
    forward_periods: int,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Load all symbol data, compute features, and combine."""
    all_frames = []
    for symbol in symbols:
        path = data_dir / f"{symbol}_1d.csv"
        if not path.exists():
            print(f"  SKIP {symbol}: no data file")
            continue

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = compute_features(df)
        df = add_extra_features(df)
        df["target"] = create_target(
            df,
            forward_periods=forward_periods,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        df = df.dropna()
        all_frames.append(df)
        print(f"  {symbol}: {len(df)} bars after feature engineering")

    if not all_frames:
        raise ValueError("No data loaded")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Select features
    feature_cols = select_feature_columns(combined)
    feature_cols = remove_low_variance(combined, feature_cols)
    feature_cols = remove_highly_correlated(combined, feature_cols, threshold=0.95)

    print(f"\n  Combined: {len(combined)} bars, {len(feature_cols)} features")
    print(f"  Class distribution: {combined['target'].value_counts().to_dict()}")

    return combined, feature_cols


def run_optimization(
    combined: pd.DataFrame,
    feature_cols: list[str],
    n_cv_folds: int = 5,
) -> dict:
    """Run hyperparameter optimization with expanded grid."""
    import xgboost as xgb

    # Time-based split
    train_df, val_df, test_df = time_based_split(combined, 0.70, 0.15)

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    # Class weights
    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    sample_weights = y_train.map(
        lambda c: total / (n_classes * class_counts.get(c, 1))
    )

    # Expanded hyperparameter grid
    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 1.0, "min_child_weight": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 2.0, "min_child_weight": 3},
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.3, "reg_lambda": 1.5, "min_child_weight": 5},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.6, "reg_alpha": 1.0, "reg_lambda": 1.0, "min_child_weight": 7},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 0.5, "min_child_weight": 3},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.07, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 1.0, "min_child_weight": 3},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 2.0, "min_child_weight": 7},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 1.5, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.3, "reg_lambda": 1.0, "min_child_weight": 3},
    ]

    print(f"\nSearching {len(param_grid)} hyperparameter combos with {n_cv_folds}-fold CV...")
    print("-" * 60)

    best_score = -1
    best_params = {}
    cv_results = []

    for i, params in enumerate(param_grid):
        # Extract XGBClassifier-specific params
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            **params,
        }

        model = xgb.XGBClassifier(**xgb_params)

        tscv = TimeSeriesSplit(n_splits=n_cv_folds)
        fold_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]
            sw_tr = sample_weights.iloc[train_idx]

            model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
            score = accuracy_score(y_va, model.predict(X_va))
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        cv_results.append({
            "params": params,
            "cv_mean": float(avg_score),
            "cv_std": float(std_score),
            "cv_scores": [float(s) for s in fold_scores],
        })

        marker = " ***" if avg_score > best_score else ""
        print(f"  [{i+1:2d}/{len(param_grid)}] CV={avg_score:.4f} (+/- {std_score:.4f}) "
              f"depth={params['max_depth']} lr={params['learning_rate']} "
              f"n={params['n_estimators']}{marker}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Sort results
    cv_results.sort(key=lambda x: x["cv_mean"], reverse=True)

    print(f"\n{'='*60}")
    print(f"Best CV: {best_score:.4f}")
    print(f"Best params: {best_params}")

    # Retrain best model on full training data
    print(f"\nRetraining best model on full training set ({len(X_train)} samples)...")
    final_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        **best_params,
    )
    final_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    val_preds = final_model.predict(X_val)
    test_preds = final_model.predict(X_test)
    val_accuracy = accuracy_score(y_val, val_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    print(f"\nValidation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy:       {test_accuracy:.4f}")

    print("\n--- Validation Classification Report ---")
    print(classification_report(
        y_val, val_preds,
        target_names=["SELL (0)", "HOLD (1)", "BUY (2)"],
        zero_division=0,
    ))

    print("--- Test Classification Report ---")
    print(classification_report(
        y_test, test_preds,
        target_names=["SELL (0)", "HOLD (1)", "BUY (2)"],
        zero_division=0,
    ))

    # Feature importance
    importances = final_model.feature_importances_
    imp_dict = dict(zip(feature_cols, importances))
    sorted_imp = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:15])

    print("\nTop 15 features:")
    for feat, imp in sorted_imp.items():
        print(f"  {feat:<25s} {imp:.4f}")

    return {
        "model": final_model,
        "feature_columns": feature_cols,
        "best_params": best_params,
        "cv_mean_accuracy": float(best_score),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "feature_importance": {k: str(v) for k, v in sorted_imp.items()},
        "cv_results": cv_results[:5],  # Top 5
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "class_distribution": y_train.value_counts().to_dict(),
    }


def save_model(result: dict, model_dir: Path) -> None:
    """Save model and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_params": result["best_params"],
        "cv_mean_accuracy": result["cv_mean_accuracy"],
        "val_accuracy": result["val_accuracy"],
        "test_accuracy": result["test_accuracy"],
        "feature_count": len(result["feature_columns"]),
        "train_samples": result["train_samples"],
        "val_samples": result["val_samples"],
        "test_samples": result["test_samples"],
        "class_distribution": result["class_distribution"],
        "feature_importance": result["feature_importance"],
        "cv_results": result["cv_results"],
    }

    pkl_path = model_dir / "xgboost_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": result["model"],
            "feature_columns": result["feature_columns"],
            "metadata": metadata,
        }, f)

    json_path = model_dir / "xgboost_model.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nModel saved to {pkl_path}")
    print(f"Metadata saved to {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize XGBoost trading model")
    parser.add_argument("--forward-periods", type=int, default=5,
                        help="Forward look-ahead periods for target (default: 5)")
    parser.add_argument("--buy-threshold", type=float, default=0.01,
                        help="Buy target threshold (default: 0.01 = 1%%)")
    parser.add_argument("--sell-threshold", type=float, default=-0.01,
                        help="Sell target threshold (default: -0.01 = -1%%)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--symbols", nargs="+",
                        default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA",
                                 "AMZN", "META", "IWM", "EFA", "VGK"])
    return parser.parse_args()


def main():
    args = parse_args()

    print("MODEL OPTIMIZATION")
    print("=" * 60)
    print(f"  Forward periods: {args.forward_periods}")
    print(f"  Buy threshold: {args.buy_threshold:.1%}")
    print(f"  Sell threshold: {args.sell_threshold:.1%}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print()

    combined, feature_cols = load_and_prepare_data(
        Path(args.data_dir),
        args.symbols,
        args.forward_periods,
        args.buy_threshold,
        args.sell_threshold,
    )

    result = run_optimization(combined, feature_cols, args.cv_folds)

    save_model(result, MODEL_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
