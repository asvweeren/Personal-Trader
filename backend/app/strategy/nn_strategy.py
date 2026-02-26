"""PyTorch LSTM neural network strategy for sequence-based prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from app.data.indicators import compute_features
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal, TrainResult
from app.strategy.feature_pipeline import (
    create_target,
    normalize_features,
    remove_highly_correlated,
    remove_low_variance,
    select_feature_columns,
    time_based_split,
)

logger = structlog.get_logger()

MODEL_DIR = Path("ml/models")


def _create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM input.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        lookback: Number of past timesteps per sequence.

    Returns:
        Xs of shape (n_sequences, lookback, n_features),
        ys of shape (n_sequences,).
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


class NNStrategy(Strategy):
    """PyTorch LSTM strategy for time-series classification.

    Uses a sequence of technical indicators over a lookback window
    to predict BUY/HOLD/SELL.
    """

    @property
    def name(self) -> str:
        return "nn_lstm"

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        confidence_threshold: float = 0.45,
        model_path: str | None = None,
    ):
        self._lookback = lookback
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._feature_columns: list[str] = []
        self._norm_stats: dict | None = None
        self._model_path = Path(model_path) if model_path else MODEL_DIR / "lstm_model.pkl"
        self._load_model()

    def _load_model(self) -> None:
        if not self._model_path.exists():
            return
        try:
            import torch
            saved = torch.load(self._model_path, weights_only=False)
            self._feature_columns = saved["feature_columns"]
            self._norm_stats = saved["norm_stats"]
            n_features = len(self._feature_columns)

            model = _LSTMClassifier(
                n_features, self._hidden_size, self._num_layers, self._dropout
            )
            model.load_state_dict(saved["state_dict"])
            model.eval()
            self._model = model
            logger.info("nn_strategy.model_loaded", features=n_features)
        except Exception:
            logger.warning("nn_strategy.model_load_error")

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []
        if self._model is None:
            return signals

        import torch

        for symbol, df in market_data.ohlcv.items():
            if df.empty or len(df) < self._lookback + 50:
                continue

            try:
                features_df = compute_features(df)
                missing = set(self._feature_columns) - set(features_df.columns)
                if missing:
                    continue

                # Get last lookback rows of features
                recent = features_df[self._feature_columns].iloc[-self._lookback:]
                if recent.isnull().any().any():
                    continue

                # Normalize using training statistics
                values = recent.values.astype(np.float32)
                if self._norm_stats:
                    means = np.array([
                        self._norm_stats["mean"].get(c, 0)
                        for c in self._feature_columns
                    ])
                    stds = np.array([
                        self._norm_stats["std"].get(c, 1)
                        for c in self._feature_columns
                    ])
                    stds[stds == 0] = 1
                    values = (values - means) / stds

                # Shape: (1, lookback, n_features)
                x_tensor = torch.FloatTensor(values).unsqueeze(0)

                with torch.no_grad():
                    logits = self._model(x_tensor)
                    proba = torch.softmax(logits, dim=1).numpy()[0]

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

                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        strategy_name=self.name,
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
                logger.exception("nn_strategy.signal_error", symbol=symbol)

        return signals

    async def train(self, historical_data: pd.DataFrame) -> TrainResult:
        """Train the LSTM model."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            # Prepare features (skip if already computed)
            if "rsi_14" in historical_data.columns:
                features_df = historical_data.copy()
            else:
                features_df = compute_features(historical_data)
            if "target" not in features_df.columns:
                features_df["target"] = create_target(features_df)
            features_df = features_df.dropna()

            if len(features_df) < 200:
                return TrainResult(
                    success=False, metrics={},
                    message=f"Insufficient data: {len(features_df)} rows",
                )

            feature_cols = select_feature_columns(features_df)
            feature_cols = [c for c in feature_cols if c != "target"]
            feature_cols = remove_low_variance(features_df, feature_cols)
            feature_cols = remove_highly_correlated(features_df, feature_cols)
            self._feature_columns = feature_cols

            # Split
            train_df, val_df, test_df = time_based_split(features_df, 0.70, 0.15)

            # Normalize
            X_train_norm, X_val_norm, X_test_norm, norm_stats = normalize_features(
                train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]
            )
            self._norm_stats = norm_stats

            # Create sequences
            X_tr_seq, y_tr_seq = _create_sequences(
                X_train_norm.values.astype(np.float32),
                train_df["target"].values.astype(np.int64),
                self._lookback,
            )
            X_va_seq, y_va_seq = _create_sequences(
                X_val_norm.values.astype(np.float32),
                val_df["target"].values.astype(np.int64),
                self._lookback,
            )
            X_te_seq, y_te_seq = _create_sequences(
                X_test_norm.values.astype(np.float32),
                test_df["target"].values.astype(np.int64),
                self._lookback,
            )

            if len(X_tr_seq) < 50:
                return TrainResult(
                    success=False, metrics={},
                    message="Insufficient sequences after windowing",
                )

            # Datasets
            train_ds = TensorDataset(
                torch.FloatTensor(X_tr_seq), torch.LongTensor(y_tr_seq)
            )
            val_ds = TensorDataset(
                torch.FloatTensor(X_va_seq), torch.LongTensor(y_va_seq)
            )

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            # Model
            n_features = len(feature_cols)
            model = _LSTMClassifier(
                n_features, self._hidden_size, self._num_layers, self._dropout
            )

            # Class weights
            class_counts = np.bincount(y_tr_seq, minlength=3).astype(np.float32)
            class_counts[class_counts == 0] = 1
            weights = len(y_tr_seq) / (3 * class_counts)
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

            # Training loop
            best_val_loss = float("inf")
            best_state = None
            epochs = 50
            patience_counter = 0
            patience = 10

            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        output = model(X_batch)
                        val_loss += criterion(output, y_batch).item()

                avg_val_loss = val_loss / max(len(val_loader), 1)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.debug("nn_strategy.early_stop", epoch=epoch)
                        break

            # Load best model
            if best_state:
                model.load_state_dict(best_state)

            # Evaluate
            model.eval()
            with torch.no_grad():
                val_preds = model(torch.FloatTensor(X_va_seq)).argmax(dim=1).numpy()
                test_preds = model(torch.FloatTensor(X_te_seq)).argmax(dim=1).numpy()

            val_acc = float(np.mean(val_preds == y_va_seq))
            test_acc = float(np.mean(test_preds == y_te_seq))

            # Save
            self._model = model
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_columns": feature_cols,
                    "norm_stats": norm_stats,
                    "config": {
                        "lookback": self._lookback,
                        "hidden_size": self._hidden_size,
                        "num_layers": self._num_layers,
                        "dropout": self._dropout,
                    },
                },
                self._model_path,
            )

            logger.info(
                "nn_strategy.trained",
                val_accuracy=round(val_acc, 4),
                test_accuracy=round(test_acc, 4),
                epochs=epoch + 1,
            )

            return TrainResult(
                success=True,
                metrics={
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "epochs_trained": epoch + 1,
                    "best_val_loss": float(best_val_loss),
                },
                message=f"LSTM trained: Val={val_acc:.4f}, Test={test_acc:.4f}",
            )

        except Exception as e:
            logger.exception("nn_strategy.train_error")
            return TrainResult(success=False, metrics={}, message=str(e))

    def get_confidence(self) -> float:
        return 0.5


class _LSTMClassifier:
    """LSTM classifier for 3-class time series classification.

    This is a plain class wrapping PyTorch modules. The actual torch.nn.Module
    is created internally so the import is deferred.
    """

    def __new__(cls, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        import torch.nn as nn

        class LSTMModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 3)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                out = self.dropout(last_hidden)
                out = self.relu(self.fc1(out))
                out = self.fc2(out)
                return out

        return LSTMModule()
