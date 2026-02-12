"""Walk-forward validation: rolling train/test to detect overfitting."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from app.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.backtest.metrics import calculate_metrics

logger = structlog.get_logger()


@dataclass
class WalkForwardConfig:
    train_days: int = 180      # 6 months training window
    test_days: int = 30        # 1 month test window
    step_days: int = 30        # Step forward by 1 month
    min_train_trades: int = 20
    initial_capital: float = 5000.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    max_position_pct: float = 20.0
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 2.0


@dataclass
class WindowResult:
    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: dict
    test_metrics: dict
    train_trades: int
    test_trades: int


@dataclass
class WalkForwardResult:
    windows: list[WindowResult] = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    degradation_pct: float = 0.0
    robustness_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "windows": [
                {
                    "window_num": w.window_num,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "train_metrics": w.train_metrics,
                    "test_metrics": w.test_metrics,
                    "train_trades": w.train_trades,
                    "test_trades": w.test_trades,
                }
                for w in self.windows
            ],
            "aggregate_metrics": self.aggregate_metrics,
            "degradation_pct": round(self.degradation_pct, 2),
            "robustness_score": round(self.robustness_score, 3),
            "num_windows": len(self.windows),
        }


class WalkForwardEngine:
    """Walk-forward analysis: train on window, test on next period, step forward."""

    async def run(
        self,
        config: WalkForwardConfig,
        strategy,
        data: pd.DataFrame,
        symbol: str,
    ) -> WalkForwardResult:
        """Run walk-forward analysis over the data.

        Args:
            config: Walk-forward configuration
            strategy: Strategy instance (must have train() method)
            data: Full OHLCV DataFrame
            symbol: Symbol being tested

        Returns:
            WalkForwardResult with per-window breakdown
        """
        if "timestamp" not in data.columns:
            raise ValueError("Data must have a 'timestamp' column")

        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)

        total_days = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).days
        if total_days < config.train_days + config.test_days:
            raise ValueError(
                f"Insufficient data: {total_days} days, need at least "
                f"{config.train_days + config.test_days}"
            )

        windows: list[WindowResult] = []
        all_test_pnls: list[float] = []
        all_test_equities: list[float] = []
        window_num = 0

        start_date = data["timestamp"].iloc[0]
        end_date = data["timestamp"].iloc[-1]

        current_train_start = start_date

        while True:
            train_end = current_train_start + pd.Timedelta(days=config.train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=config.test_days)

            if test_end > end_date:
                break

            # Split data
            train_mask = (data["timestamp"] >= current_train_start) & (
                data["timestamp"] < train_end
            )
            test_mask = (data["timestamp"] >= test_start) & (
                data["timestamp"] < test_end
            )

            train_data = data[train_mask].reset_index(drop=True)
            test_data = data[test_mask].reset_index(drop=True)

            if len(train_data) < 50 or len(test_data) < 10:
                current_train_start += pd.Timedelta(days=config.step_days)
                continue

            window_num += 1
            logger.info(
                "walk_forward.window",
                window=window_num,
                train_start=str(current_train_start.date()),
                test_start=str(test_start.date()),
                train_bars=len(train_data),
                test_bars=len(test_data),
            )

            # Train on training data
            train_metrics = {}
            try:
                train_result = await strategy.train(train_data)
                train_metrics = train_result.metrics if train_result.success else {}
            except Exception:
                logger.warning("walk_forward.train_error", window=window_num)

            # Backtest on test data
            bt_config = BacktestConfig(
                strategy=strategy,
                symbol=symbol,
                start_date=str(test_start.date()),
                end_date=str(test_end.date()),
                initial_capital=config.initial_capital,
                commission_pct=config.commission_pct,
                slippage_pct=config.slippage_pct,
                max_position_pct=config.max_position_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
                min_bars=min(50, len(test_data) - 1),
            )

            test_metrics = {}
            test_trades = 0
            try:
                bt_engine = BacktestEngine()
                bt_result = await bt_engine.run(bt_config, test_data)
                test_metrics = bt_result.metrics
                test_trades = len(bt_result.trades)
                all_test_pnls.extend([t.pnl for t in bt_result.trades])
                if bt_result.equity_curve:
                    all_test_equities.extend(
                        [e["equity"] for e in bt_result.equity_curve]
                    )
            except Exception:
                logger.warning("walk_forward.test_error", window=window_num, exc_info=True)

            # Backtest on training data for comparison
            train_bt_config = BacktestConfig(
                strategy=strategy,
                symbol=symbol,
                start_date=str(current_train_start.date()),
                end_date=str(train_end.date()),
                initial_capital=config.initial_capital,
                commission_pct=config.commission_pct,
                slippage_pct=config.slippage_pct,
                max_position_pct=config.max_position_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
                min_bars=min(50, len(train_data) - 1),
            )

            train_bt_metrics = {}
            train_trades = 0
            try:
                bt_engine2 = BacktestEngine()
                train_bt_result = await bt_engine2.run(train_bt_config, train_data)
                train_bt_metrics = train_bt_result.metrics
                train_trades = len(train_bt_result.trades)
            except Exception:
                logger.warning("walk_forward.train_bt_error", window=window_num, exc_info=True)

            windows.append(WindowResult(
                window_num=window_num,
                train_start=str(current_train_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_metrics=train_bt_metrics or train_metrics,
                test_metrics=test_metrics,
                train_trades=train_trades,
                test_trades=test_trades,
            ))

            current_train_start += pd.Timedelta(days=config.step_days)

        # Calculate aggregate metrics
        aggregate = {}
        if all_test_pnls:
            equity = [config.initial_capital]
            running = config.initial_capital
            for pnl in all_test_pnls:
                running += pnl
                equity.append(running)

            aggregate = calculate_metrics(
                trade_pnls=all_test_pnls,
                equity_curve=equity,
                initial_capital=config.initial_capital,
            )

        # Calculate degradation: avg train return vs avg test return
        train_returns = []
        test_returns = []
        for w in windows:
            tr = w.train_metrics.get("total_return_pct", 0.0)
            te = w.test_metrics.get("total_return_pct", 0.0)
            train_returns.append(tr)
            test_returns.append(te)

        avg_train = np.mean(train_returns) if train_returns else 0.0
        avg_test = np.mean(test_returns) if test_returns else 0.0

        degradation = 0.0
        if avg_train != 0:
            degradation = ((avg_train - avg_test) / abs(avg_train)) * 100

        # Robustness score: 0 = heavily overfit, 1 = robust
        if avg_train > 0 and avg_test > 0:
            robustness = min(avg_test / avg_train, 1.0)
        elif avg_train <= 0 and avg_test <= 0:
            robustness = 0.5  # Both negative, neutral
        else:
            robustness = 0.0 if avg_test < 0 else 1.0

        return WalkForwardResult(
            windows=windows,
            aggregate_metrics=aggregate,
            degradation_pct=degradation,
            robustness_score=robustness,
        )
