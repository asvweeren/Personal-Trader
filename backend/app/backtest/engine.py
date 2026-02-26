"""Event-driven backtesting engine using the same Strategy interface as live trading."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import structlog

from app.backtest.metrics import calculate_benchmark_comparison, calculate_metrics
from app.backtest.simulator import (
    FillStatus,
    MarketSimulator,
    SimulatedPosition,
)
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    strategy: Strategy
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 5000.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    spread_pct: float = 0.02
    max_position_pct: float = 20.0
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 2.0
    enable_eod_close: bool = True
    trailing_stop_tiers: str = "1.0:0.5,2.0:0.75,3.0:1.0,5.0:1.5"
    min_bars: int = 100


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    entry_time: str
    exit_time: str
    exit_reason: str = "signal"
    bars_held: int = 0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    config: dict
    metrics: dict
    benchmark_metrics: dict
    trades: list[BacktestTrade]
    equity_curve: list[dict]
    total_bars: int
    duration_days: int


class BacktestEngine:
    """Event-driven backtester using the same Strategy interface as live trading.

    For each bar in the historical data:
    1. Check stop-losses on open positions
    2. Check take-profits on open positions
    3. Check EOD close (last bar of trading day)
    4. Update progressive trailing stops
    5. Create a MarketSnapshot from historical data up to current bar
    6. Call strategy.generate_signals()
    7. Execute signals via MarketSimulator
    8. Record equity
    """

    def __init__(self):
        self._simulator: MarketSimulator | None = None

    @staticmethod
    def _parse_trailing_tiers(tiers_str: str) -> list[tuple[float, float]]:
        """Parse trailing stop tiers string into sorted list."""
        tiers = []
        for pair in tiers_str.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            gain_s, trail_s = pair.split(":", 1)
            tiers.append((float(gain_s), float(trail_s)))
        tiers.sort(key=lambda t: t[0], reverse=True)
        return tiers

    @staticmethod
    def _is_last_bar_of_day(data: pd.DataFrame, current_idx: int) -> bool:
        """Check if the current bar is the last bar of its trading day."""
        if "timestamp" not in data.columns:
            return False
        try:
            current_ts = pd.Timestamp(data.iloc[current_idx]["timestamp"])
            if current_idx + 1 >= len(data):
                return True  # Last bar in dataset
            next_ts = pd.Timestamp(data.iloc[current_idx + 1]["timestamp"])
            return current_ts.date() != next_ts.date()
        except Exception:
            return False

    def _close_position(
        self,
        symbol: str,
        pos: SimulatedPosition,
        fill_price: float,
        bar_time: str,
        bar_idx: int,
        exit_reason: str,
        entry_bar_idx: dict[str, int],
        bar_volume: int,
    ) -> tuple[BacktestTrade, float, float] | None:
        """Close a position and return (trade, cash_received, commission) or None."""
        fill = self._simulator.simulate_fill(fill_price, "SELL", pos.quantity, bar_volume)
        if fill.status != FillStatus.FILLED:
            return None
        pnl = (fill.fill_price - pos.avg_cost) * pos.quantity - fill.commission
        trade = BacktestTrade(
            symbol=symbol,
            side="SELL",
            quantity=pos.quantity,
            entry_price=pos.avg_cost,
            exit_price=fill.fill_price,
            pnl=round(pnl, 2),
            commission=fill.commission,
            entry_time=str(entry_bar_idx.get(symbol, "")),
            exit_time=bar_time,
            exit_reason=exit_reason,
            bars_held=bar_idx - entry_bar_idx.get(symbol, bar_idx),
        )
        cash_received = fill.fill_price * pos.quantity - fill.commission
        return trade, cash_received, fill.commission

    async def run(self, config: BacktestConfig, data: pd.DataFrame) -> BacktestResult:
        """Run a full backtest on historical OHLCV data."""
        if len(data) < config.min_bars:
            raise ValueError(
                f"Insufficient data: {len(data)} bars, need at least {config.min_bars}"
            )

        self._simulator = MarketSimulator(
            commission_pct=config.commission_pct,
            slippage_pct=config.slippage_pct,
            spread_pct=config.spread_pct,
        )

        trailing_tiers = self._parse_trailing_tiers(config.trailing_stop_tiers)

        cash = config.initial_capital
        positions: dict[str, SimulatedPosition] = {}
        trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        trade_pnls: list[float] = []
        total_commission = 0.0

        # Track entry bar index for bars_held calculation
        entry_bar_idx: dict[str, int] = {}

        logger.info(
            "backtest.started",
            strategy=config.strategy.name,
            symbol=config.symbol,
            bars=len(data),
            take_profit_pct=config.take_profit_pct,
            eod_close=config.enable_eod_close,
        )

        # We need a lookback window for the strategy to work on
        lookback = config.min_bars

        for i in range(lookback, len(data)):
            bar = data.iloc[i]
            bar_time = str(bar.get("timestamp", i))
            bar_close = float(bar["close"])
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_volume = int(bar.get("volume", 10000))

            # 1. Check stop-losses
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                if pos.stop_loss is not None:
                    if self._simulator.simulate_stop_check(
                        pos.stop_loss, bar_low, bar_high, side="SELL"
                    ):
                        fill = self._simulator.simulate_stop_fill(
                            pos.stop_loss, bar_open, bar_low, pos.quantity
                        )
                        if fill.status == FillStatus.FILLED:
                            pnl = (fill.fill_price - pos.avg_cost) * pos.quantity - fill.commission
                            trades.append(BacktestTrade(
                                symbol=symbol,
                                side="SELL",
                                quantity=pos.quantity,
                                entry_price=pos.avg_cost,
                                exit_price=fill.fill_price,
                                pnl=round(pnl, 2),
                                commission=fill.commission,
                                entry_time=str(entry_bar_idx.get(symbol, "")),
                                exit_time=bar_time,
                                exit_reason="stop_loss",
                                bars_held=i - entry_bar_idx.get(symbol, i),
                            ))
                            trade_pnls.append(pnl)
                            cash += fill.fill_price * pos.quantity - fill.commission
                            total_commission += fill.commission
                            del positions[symbol]
                            entry_bar_idx.pop(symbol, None)

            # 2. Check take-profits
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                if pos.take_profit is not None:
                    if self._simulator.simulate_take_profit_check(
                        pos.take_profit, bar_high
                    ):
                        fill = self._simulator.simulate_take_profit_fill(
                            pos.take_profit, bar_open, bar_high, pos.quantity
                        )
                        if fill.status == FillStatus.FILLED:
                            pnl = (fill.fill_price - pos.avg_cost) * pos.quantity - fill.commission
                            trades.append(BacktestTrade(
                                symbol=symbol,
                                side="SELL",
                                quantity=pos.quantity,
                                entry_price=pos.avg_cost,
                                exit_price=fill.fill_price,
                                pnl=round(pnl, 2),
                                commission=fill.commission,
                                entry_time=str(entry_bar_idx.get(symbol, "")),
                                exit_time=bar_time,
                                exit_reason="take_profit",
                                bars_held=i - entry_bar_idx.get(symbol, i),
                            ))
                            trade_pnls.append(pnl)
                            cash += fill.fill_price * pos.quantity - fill.commission
                            total_commission += fill.commission
                            del positions[symbol]
                            entry_bar_idx.pop(symbol, None)

            # 3. EOD close — force-sell all positions on last bar of trading day
            if config.enable_eod_close and positions:
                if self._is_last_bar_of_day(data, i):
                    for symbol in list(positions.keys()):
                        pos = positions[symbol]
                        result = self._close_position(
                            symbol, pos, bar_close, bar_time, i,
                            "eod_close", entry_bar_idx, bar_volume,
                        )
                        if result:
                            trade, cash_recv, comm = result
                            trades.append(trade)
                            trade_pnls.append(trade.pnl)
                            cash += cash_recv
                            total_commission += comm
                            del positions[symbol]
                            entry_bar_idx.pop(symbol, None)

            # 4. Update progressive trailing stops on remaining positions
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                if pos.avg_cost is None or pos.stop_loss is None:
                    continue
                gain_pct = ((bar_close - pos.avg_cost) / pos.avg_cost) * 100
                # Find applicable tier (highest threshold exceeded)
                for threshold, trail in trailing_tiers:
                    if gain_pct >= threshold:
                        new_stop = round(bar_close * (1 - trail / 100), 2)
                        if new_stop > pos.stop_loss:
                            pos.stop_loss = new_stop
                        break

            # 5. Create market snapshot from historical window
            window = data.iloc[max(0, i - lookback):i + 1].copy()
            snapshot = MarketSnapshot(
                timestamp=datetime.now(UTC),
                prices={config.symbol: bar_close},
                ohlcv={config.symbol: window},
                features={},
            )

            # 6. Generate signals
            try:
                signals = await config.strategy.generate_signals(snapshot)
            except Exception:
                logger.debug("backtest.strategy_error", bar=i)
                signals = []

            # 7. Execute signals
            for signal in signals:
                if signal.symbol != config.symbol:
                    continue

                if signal.action == SignalAction.BUY and config.symbol not in positions:
                    # Skip buying on the last bar of the day (would be closed immediately)
                    if config.enable_eod_close and self._is_last_bar_of_day(data, i):
                        continue

                    # Calculate position size
                    max_value = cash * (config.max_position_pct / 100)
                    quantity = int(max_value / bar_close)
                    if quantity <= 0:
                        continue

                    # Adjust by confidence
                    quantity = max(1, int(quantity * signal.confidence))

                    fill = self._simulator.simulate_fill(
                        bar_close, "BUY", quantity, bar_volume
                    )
                    if fill.status != FillStatus.FILLED:
                        continue

                    cost = fill.fill_price * fill.filled_quantity + fill.commission
                    if cost > cash:
                        # Reduce quantity to fit
                        quantity = int((cash - fill.commission) / fill.fill_price)
                        if quantity <= 0:
                            continue
                        fill = self._simulator.simulate_fill(
                            bar_close, "BUY", quantity, bar_volume
                        )
                        cost = fill.fill_price * fill.filled_quantity + fill.commission

                    cash -= cost
                    total_commission += fill.commission

                    stop_loss = round(
                        fill.fill_price * (1 - config.stop_loss_pct / 100), 2
                    )
                    take_profit = round(
                        fill.fill_price * (1 + config.take_profit_pct / 100), 2
                    )
                    positions[config.symbol] = SimulatedPosition(
                        symbol=config.symbol,
                        quantity=fill.filled_quantity,
                        avg_cost=fill.fill_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    entry_bar_idx[config.symbol] = i

                elif signal.action == SignalAction.SELL and config.symbol in positions:
                    pos = positions[config.symbol]
                    fill = self._simulator.simulate_fill(
                        bar_close, "SELL", pos.quantity, bar_volume
                    )
                    if fill.status != FillStatus.FILLED:
                        continue

                    pnl = (fill.fill_price - pos.avg_cost) * pos.quantity - fill.commission
                    trades.append(BacktestTrade(
                        symbol=config.symbol,
                        side="SELL",
                        quantity=pos.quantity,
                        entry_price=pos.avg_cost,
                        exit_price=fill.fill_price,
                        pnl=round(pnl, 2),
                        commission=fill.commission,
                        entry_time=str(entry_bar_idx.get(config.symbol, "")),
                        exit_time=bar_time,
                        exit_reason="signal",
                        bars_held=i - entry_bar_idx.get(config.symbol, i),
                    ))
                    trade_pnls.append(pnl)
                    cash += fill.fill_price * pos.quantity - fill.commission
                    total_commission += fill.commission
                    del positions[config.symbol]
                    entry_bar_idx.pop(config.symbol, None)

            # 8. Record equity
            positions_value = sum(
                float(data.iloc[i]["close"]) * pos.quantity
                for pos in positions.values()
            )
            total_equity = cash + positions_value
            equity_curve.append({
                "bar": i,
                "time": bar_time,
                "equity": round(total_equity, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
            })

        # Close any remaining open positions at last bar price
        last_bar = data.iloc[-1]
        last_close = float(last_bar["close"])
        for symbol, pos in list(positions.items()):
            fill = self._simulator.simulate_fill(
                last_close, "SELL", pos.quantity, int(last_bar.get("volume", 10000))
            )
            if fill.status == FillStatus.FILLED:
                pnl = (fill.fill_price - pos.avg_cost) * pos.quantity - fill.commission
                trades.append(BacktestTrade(
                    symbol=symbol,
                    side="SELL",
                    quantity=pos.quantity,
                    entry_price=pos.avg_cost,
                    exit_price=fill.fill_price,
                    pnl=round(pnl, 2),
                    commission=fill.commission,
                    entry_time=str(entry_bar_idx.get(symbol, "")),
                    exit_time=str(last_bar.get("timestamp", len(data) - 1)),
                    exit_reason="end_of_backtest",
                    bars_held=len(data) - 1 - entry_bar_idx.get(symbol, len(data) - 1),
                ))
                trade_pnls.append(pnl)
                cash += fill.fill_price * pos.quantity - fill.commission
                total_commission += fill.commission

        # Calculate metrics
        equity_values = [config.initial_capital] + [e["equity"] for e in equity_curve]
        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_values,
            initial_capital=config.initial_capital,
        )
        metrics["total_commission"] = round(total_commission, 2)

        # Buy-and-hold benchmark
        benchmark_metrics = calculate_benchmark_comparison(
            equity_curve=equity_values,
            price_series=data["close"].values[lookback:],
            initial_capital=config.initial_capital,
        )

        # Duration
        duration_days = 0
        if "timestamp" in data.columns:
            try:
                ts = pd.to_datetime(data["timestamp"])
                duration_days = (ts.iloc[-1] - ts.iloc[lookback]).days
            except Exception:
                pass

        logger.info(
            "backtest.completed",
            strategy=config.strategy.name,
            trades=len(trades),
            final_equity=equity_values[-1] if equity_values else config.initial_capital,
        )

        return BacktestResult(
            strategy_name=config.strategy.name,
            symbol=config.symbol,
            config={
                "initial_capital": config.initial_capital,
                "commission_pct": config.commission_pct,
                "slippage_pct": config.slippage_pct,
                "max_position_pct": config.max_position_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "take_profit_pct": config.take_profit_pct,
                "enable_eod_close": config.enable_eod_close,
                "trailing_stop_tiers": config.trailing_stop_tiers,
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            trades=trades,
            equity_curve=equity_curve,
            total_bars=len(data) - lookback,
            duration_days=duration_days,
        )
