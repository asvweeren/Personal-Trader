"""Value at Risk (VaR) and stress testing for portfolio risk assessment."""

import asyncio
from dataclasses import dataclass, field

import numpy as np
import structlog

from app.broker.base import Portfolio
from app.data.market_data import MarketDataService

logger = structlog.get_logger()


@dataclass
class VaRResult:
    var_95: float = 0.0       # 95% VaR in dollars
    var_99: float = 0.0       # 99% VaR in dollars
    cvar_95: float = 0.0      # Conditional VaR (Expected Shortfall)
    stress_scenarios: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "cvar_95": round(self.cvar_95, 2),
            "stress_scenarios": {
                k: round(v, 2) for k, v in self.stress_scenarios.items()
            },
        }


# Predefined stress scenarios: (name, market_shock_pct)
STRESS_SCENARIOS = {
    "Market Crash (-10%)": -0.10,
    "Sharp Correction (-5%)": -0.05,
    "Sector Rotation (-3%)": -0.03,
    "Rate Hike Shock (-7%)": -0.07,
    "Flash Crash (-15%)": -0.15,
    "Mild Pullback (-2%)": -0.02,
}


class VaRCalculator:
    """Calculate Value at Risk using historical simulation."""

    async def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        market_data: MarketDataService,
        lookback_days: int = 60,
        confidence: float = 0.95,
    ) -> VaRResult:
        """Calculate VaR for the current portfolio using historical returns."""
        if not portfolio.positions:
            return VaRResult()

        total_value = portfolio.account_summary.total_value
        if total_value <= 0:
            return VaRResult()

        from app.dependencies import should_skip_eu

        # Use the same cache key the trading engine uses ("1 Y", "1 day") so we
        # share its in-memory cache instead of issuing parallel fresh IBKR requests
        # on every dashboard poll (which hits IBKR's 60-req/10-min pacing limit
        # and triggers Error 366 cancellations).
        eligible = [p for p in portfolio.positions if not should_skip_eu(p.symbol)]
        position_weights = {
            p.symbol: (p.market_value / total_value if total_value > 0 else 0.0)
            for p in eligible
        }

        async def _fetch(symbol: str):
            return symbol, await market_data.get_historical_data(
                symbol, duration="1 Y", bar_size="1 day"
            )

        results = await asyncio.gather(
            *[_fetch(p.symbol) for p in eligible],
            return_exceptions=True,
        )

        # Per-symbol return series, trimmed to the lookback window
        per_symbol_returns: dict[str, np.ndarray] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.debug("var.data_error", error=str(result))
                continue
            symbol, df = result
            if df is None or len(df) < 5:
                continue
            closes = df["close"].values.astype(float)[-(lookback_days + 1):]
            if len(closes) < 5:
                continue
            per_symbol_returns[symbol] = np.diff(closes) / closes[:-1]

        if not per_symbol_returns:
            return self._estimate_var(total_value, portfolio)

        # Align series on the shortest length (right-align: most recent days)
        min_len = min(len(r) for r in per_symbol_returns.values())
        portfolio_returns = np.zeros(min_len)
        for symbol, returns in per_symbol_returns.items():
            portfolio_returns += position_weights.get(symbol, 0.0) * returns[-min_len:]

        returns_array = portfolio_returns[np.isfinite(portfolio_returns)]
        if len(returns_array) < 5:
            return self._estimate_var(total_value, portfolio)

        # Historical VaR
        var_95_pct = float(np.percentile(returns_array, 5))  # 5th percentile = 95% VaR
        var_99_pct = float(np.percentile(returns_array, 1))

        # CVaR (Expected Shortfall): average of returns below VaR
        tail_returns = returns_array[returns_array <= var_95_pct]
        cvar_95_pct = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_95_pct

        var_95 = abs(var_95_pct * total_value)
        var_99 = abs(var_99_pct * total_value)
        cvar_95 = abs(cvar_95_pct * total_value)

        # Stress scenarios
        stress = {}
        for name, shock in STRESS_SCENARIOS.items():
            # Simple: apply shock to total positions value
            positions_value = sum(p.market_value for p in portfolio.positions)
            impact = shock * positions_value
            stress[name] = impact

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            stress_scenarios=stress,
        )

    def _estimate_var(self, total_value: float, portfolio: Portfolio) -> VaRResult:
        """Fallback VaR estimation when historical data is insufficient."""
        # Assume 2% daily volatility as conservative estimate
        daily_vol = 0.02
        positions_value = sum(p.market_value for p in portfolio.positions)

        var_95 = positions_value * daily_vol * 1.645  # 95% z-score
        var_99 = positions_value * daily_vol * 2.326  # 99% z-score
        cvar_95 = var_95 * 1.2  # Approximate CVaR

        stress = {}
        for name, shock in STRESS_SCENARIOS.items():
            stress[name] = shock * positions_value

        return VaRResult(
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            cvar_95=round(cvar_95, 2),
            stress_scenarios=stress,
        )
