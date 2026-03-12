"""Value at Risk (VaR) and stress testing for portfolio risk assessment."""

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

        # Collect historical returns for each position
        portfolio_returns = []
        position_weights: dict[str, float] = {}

        from app.dependencies import should_skip_eu

        for pos in portfolio.positions:
            # Skip EU symbols when EU trading is disabled (no IBKR data subscription)
            if should_skip_eu(pos.symbol):
                continue

            weight = pos.market_value / total_value if total_value > 0 else 0.0
            position_weights[pos.symbol] = weight

            try:
                df = await market_data.get_historical_data(
                    pos.symbol,
                    duration=f"{lookback_days} D",
                    bar_size="1 day",
                )
                if df is not None and len(df) >= 5:
                    closes = df["close"].values.astype(float)
                    returns = np.diff(closes) / closes[:-1]
                    if len(portfolio_returns) == 0:
                        portfolio_returns = [
                            np.zeros(len(returns))
                            for _ in range(len(returns))
                        ]
                    # Weight the returns
                    min_len = min(len(returns), len(portfolio_returns))
                    for j in range(min_len):
                        portfolio_returns[j] = (
                            portfolio_returns[j] + weight * returns[-(min_len - j)]
                            if isinstance(portfolio_returns[j], np.ndarray)
                            else weight * returns[-(min_len - j)]
                        )
            except Exception:
                logger.debug("var.data_error", symbol=pos.symbol, exc_info=True)
                continue

        if not portfolio_returns or all(
            isinstance(r, np.ndarray) and np.all(r == 0) for r in portfolio_returns
        ):
            # Fallback: use a simple estimation
            return self._estimate_var(total_value, portfolio)

        # Convert to simple array
        try:
            returns_array = np.array([
                float(r) if not isinstance(r, np.ndarray) else float(np.sum(r))
                for r in portfolio_returns
            ])
            returns_array = returns_array[np.isfinite(returns_array)]
        except Exception:
            return self._estimate_var(total_value, portfolio)

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
