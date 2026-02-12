"""Monte Carlo simulation: trade P&L bootstrapping for outcome distribution."""

from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class MonteCarloConfig:
    num_simulations: int = 1000
    confidence_levels: list[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95]
    )


@dataclass
class MonteCarloResult:
    percentiles: dict[float, float] = field(default_factory=dict)
    max_drawdown_dist: dict[float, float] = field(default_factory=dict)
    ruin_probability: float = 0.0
    median_return_pct: float = 0.0
    equity_bands: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "percentiles": {
                str(k): round(v, 2) for k, v in self.percentiles.items()
            },
            "max_drawdown_dist": {
                str(k): round(v, 2) for k, v in self.max_drawdown_dist.items()
            },
            "ruin_probability": round(self.ruin_probability, 4),
            "median_return_pct": round(self.median_return_pct, 2),
            "equity_bands": self.equity_bands,
        }


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulation from historical trade P&Ls."""

    def run(
        self,
        trade_pnls: list[float],
        initial_capital: float,
        config: MonteCarloConfig | None = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation by resampling trade P&Ls.

        Args:
            trade_pnls: List of realized P&L from each trade
            initial_capital: Starting capital
            config: Simulation configuration

        Returns:
            MonteCarloResult with distributions and equity bands
        """
        if config is None:
            config = MonteCarloConfig()

        if not trade_pnls:
            return MonteCarloResult()

        pnls = np.array(trade_pnls, dtype=float)
        n_trades = len(pnls)
        n_sims = config.num_simulations

        # Generate all random samples at once for efficiency
        rng = np.random.default_rng(seed=42)
        # Each simulation: sample n_trades with replacement
        indices = rng.integers(0, n_trades, size=(n_sims, n_trades))
        sampled_pnls = pnls[indices]  # shape: (n_sims, n_trades)

        # Build equity curves
        cum_pnls = np.cumsum(sampled_pnls, axis=1)
        equity_curves = initial_capital + cum_pnls

        # Final equity for each simulation
        final_equities = equity_curves[:, -1]

        # Percentiles of final equity
        percentiles = {}
        for level in config.confidence_levels:
            percentiles[level] = float(np.percentile(final_equities, level * 100))

        # Max drawdown distribution
        max_drawdowns = np.zeros(n_sims)
        for i in range(n_sims):
            curve = equity_curves[i]
            peak = np.maximum.accumulate(curve)
            drawdowns = (peak - curve) / peak
            max_drawdowns[i] = float(np.max(drawdowns)) * 100

        dd_dist = {}
        for level in config.confidence_levels:
            dd_dist[level] = float(np.percentile(max_drawdowns, level * 100))

        # Ruin probability: % of simulations dropping below 50% of initial
        ruin_threshold = initial_capital * 0.5
        min_equities = np.min(equity_curves, axis=1)
        ruin_probability = float(np.mean(min_equities < ruin_threshold))

        # Median return %
        median_final = float(np.median(final_equities))
        median_return_pct = ((median_final - initial_capital) / initial_capital) * 100

        # Equity bands for plotting
        # Sample at evenly spaced trade indices
        n_points = min(50, n_trades)
        step = max(1, n_trades // n_points)
        band_indices = list(range(0, n_trades, step))
        if band_indices[-1] != n_trades - 1:
            band_indices.append(n_trades - 1)

        equity_bands = []
        for idx in band_indices:
            values = equity_curves[:, idx]
            band = {
                "trade_num": int(idx + 1),
                "p5": round(float(np.percentile(values, 5)), 2),
                "p25": round(float(np.percentile(values, 25)), 2),
                "p50": round(float(np.percentile(values, 50)), 2),
                "p75": round(float(np.percentile(values, 75)), 2),
                "p95": round(float(np.percentile(values, 95)), 2),
            }
            equity_bands.append(band)

        logger.info(
            "monte_carlo.completed",
            simulations=n_sims,
            trades=n_trades,
            median_return=round(median_return_pct, 2),
            ruin_probability=round(ruin_probability, 4),
        )

        return MonteCarloResult(
            percentiles=percentiles,
            max_drawdown_dist=dd_dist,
            ruin_probability=ruin_probability,
            median_return_pct=median_return_pct,
            equity_bands=equity_bands,
        )
