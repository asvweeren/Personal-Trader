import { useState, useEffect, useCallback } from "react";
import { api } from "../api/client";
import type { Portfolio, Performance, RiskMetricsData } from "../types";

export function usePortfolio(refreshInterval = 10000) {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [risk, setRisk] = useState<RiskMetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [p, perf, r] = await Promise.all([
        api.getPortfolio(),
        api.getPerformance(),
        api.getRiskMetrics(),
      ]);
      setPortfolio(p);
      setPerformance(perf);
      setRisk(r);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, refreshInterval);
    return () => clearInterval(interval);
  }, [refresh, refreshInterval]);

  return { portfolio, performance, risk, loading, error, refresh };
}
