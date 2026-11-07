"""Per-platform response curves: revenue = a * spend**b (b < 1 means diminishing returns).

Fitted by ordinary least squares on log(daily revenue) ~ log(daily spend) across days. Spend and revenue are first
rescaled by one common factor so that the historical programme equals `baseline_monthly_budget` per `horizon_days`
(ROAS and the elasticity b are scale-free, so this only puts the curves on the scale of the Rs 1,000,000 question)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

B_MIN, B_MAX = 0.05, 0.95        # keep fitted elasticities in a sane concave range (flagged when clipped)


@dataclass
class Curve:
    platform: str
    a: float
    b: float
    b_raw: float
    b_se: float
    r2: float
    n_days: int
    mean_daily_spend: float
    mean_daily_revenue: float
    min_daily_spend: float
    max_daily_spend: float
    clipped: bool

    def revenue(self, daily_spend):
        return self.a * np.power(np.maximum(daily_spend, 1e-9), self.b)

    def marginal(self, daily_spend):
        return self.a * self.b * np.power(np.maximum(daily_spend, 1e-9), self.b - 1)


def daily_platform_table(cube: pd.DataFrame) -> pd.DataFrame:
    """Sum the (date, platform, campaign) cube to one row per (date, platform)."""
    d = cube.groupby(["date", "platform"], as_index=False)[["spend", "revenue", "conversions", "clicks", "impressions"]].sum()
    return d


def scale_factor(daily: pd.DataFrame, baseline_monthly_budget: float, horizon_days: int = 30) -> float:
    per_day = daily.groupby("date")["spend"].sum().mean()
    if per_day <= 0:
        raise ValueError("No spend in the data.")
    return baseline_monthly_budget / (per_day * horizon_days)


def fit_curve(platform: str, spend: np.ndarray, revenue: np.ndarray) -> Curve:
    ok = (spend > 0) & (revenue > 0)
    x, y = np.log(spend[ok]), np.log(revenue[ok])
    n = len(x)
    if n < 10 or np.ptp(x) < 1e-9:
        raise ValueError(f"{platform}: not enough varied daily observations to fit a response curve ({n}).")
    b_raw, c = np.polyfit(x, y, 1)
    resid = y - (c + b_raw * x)
    sxx = ((x - x.mean()) ** 2).sum()
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
    b = float(np.clip(b_raw, B_MIN, B_MAX))
    # keep the level consistent with the (possibly clipped) slope: match mean log revenue
    c = float(np.mean(y) - b * np.mean(x)) if b != b_raw else float(c)
    return Curve(platform, float(np.exp(c)), b, float(b_raw), se, r2, int(n), float(spend.mean()), float(revenue.mean()),
                 float(spend[ok].min()), float(spend[ok].max()), bool(b != b_raw))


def fit_all(daily: pd.DataFrame, factor: float = 1.0, platforms=None) -> dict:
    curves = {}
    for p, g in daily.groupby("platform"):
        if platforms and p not in platforms:
            continue
        curves[p] = fit_curve(p, g["spend"].to_numpy(float) * factor, g["revenue"].to_numpy(float) * factor)
    return curves
