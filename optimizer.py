"""Budget allocation: maximise total predicted revenue subject to sum(spend) = budget.

With concave response curves (b < 1) a greedy allocation in small equal increments, always giving the next increment
to the platform with the highest marginal revenue, is optimal up to the increment size. Constraints:
  * min_share / max_share of the budget per platform;
  * a trust region around the historical mix (default 0.5x to 2x of each platform's current share), because the curves
    are only supported by spend levels that were actually observed."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .response_curves import (Curve, daily_platform_table, scale_factor, fit_all)

HORIZON_DAYS = 30


def bounds(current_share: dict, budget: float, min_share=0.05, max_share=0.50, trust=(0.5, 2.0)):
    lo, hi = {}, {}
    for p, cs in current_share.items():
        lo[p] = max(min_share, trust[0] * cs) * budget
        hi[p] = min(max_share, trust[1] * cs) * budget
    if sum(lo.values()) > budget + 1e-6 or sum(hi.values()) < budget - 1e-6:
        raise ValueError("Constraints are infeasible for this budget: widen min/max share or the trust region.")
    return lo, hi


def monthly_revenue(curve: Curve, monthly_spend: float, horizon_days: int = HORIZON_DAYS) -> float:
    return float(horizon_days * curve.revenue(monthly_spend / horizon_days))


def allocate(curves: dict, budget: float, lo: dict, hi: dict, steps: int = 1000, horizon_days: int = HORIZON_DAYS) -> dict:
    """Greedy equal-increment allocation. Returns {platform: monthly spend} summing exactly to budget."""
    s = dict(lo)
    remaining = budget - sum(lo.values())
    inc = budget / steps
    while remaining > 1e-9:
        step = min(inc, remaining)
        best, best_gain = None, -1.0
        for p, c in curves.items():
            room = hi[p] - s[p]
            if room <= 1e-9:
                continue
            d = min(step, room)
            gain = monthly_revenue(c, s[p] + d, horizon_days) - monthly_revenue(c, s[p], horizon_days)
            gain /= d                                   # per rupee, so partial steps compare fairly
            if gain > best_gain:
                best, best_gain, best_d = p, gain, d
        if best is None:
            raise ValueError("No platform has room left; constraints too tight.")
        s[best] += best_d; remaining -= best_d
    return s


def total_revenue(curves: dict, spend: dict, horizon_days: int = HORIZON_DAYS) -> float:
    return sum(monthly_revenue(curves[p], spend[p], horizon_days) for p in curves)


def marginal_roas(curve: Curve, monthly_spend: float, horizon_days: int = HORIZON_DAYS) -> float:
    return float(curve.marginal(monthly_spend / horizon_days))     # d(revenue)/d(spend) is scale-free in the horizon


def recommend(cube: pd.DataFrame, budget: float = 1_000_000, min_share=0.05, max_share=0.50, trust=(0.5, 2.0),
              horizon_days: int = HORIZON_DAYS, baseline_monthly_budget: float | None = None, steps: int = 1000) -> dict:
    """Full pipeline from an aggregated cube to a recommended split. Returns dict with table, curves and comparisons."""
    daily = daily_platform_table(cube)
    factor = scale_factor(daily, baseline_monthly_budget or budget, horizon_days)
    curves = fit_all(daily, factor)
    spend_by_p = daily.groupby("platform")["spend"].sum()
    current = (spend_by_p / spend_by_p.sum()).to_dict()
    lo, hi = bounds(current, budget, min_share, max_share, trust)
    rec = allocate(curves, budget, lo, hi, steps, horizon_days)
    cur_alloc = {p: current[p] * budget for p in curves}
    eq_alloc = {p: budget / len(curves) for p in curves}
    roas_w = {p: curves[p].mean_daily_revenue / curves[p].mean_daily_spend for p in curves}
    tot = sum(roas_w.values())
    prop_alloc = {p: budget * roas_w[p] / tot for p in curves}
    rows = []
    for p, c in curves.items():
        rev = monthly_revenue(c, rec[p], horizon_days)
        rows.append(dict(platform=p, current_share=current[p], recommended_share=rec[p] / budget, recommended_budget=rec[p],
                         change_pts=100 * (rec[p] / budget - current[p]), expected_revenue=rev, expected_roas=rev / rec[p],
                         marginal_roas=marginal_roas(c, rec[p], horizon_days), elasticity=c.b, r2=c.r2,
                         at_bound=("max" if rec[p] >= hi[p] - 1e-6 else "min" if rec[p] <= lo[p] + 1e-6 else "")))
    table = pd.DataFrame(rows).sort_values("recommended_share", ascending=False).reset_index(drop=True)
    def feasible(a):   # does this allocation respect the same bounds as the optimiser?
        return all(lo[p] - 1e-6 <= a[p] <= hi[p] + 1e-6 for p in a)

    comparison = pd.DataFrame([
        dict(strategy=name, expected_revenue=total_revenue(curves, a, horizon_days),
             expected_roas=total_revenue(curves, a, horizon_days) / budget, within_constraints=feasible(a))
        for name, a in (("Current mix", cur_alloc), ("Equal split", eq_alloc), ("Proportional to ROAS", prop_alloc),
                        ("Optimised (recommended)", rec))])
    return dict(table=table, comparison=comparison, curves=curves, budget=budget, factor=factor, daily=daily,
                allocations=dict(current=cur_alloc, equal=eq_alloc, proportional=prop_alloc, recommended=rec),
                bounds=(lo, hi), horizon_days=horizon_days, params=dict(min_share=min_share, max_share=max_share, trust=trust))


def format_split(table: pd.DataFrame, digits: int = 0) -> str:
    """e.g. 'Google: 35%, Facebook: 25% ...' (rounded shares forced to sum to 100)."""
    shares = (table.recommended_share * 100).round(digits).tolist()
    diff = round(100 - sum(shares), digits)
    if diff:
        shares[int(np.argmax(table.recommended_share.to_numpy()))] += diff
    return "\n".join(f"{p}: {s:g}%" for p, s in zip(table.platform, shares))


def sensitivity(cube: pd.DataFrame, budget: float = 1_000_000, settings=None) -> pd.DataFrame:
    """Recommended shares under different constraint settings (how much does the answer depend on them?)."""
    settings = settings or [("Default (max 50%, trust 0.5x-2x)", dict()),
                            ("No platform cap (max 100%)", dict(max_share=1.0)),
                            ("Tighter cap (max 40%)", dict(max_share=0.40)),
                            ("Narrow trust region 0.7x-1.4x", dict(trust=(0.7, 1.4))),
                            ("Wide trust region 0.25x-4x, no cap", dict(trust=(0.25, 4.0), max_share=1.0))]
    rows = []
    for name, kw in settings:
        r = recommend(cube, budget, **kw)
        row = {"setting": name, **{p: s for p, s in zip(r["table"].platform, r["table"].recommended_share)},
               "expected_revenue": float(r["comparison"].iloc[-1].expected_revenue)}
        rows.append(row)
    return pd.DataFrame(rows)
