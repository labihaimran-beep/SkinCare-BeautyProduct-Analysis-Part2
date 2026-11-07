"""Advertising KPIs. All ratios return NaN (not inf) when the denominator is zero."""
from __future__ import annotations
import numpy as np
import pandas as pd

SUM_COLS = ["impressions", "clicks", "spend", "conversions", "revenue"]


def _div(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return a / b.where(b != 0)


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add CTR, CPC, CPM, CPA, conversion rate (conversions/clicks), ROAS and profit to aggregated or raw rows."""
    out = df.copy()
    out["ctr"] = _div(out["clicks"], out["impressions"])                 # clicks / impressions
    out["cpc"] = _div(out["spend"], out["clicks"])                       # Rs per click
    out["cpm"] = _div(out["spend"], out["impressions"]) * 1000           # Rs per 1,000 impressions
    out["cpa"] = _div(out["spend"], out["conversions"])                  # Rs per conversion
    out["conversion_rate"] = _div(out["conversions"], out["clicks"])     # conversions / clicks
    out["roas"] = _div(out["revenue"], out["spend"])                     # revenue per Rs spent
    out["profit"] = out["revenue"] - out["spend"]                        # gross revenue minus ad spend (no COGS)
    return out


def rollup(cube: pd.DataFrame, by) -> pd.DataFrame:
    """Sum an aggregated cube over `by` columns and compute KPIs on the sums (never average ratios)."""
    g = cube.groupby(by, observed=True)[[c for c in SUM_COLS + ["rows"] if c in cube.columns]].sum().reset_index()
    return add_kpis(g)
