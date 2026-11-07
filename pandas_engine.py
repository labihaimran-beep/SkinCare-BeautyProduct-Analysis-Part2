"""Single-machine baseline: pandas read_csv + groupby."""
from __future__ import annotations
import pandas as pd
from .common import KEY, VALUES, USECOLS, tidy


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group an in-memory DataFrame into the (date, platform, campaign) cube."""
    g = df.groupby(KEY, sort=False)
    out = g[VALUES].sum()
    out["rows"] = g.size()
    return tidy(out.reset_index())


def run(path) -> pd.DataFrame:
    return aggregate(pd.read_csv(path, usecols=USECOLS))
