"""Shared Streamlit helpers (session state, empty states, chart-width compatibility)."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from src.data_generation.simulator import generate, COLUMNS
from src.engines.pandas_engine import aggregate

SIM_NOTE = ("All data on these pages is SIMULATED by `src/data_generation/simulator.py`. Results describe the simulator "
            "(a teaching model), not real campaigns, and must not be used to spend real money.")


def _stretch() -> dict:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


STRETCH = _stretch()


def require(key: str, message: str):
    if st.session_state.get(key) is None:
        st.info(message)
        st.stop()


def set_data(df: pd.DataFrame, simulated: bool, label: str):
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    st.session_state.update(df=df, cube=aggregate(df), simulated=simulated, data_label=label, rec=None)


def banner():
    if st.session_state.get("simulated"):
        st.warning(SIM_NOTE)


def make_sim(n: int, seed: int):
    set_data(generate(n, seed), True, f"simulated, {n:,} rows, seed {seed}")
