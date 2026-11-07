import streamlit as st, pandas as pd
from src.data_generation.simulator import COLUMNS
from src.ui.common import make_sim, set_data, banner, STRETCH
from src.engines.pandas_engine import aggregate  # noqa: F401  (ensures engine imports cleanly)

st.title("Data Simulation")
banner()
tab1, tab2 = st.tabs(["Generate simulated data", "Upload CSV"])
with tab1:
    n = st.select_slider("Number of records", [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], value=100_000,
                         format_func=lambda v: f"{v:,}")
    seed = st.number_input("Random seed", 0, 10_000, 42)
    st.caption("In-app generation is capped at 1,000,000 rows to protect memory. Use `run_benchmark.py` for 10,000,000.")
    if st.button("Generate", type="primary"):
        with st.spinner("Generating..."):
            make_sim(int(n), int(seed))
        st.success(f"Generated {n:,} simulated records.")
with tab2:
    st.write("Columns required: " + ", ".join(f"`{c}`" for c in COLUMNS))
    up = st.file_uploader("Upload a CSV", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            set_data(df, False, f"uploaded: {up.name}, {len(df):,} rows")
            st.success(f"Loaded {len(df):,} rows.")
        except Exception as e:
            st.error(f"Could not use this file: {e}")
df = st.session_state.get("df")
if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head(20), **STRETCH)
    st.write(df.describe().T)
