import streamlit as st
from src.ui.common import banner, SIM_NOTE

st.set_page_config(page_title="Digital Advertising Budget Optimizer", page_icon="📈", layout="wide")
st.title("📈 Digital Advertising Budget Optimizer")
st.caption("Where should the next Rs 1,000,000 go? KPI analysis, a Big Data engine comparison (pandas vs MapReduce vs Spark) "
           "and a budget recommendation engine.")
banner()
df, cube = st.session_state.get("df"), st.session_state.get("cube")
c1, c2, c3 = st.columns(3)
c1.metric("Records loaded", f"{len(df):,}" if df is not None else "-")
c2.metric("Aggregated groups", f"{len(cube):,}" if cube is not None else "-")
c3.metric("Data", st.session_state.get("data_label", "none"))
if df is None:
    st.info("Start on **Data Simulation** in the sidebar: generate simulated campaign data (or upload your own CSV).")
st.markdown(f"""
**Pages**
1. **Data Simulation**: generate 10k to 1M simulated rows, or upload a CSV with the same columns.
2. **KPI Dashboard**: CTR, CPC, CPA, conversion rate, ROAS, revenue and campaign performance.
3. **Big Data Benchmark**: pandas vs MapReduce vs Spark results from `run_benchmark.py`.
4. **Budget Recommendation**: response curves, optimised split, uncertainty and validation.
5. **Reports**: CSV / Excel downloads.

_{SIM_NOTE}_
""")
