import streamlit as st, pandas as pd, plotly.express as px
from pathlib import Path
from src.ui.common import banner, STRETCH

st.title("Big Data Benchmark: pandas vs MapReduce vs Spark")
st.caption("Same job on every engine: read the CSV, group by (date, platform, campaign), sum five measures. "
           "Each run is a separate process; times include reading the file.")
f = next((q for q in (Path("outputs/benchmark_results.csv"), Path("results/benchmark_results.csv")) if q.exists()), Path("outputs/benchmark_results.csv"))
if not f.exists():
    st.info("No results yet. Run `python run_benchmark.py` in a terminal (10,000,000 rows takes a few minutes).")
    st.stop()
df = pd.read_csv(f)
mach = f.with_name("benchmark_machine.txt")
if mach.exists():
    st.caption("Machine: " + mach.read_text().strip())
ok = df[df.status == "ok"]
st.dataframe(df, hide_index=True, **STRETCH)
if not ok.empty:
    st.plotly_chart(px.line(ok, x="rows", y="seconds_median", color="engine", markers=True, log_x=True, log_y=True,
                            title="Median seconds vs records (log-log)"), **STRETCH)
    st.plotly_chart(px.line(ok, x="rows", y="peak_mb", color="engine", markers=True, log_x=True, log_y=True,
                            title="Peak memory (MB)"), **STRETCH)
if (df.status != "ok").any():
    st.warning("Some engines were skipped or failed (see the status column). Nothing is estimated for them.")
if not ok.empty and not ok.agrees_with_first.all():
    st.error("Engines disagree on some scale: the comparison is invalid there.")
