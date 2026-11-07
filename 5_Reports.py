import streamlit as st, io, pandas as pd
from src.reporting.export import csv_bytes, excel_bytes
from src.metrics.kpis import rollup
from src.ui.common import require, banner

st.title("Reports & Downloads")
banner()
require("cube", "Load data first on **Data Simulation**.")
cube, rec = st.session_state["cube"], st.session_state.get("rec")
items = {"aggregated_cube": cube, "platform_kpis": rollup(cube, "platform"), "campaign_kpis": rollup(cube, "campaign"),
         "platform_campaign_kpis": rollup(cube, ["platform", "campaign"])}
if rec is not None:
    items.update(recommendation=rec["table"], strategy_comparison=rec["comparison"])
for name, d in items.items():
    st.download_button(f"Download {name}.csv ({len(d):,} rows)", csv_bytes(d), f"{name}.csv", "text/csv", key=name)
if st.button("Build Excel workbook"):
    st.session_state["xlsx"] = excel_bytes({k: v for k, v in items.items() if k != "aggregated_cube"})
if st.session_state.get("xlsx"):
    st.download_button("Download workbook.xlsx", st.session_state["xlsx"], "ad_budget_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
if rec is None:
    st.caption("Open **Budget Recommendation** first to include the recommendation in the exports.")
