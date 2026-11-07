import streamlit as st, plotly.express as px
from src.metrics.kpis import rollup
from src.ui.common import require, banner, STRETCH

st.title("KPI Dashboard")
banner()
require("cube", "Load data first on **Data Simulation**.")
cube, df = st.session_state["cube"], st.session_state["df"]
tot = rollup(cube.assign(g="all"), "g").iloc[0]
m = st.columns(6)
m[0].metric("Spend (Rs)", f"{tot.spend:,.0f}"); m[1].metric("Revenue (Rs)", f"{tot.revenue:,.0f}")
m[2].metric("ROAS", f"{tot.roas:.2f}"); m[3].metric("CTR", f"{100 * tot.ctr:.2f}%")
m[4].metric("CPA (Rs)", f"{tot.cpa:,.0f}"); m[5].metric("Conv. rate", f"{100 * tot.conversion_rate:.2f}%")
st.caption("Ratios are computed on summed totals (never averages of row ratios). Conversion rate = conversions / clicks. "
           "Profit here is revenue minus ad spend; no product cost is modelled.")
plat = rollup(cube, "platform").sort_values("roas", ascending=False)
st.subheader("Platform performance")
st.dataframe(plat[["platform", "impressions", "clicks", "spend", "conversions", "revenue", "ctr", "cpc", "cpa",
                   "conversion_rate", "roas", "profit"]], hide_index=True, **STRETCH)
a, b = st.columns(2)
a.plotly_chart(px.bar(plat, x="platform", y="roas", title="ROAS by platform"), **STRETCH)
b.plotly_chart(px.bar(plat, x="platform", y="cpa", title="Cost per acquisition (Rs)"), **STRETCH)
st.subheader("Campaign performance")
camp = rollup(cube, ["campaign"]).sort_values("roas", ascending=False)
st.dataframe(camp[["campaign", "spend", "revenue", "conversions", "ctr", "cpa", "roas", "profit"]], hide_index=True, **STRETCH)
pc = rollup(cube, ["platform", "campaign"])
st.plotly_chart(px.density_heatmap(pc, x="platform", y="campaign", z="roas", histfunc="avg", text_auto=".1f",
                                   title="ROAS by campaign and platform"), **STRETCH)
st.subheader("Trend")
daily = rollup(cube.assign(month=cube["date"].str[:7]), ["month", "platform"])
st.plotly_chart(px.line(daily, x="month", y="roas", color="platform", title="Monthly ROAS"), **STRETCH)
st.subheader("Audience segments (from raw rows)")
seg = st.selectbox("Break down by", ["city", "age", "gender", "device"])
g = df.groupby(["platform", seg])[["impressions", "clicks", "spend", "conversions", "revenue"]].sum().reset_index()
from src.metrics.kpis import add_kpis
g = add_kpis(g)
st.plotly_chart(px.bar(g, x=seg, y="roas", color="platform", barmode="group", title=f"ROAS by {seg}"), **STRETCH)
