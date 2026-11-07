import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from src.data_generation.simulator import TRUE_ELASTICITY
from src.recommendation.optimizer import recommend, format_split, sensitivity
from src.recommendation.validation import bootstrap_shares, backtest, ground_truth_check
from src.ui.common import require, banner, STRETCH

st.title("Marketing Budget Recommendation Engine")
banner()
require("cube", "Load data first on **Data Simulation**.")
cube = st.session_state["cube"]
c1, c2, c3, c4 = st.columns(4)
budget = c1.number_input("Budget to allocate (Rs)", 10_000, 1_000_000_000, 1_000_000, step=50_000)
max_share = c2.slider("Max share per platform", 0.2, 1.0, 0.5, 0.05)
min_share = c3.slider("Min share per platform", 0.0, 0.2, 0.05, 0.01)
lo_t, hi_t = c4.slider("Trust region vs current mix (x)", 0.25, 4.0, (0.5, 2.0), 0.05)
st.caption("The trust region keeps each platform near spend levels that were actually observed. Curves are only reliable inside that range.")
try:
    rec = recommend(cube, float(budget), min_share=min_share, max_share=max_share, trust=(lo_t, hi_t))
except ValueError as e:
    st.error(str(e)); st.stop()
st.session_state["rec"] = rec
t = rec["table"]
st.subheader(f"Recommended split of Rs {budget:,.0f}")
st.code(format_split(t))
show = t.copy()
for c in ("current_share", "recommended_share"):
    show[c] = (100 * show[c]).round(1)
st.dataframe(show.round(3), hide_index=True, **STRETCH)
st.caption("`at_bound` marks platforms held at a constraint: for them the model would move further if allowed (see the sensitivity table).")
a, b = st.columns(2)
a.plotly_chart(px.bar(t.melt(id_vars="platform", value_vars=["current_share", "recommended_share"]), x="platform", y="value",
                      color="variable", barmode="group", title="Current vs recommended share"), **STRETCH)
b.plotly_chart(px.bar(rec["comparison"], x="strategy", y="expected_revenue", title="Model-expected monthly revenue by strategy"), **STRETCH)
st.dataframe(rec["comparison"].round(2), hide_index=True, **STRETCH)
st.subheader("Fitted response curves (revenue = a x spend^b)")
fig = go.Figure()
for p, c in rec["curves"].items():
    xs = np.linspace(c.min_daily_spend, c.max_daily_spend, 60)
    fig.add_trace(go.Scatter(x=xs, y=c.revenue(xs), name=f"{p} (b={c.b:.2f})"))
fig.update_layout(xaxis_title="Daily spend (Rs, rescaled)", yaxis_title="Daily revenue (Rs, rescaled)")
st.plotly_chart(fig, **STRETCH)
with st.expander("Uncertainty, validation and sensitivity"):
    if st.button("Run bootstrap (200 resamples of days)"):
        st.dataframe(bootstrap_shares(rec).round(3), hide_index=True, **STRETCH)
        st.caption("Intervals are narrow when a platform sits at a bound, because the bound, not the data, fixes its share.")
    st.write("Out-of-time backtest (fit on the first 75% of days, predict the rest):")
    st.dataframe(backtest(cube)["table"].round(3), hide_index=True, **STRETCH)
    st.write("Sensitivity to constraints:")
    st.dataframe(sensitivity(cube, float(budget)).round(3), hide_index=True, **STRETCH)
    if st.session_state.get("simulated"):
        gt = ground_truth_check(rec, TRUE_ELASTICITY)
        st.write("Simulation-only check against the simulator's true elasticities:")
        st.dataframe(gt["elasticities"].round(3), hide_index=True, **STRETCH)
        st.write(f"Uplift vs current mix under the TRUE curves: recommended {gt['uplift_recommended_pct']:.1f}%, "
                 f"oracle {gt['uplift_oracle_pct']:.1f}%.")
st.info("Scope: a heuristic for planning on simulated history. It assumes past response curves hold, ignores seasonality "
        "beyond the data, creative quality and attribution overlap, and must be tested with a small live experiment before real use.")
