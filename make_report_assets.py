"""Produces the tables and figures used in the project report from real runs (outputs/ and docs/figures/)."""
import time, statistics as st
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_generation.simulator import TRUE_ELASTICITY
from src.engines import pandas_engine, mapreduce_engine
from src.metrics.kpis import rollup
from src.recommendation.optimizer import recommend, format_split, sensitivity
from src.recommendation.validation import bootstrap_shares, backtest, ground_truth_check

OUT, FIG = Path("outputs"), Path("docs/figures"); OUT.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = {"Facebook": "#3b5998", "Instagram": "#c13584", "Google": "#2e8b57", "TikTok": "#222222", "YouTube": "#d62828"}

# ---- MapReduce ablation (1M rows, median of 3) ----
f = "data/ads_1000000.csv"
rows = []
for name, kw in (("1 worker, combiner", dict(workers=1)), ("2 workers, combiner", dict(workers=2)),
                 ("1 worker, no combiner", dict(workers=1, combine=False)), ("2 workers, no combiner", dict(workers=2, combine=False))):
    ts = []
    for _ in range(3):
        t = time.perf_counter(); mapreduce_engine.run(f, **kw); ts.append(time.perf_counter() - t)
    rows.append(dict(configuration=name, seconds_median=st.median(ts), seconds_min=min(ts), seconds_max=max(ts)))
pd.DataFrame(rows).to_csv(OUT / "mapreduce_ablation.csv", index=False)

# ---- Benchmark figure ----
b = pd.read_csv(OUT / "benchmark_results.csv"); ok = b[b.status == "ok"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for eng, g in ok.groupby("engine"):
    ax[0].plot(g.rows, g.seconds_median, "o-", label=eng); ax[1].plot(g.rows, g.peak_mb, "o-", label=eng)
for a, t, y in zip(ax, ("Median run time (3 runs)", "Peak memory of largest process"), ("seconds", "MB")):
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("records"); a.set_ylabel(y); a.set_title(t); a.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "benchmark.png"); plt.close(fig)

# ---- Recommendation on 1M and 10M ----
res = {}
for n in (1_000_000, 10_000_000):
    cube = pandas_engine.run(f"data/ads_{n}.csv"); res[n] = (cube, recommend(cube, 1_000_000))
cube, rec = res[1_000_000]
rec["table"].to_csv(OUT / "recommendation.csv", index=False); rec["comparison"].to_csv(OUT / "strategy_comparison.csv", index=False)
bs = bootstrap_shares(rec, 200); bs.to_csv(OUT / "bootstrap_shares.csv", index=False)
bt = backtest(cube)["table"]; bt.to_csv(OUT / "backtest.csv", index=False)
sens = sensitivity(cube); sens.to_csv(OUT / "sensitivity.csv", index=False)
gt = ground_truth_check(rec, TRUE_ELASTICITY)
gt["elasticities"].to_csv(OUT / "elasticity_recovery.csv", index=False); gt["shares"].to_csv(OUT / "oracle_shares.csv", index=False)
pd.Series({**gt["revenue_under_true_curves"], "uplift_recommended_pct": gt["uplift_recommended_pct"],
           "uplift_oracle_pct": gt["uplift_oracle_pct"], "capture_ratio": gt["capture_ratio"]}).to_csv(OUT / "ground_truth_summary.csv")
stab = pd.DataFrame({"share_1M": rec["table"].set_index("platform").recommended_share,
                     "share_10M": res[10_000_000][1]["table"].set_index("platform").recommended_share}).reset_index()
stab.to_csv(OUT / "stability_1M_vs_10M.csv", index=False)
kp = rollup(cube, "platform").sort_values("roas", ascending=False); kp.to_csv(OUT / "platform_kpis.csv", index=False)
kc = rollup(cube, "campaign").sort_values("roas", ascending=False); kc.to_csv(OUT / "campaign_kpis.csv", index=False)
(OUT / "recommended_split.txt").write_text(format_split(rec["table"]))

# ---- Figures ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
ax[0].bar(kp.platform, kp.roas, color=[COL[p] for p in kp.platform]); ax[0].set_title("ROAS (revenue per Rs spent)")
ax[1].bar(kp.platform, kp.cpa, color=[COL[p] for p in kp.platform]); ax[1].set_title("CPA (Rs per conversion)")
ax[2].bar(kp.platform, 100 * kp.ctr, color=[COL[p] for p in kp.platform]); ax[2].set_title("CTR (%)")
for a in ax: a.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(FIG / "platform_kpis.png"); plt.close(fig)

t = rec["table"]; x = np.arange(len(t)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(x - w / 2, 100 * t.current_share, w, label="Current mix", color="#9aa5b1"); ax.bar(x + w / 2, 100 * t.recommended_share, w, label="Recommended", color="#c2185b")
lo = 100 * bs.set_index("platform").loc[t.platform, "p05"].values; hi = 100 * bs.set_index("platform").loc[t.platform, "p95"].values
ax.errorbar(x + w / 2, 100 * t.recommended_share, yerr=[100 * t.recommended_share - lo, hi - 100 * t.recommended_share], fmt="none", ecolor="k", capsize=3)
ax.set_xticks(x); ax.set_xticklabels(t.platform); ax.set_ylabel("% of budget"); ax.legend(frameon=False); ax.set_title("Current vs recommended split (bars: bootstrap 5-95%)")
fig.tight_layout(); fig.savefig(FIG / "recommendation.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for p, c in rec["curves"].items():
    xs = np.linspace(c.min_daily_spend, c.max_daily_spend, 60); ax.plot(xs, c.revenue(xs), label=f"{p} (b={c.b:.2f})", color=COL[p])
ax.set_xlabel("Daily spend (Rs, rescaled to a Rs 1M/month programme)"); ax.set_ylabel("Daily revenue (Rs)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Fitted response curves: revenue = a x spend^b"); fig.tight_layout(); fig.savefig(FIG / "response_curves.png"); plt.close(fig)

e = gt["elasticities"]; fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.scatter(e.true_b, e.fitted_b, color="#c2185b"); ax.plot([0.5, 0.9], [0.5, 0.9], "k--", lw=1)
for _, r in e.iterrows(): ax.annotate(r.platform, (r.true_b, r.fitted_b), textcoords="offset points", xytext=(5, -9), fontsize=8)
ax.set_xlabel("True elasticity (simulator)"); ax.set_ylabel("Fitted elasticity"); ax.set_title("Elasticity recovery")
fig.tight_layout(); fig.savefig(FIG / "elasticity_recovery.png"); plt.close(fig)
print("done"); print(rec["table"].round(3).to_string()); print(rec["comparison"].round(1).to_string())
