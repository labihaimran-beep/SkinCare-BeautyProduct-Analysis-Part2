# Digital Advertising Budget Optimizer: project summary

Full report: `docs/Project_2_Report.docx`. All data is **simulated**; findings describe the simulator, not real campaigns.

**Question:** where should the next Rs 1,000,000 be spent across Facebook, Instagram, Google, TikTok and YouTube?

**What was built:** seeded data simulator (10k to 10M rows) with known ground truth; KPI module (CTR, CPC, CPM, CPA, conversion rate,
ROAS, profit); three engines for one aggregation job (pandas, MapReduce simulation, Spark); a process-isolated benchmark harness that
checks engines agree; a recommendation engine (per-platform response curves, constrained optimiser, bootstrap, backtest, sensitivity,
ground-truth check); a Streamlit app; tests.

**Results (measured, 2-core machine):** pandas 8.4 s vs MapReduce simulation 9.6 s at 10M rows (identical results at every size);
pandas is faster at every size, MapReduce's largest process uses less memory. Spark was not run (PySpark not installable in the dev sandbox).
Recommended split: Google 50%, Facebook 25%, Instagram 11%, YouTube 8%, TikTok 6% (current mix 32/18/22/16/13);
model-expected revenue +13.1%. Elasticities recovered within 0.006 of the simulator's truth; out-of-time backtest error 8 to 9%.

**Main caveat:** exact shares depend on the guardrails (Google 40% to 79% across settings); the direction (more Google/Facebook, less
Instagram/YouTube/TikTok) is stable. Validate with a small live test before real spend.

**Testing:** 21 tests passed in development; the Streamlit page tests (12) and the Spark test need Streamlit/Plotly/PySpark and were not run there.
Measured outputs are in `results/`.
