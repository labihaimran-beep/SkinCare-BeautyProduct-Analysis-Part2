# Digital Advertising Budget Optimizer (Big Data Analytics, Project 2)

Management asks: **where should we spend the next Rs 1,000,000?** This project simulates advertising records for five
platforms (Facebook, Instagram, Google, TikTok, YouTube), computes the standard KPIs, compares **pandas vs MapReduce vs
Spark** from 10,000 to 10,000,000 records, and builds a **Marketing Budget Recommendation Engine**.
Full write-up: `docs/Project_2_Report.docx` (summary: `PROJECT_SUMMARY.md`). Built in the same style as Project 1.

> **Scope note:** all data is **simulated**. Results describe the simulator (a teaching model), not real campaigns.
> The recommendation is a planning heuristic to be validated with a small live test before real spend.

## Quick start
```
python -m venv .venv
.venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```
Run it with `streamlit run app.py`, not the IDE's Run button.

## Command line
```
python run_benchmark.py                       # generates data/ads_{10000..10000000}.csv, runs engines, writes outputs/
python run_benchmark.py --engines pandas mapreduce    # without Spark
python run_recommendation.py data/ads_1000000.csv 1000000
python make_report_assets.py                  # tables and figures used in the report
```
Spark is optional: `pip install -r requirements-optional.txt` (needs Java 8/11/17/21), see `docs/SPARK_SETUP.md`.
The 10M-row file is ~860 MB; `data/` and `outputs/` are git-ignored. Measured results are kept in `results/`.

## Data columns
`date, campaign, platform, impressions, clicks, spend, conversions, revenue, age (band), gender, city, device` (Rs).

## KPIs
CTR = clicks / impressions; CPC = spend / clicks; CPM = 1000 x spend / impressions; CPA = spend / conversions;
conversion rate = conversions / clicks; ROAS = revenue / spend; profit = revenue - spend (no product cost).
Ratios are always computed on summed totals, never as an average of row ratios.

## Engines (same job on each)
Read CSV, group by (date, platform, campaign), sum five measures. `src/engines/`: `pandas_engine.py`, `mapreduce_engine.py`
(map / combine / shuffle / reduce on multiprocessing; teaching simulation, not Hadoop), `spark_engine.py` (PySpark).
Each benchmark run is a separate process; the harness checks all engines return identical results.

## Recommendation engine
1. Fit `revenue = a x spend^b` per platform on daily data (log-log OLS); b < 1 means diminishing returns.
2. Rescale history to a Rs 1M/month programme (ROAS and b are scale-free).
3. Maximise total predicted revenue with a greedy equal-increment allocation (optimal for concave curves), subject to
   min/max share per platform and a trust region (default 0.5x-2x of the current share) around observed spend levels.
4. Report uncertainty (bootstrap over days), an out-of-time backtest, constraint sensitivity and, because the data is
   simulated, a check against the simulator's true elasticities.

## Limitations
Simulated data; the answer depends on the simulator's parameters and on the guardrails (see sensitivity table);
no attribution overlap, creative effects or seasonality beyond the simulator; Spark was not measured in development
(no PyPI access); MapReduce is a single-machine simulation, so it shows the programming model rather than cluster scaling.
