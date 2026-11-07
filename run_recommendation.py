"""Usage: python run_recommendation.py [path/to/ads.csv] [budget]
Aggregates the data (pandas engine), fits response curves, recommends a split of the budget and writes outputs/."""
import sys
from pathlib import Path
import pandas as pd
from src.data_generation.simulator import write_csv, TRUE_ELASTICITY
from src.engines import pandas_engine
from src.recommendation.optimizer import recommend, format_split
from src.recommendation.validation import bootstrap_shares, backtest, ground_truth_check

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ads_1000000.csv")
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    if not path.exists():
        print(f"{path} not found: generating 1,000,000 simulated rows"); path.parent.mkdir(exist_ok=True); write_csv(path, 1_000_000)
    out = Path("outputs"); out.mkdir(exist_ok=True)
    cube = pandas_engine.run(path)
    rec = recommend(cube, budget)
    rec["table"].to_csv(out / "recommendation.csv", index=False)
    rec["comparison"].to_csv(out / "strategy_comparison.csv", index=False)
    bootstrap_shares(rec).to_csv(out / "bootstrap_shares.csv", index=False)
    backtest(cube)["table"].to_csv(out / "backtest.csv", index=False)
    if "ads_" in path.name:     # simulated data: ground truth is known
        gt = ground_truth_check(rec, TRUE_ELASTICITY)
        gt["elasticities"].to_csv(out / "elasticity_recovery.csv", index=False)
        print("Uplift vs current mix under TRUE curves: recommended %.1f%%, oracle %.1f%%" % (gt["uplift_recommended_pct"], gt["uplift_oracle_pct"]))
    print(rec["table"].round(3).to_string(index=False)); print(); print(f"Recommended split of Rs {budget:,.0f}:"); print(format_split(rec["table"]))
