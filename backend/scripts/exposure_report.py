# exposure_report.py
import pandas as pd
from pathlib import Path

eq = pd.read_csv(Path("backtest_equity.csv"), parse_dates=["timestamp"]).sort_values("timestamp")
cap = 30.0
eq["exposure_pct"] = eq["exposure_frac"] * 100

rows = [
    ("Avg Exposure % of Equity", round(eq["exposure_pct"].mean(), 2)),
    ("Median Exposure %", round(eq["exposure_pct"].median(), 2)),
    ("90th Percentile Exposure %", round(eq["exposure_pct"].quantile(0.90), 2)),
    ("95th Percentile Exposure %", round(eq["exposure_pct"].quantile(0.95), 2)),
    ("Max Exposure %", round(eq["exposure_pct"].max(), 2)),
    ("Time ≥ 99% of Cap (30%)", f"{round((eq['exposure_pct'] >= cap*0.99).mean()*100, 2)}%"),
    ("Time ≥ 90% of Cap (27%)", f"{round((eq['exposure_pct'] >= cap*0.90).mean()*100, 2)}%"),
    ("Avg Open Positions", round(eq["open_positions"].mean(), 2) if "open_positions" in eq else None),
    ("Max Open Positions", int(eq["open_positions"].max()) if "open_positions" in eq else None),
]
for k, v in rows:
    print(f"{k}: {v}")
