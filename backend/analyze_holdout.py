# analyze_holdout.py
import pandas as pd

df = pd.read_csv("training_dataset_binance_v2.csv", parse_dates=["timestamp"])
df = df.sort_values(["symbol","timestamp"])

rows = []
for sym, g in df.groupby("symbol", sort=False):
    cut = int(len(g)*0.8)
    test = g.iloc[cut:]
    up_rate = test["label_up"].mean()
    n = len(test)
    baseline = max(up_rate, 1-up_rate)  # majority class accuracy
    rows.append([sym, n, round(up_rate,3), round(baseline,3)])

out = pd.DataFrame(rows, columns=["symbol","test_rows","up_rate","baseline_acc"]).sort_values("test_rows", ascending=False)
print(out.to_string(index=False))
