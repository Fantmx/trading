import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

CSV_PATH = "training_dataset_binance_v3.csv"
OUT_DIR  = Path("per_symbol_models")
OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "close","volume",
    "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","atr_norm","bb_upper","bb_lower","bb_middle","bb_width","bb_pctb",
    "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10",
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    "ret_vol_6h","ret_vol_12h","ret_vol_24h"
]
TARGET = "label_up"

def clean_xy(X, y):
    X = X.replace([np.inf,-np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask].astype(int)

def tune_threshold(y_true, proba, metric="accuracy"):
    best_t, best_m = 0.5, -1
    for t in np.linspace(0.3, 0.7, 81):
        pred = (proba >= t).astype(int)
        m = accuracy_score(y_true, pred) if metric=="accuracy" else f1_score(y_true, pred)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

symbols = df["symbol"].unique().tolist()
summary = []

for sym in symbols:
    g = df[df["symbol"]==sym].copy()
    n = len(g)
    i1, i2 = int(n*0.70), int(n*0.85)
    train, val, test = g.iloc[:i1], g.iloc[i1:i2], g.iloc[i2:]

    Xtr, ytr = clean_xy(train[FEATURES], train[TARGET])
    Xv, yv   = clean_xy(val[FEATURES],   val[TARGET])
    Xte, yte = clean_xy(test[FEATURES],  test[TARGET])

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=42
        )
    else:
        model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)

    model.fit(Xtr, ytr)
    pv = model.predict_proba(Xv)[:,1]
    thr, _ = tune_threshold(yv, pv, metric="accuracy")

    pte = model.predict_proba(Xte)[:,1]
    pred = (pte >= thr).astype(int)
    acc  = accuracy_score(yte, pred)

    # save
    mpath = OUT_DIR / f"{sym}_model.pkl"
    tpath = OUT_DIR / f"{sym}_threshold.txt"
    joblib.dump(model, mpath)
    tpath.write_text(f"{thr:.6f}")

    summary.append([sym, len(Xtr), len(Xv), len(Xte), thr, acc])

# Print summary
out = pd.DataFrame(summary, columns=["symbol","train","val","test","thr","test_acc"]).sort_values("test_acc", ascending=False)
print(out.to_string(index=False))
