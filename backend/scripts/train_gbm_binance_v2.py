import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

# ---- Try XGBoost; fallback to sklearn GBM ----
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

CSV_PATH        = "training_dataset_binance_v2.csv"
MODEL_PATH      = "model_price_direction_xgb_v2.pkl"
FEATURES_PATH   = "model_features_v2.txt"
THRESHOLD_PATH  = "model_threshold_v2.txt"                 # global tuned threshold
PER_SYM_PATH    = "model_thresholds_per_symbol_v2.json"    # per-symbol thresholds
METRIC_FOR_TUNING = "accuracy"   # "accuracy" or "f1"
USE_CALIBRATION   = False        # set True to calibrate probabilities (isotonic CV)

FEATURES = [
    "close","volume",
    "sma_10","sma_50","ema_10",
    "rsi_14","macd","macd_signal",
    "atr_14","atr_norm",
    "bb_upper","bb_lower","bb_width","bb_pctb",
]
TARGET = "label_up"

# ---------------- Helpers ----------------
def clean_xy(X: pd.DataFrame, y: pd.Series):
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask].astype(int)

def tune_threshold(y_true, proba, metric="accuracy", search=(0.30, 0.70, 0.005),
                   min_prec=None, min_rec=None):
    best_t, best_m = 0.5, -1.0
    lo, hi, step = search
    for t in np.arange(lo, hi + 1e-9, step):
        pred = (proba >= t).astype(int)
        if metric == "f1":
            m = f1_score(y_true, pred)
        else:
            m = accuracy_score(y_true, pred)
        if min_prec is not None and precision_score(y_true, pred) < min_prec:
            continue
        if min_rec  is not None and recall_score(y_true, pred)  < min_rec:
            continue
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)

def per_symbol_thresholds(model, df_val, features, metric="accuracy"):
    thrs = {}
    for sym, g in df_val.groupby("symbol", sort=False):
        Xv, yv = clean_xy(g[features], g[TARGET])
        if len(Xv) == 0:
            continue
        proba = model.predict_proba(Xv)[:, 1]
        t, _ = tune_threshold(yv, proba, metric=metric)
        thrs[sym] = t
    return thrs

def apply_per_symbol_thresholds(model, df_test, features, thrs: dict):
    preds, ys, syms = [], [], []
    for sym, g in df_test.groupby("symbol", sort=False):
        Xt, yt = clean_xy(g[features], g[TARGET])
        if len(Xt) == 0:
            continue
        p = model.predict_proba(Xt)[:, 1]
        t = thrs.get(sym, 0.5)
        preds.append((p >= t).astype(int))
        ys.append(yt.values)
        syms.extend([sym] * len(yt))
    if not preds:
        return None
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return pd.DataFrame({"symbol": syms, "y": ys, "pred": preds})

# ---------------- Load & split ----------------
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
df = df.dropna(subset=FEATURES + [TARGET])

# 70/15/15 per symbol (time-aware)
train_parts, val_parts, test_parts = [], [], []
for sym, g in df.groupby("symbol", sort=False):
    n = len(g)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    train_parts.append(g.iloc[:i1])
    val_parts.append(g.iloc[i1:i2])
    test_parts.append(g.iloc[i2:])

train = pd.concat(train_parts, ignore_index=True)
val   = pd.concat(val_parts,   ignore_index=True)
test  = pd.concat(test_parts,  ignore_index=True)

X_train, y_train = clean_xy(train[FEATURES], train[TARGET])
X_val,   y_val   = clean_xy(val[FEATURES],   val[TARGET])
X_test,  y_test  = clean_xy(test[FEATURES],  test[TARGET])

print(f"Rows → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Symbols: {df['symbol'].nunique()}")

# ---------------- Model ----------------
if HAS_XGB:
    base_model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
else:
    print("xgboost not found; using sklearn GBM. (pip install xgboost for best speed/accuracy)")
    base_model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

if USE_CALIBRATION:
    # Calibrates probs using 3-fold CV on training set
    model = CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv=3)
else:
    model = base_model

model.fit(X_train, y_train)

# ---------------- Default threshold (0.50) on Test ----------------
proba_test = model.predict_proba(X_test)[:, 1]
pred05 = (proba_test >= 0.5).astype(int)

print("\n📊 TEST (threshold=0.50)")
print(classification_report(y_test, pred05, digits=3))
print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, pred05))
try:
    print("ROC-AUC:", round(roc_auc_score(y_test, proba_test), 3))
except Exception:
    pass

# ---------------- Tune global threshold on Validation ----------------
proba_val = model.predict_proba(X_val)[:, 1]
best_t, best_m = tune_threshold(y_val, proba_val, metric=METRIC_FOR_TUNING)
print(f"\n🔍 Best global threshold by {METRIC_FOR_TUNING.upper()} on VAL: {best_t:.3f} (score={best_m:.3f})")

# Evaluate tuned global threshold on Test
pred_test_tuned = (proba_test >= best_t).astype(int)
print("\n📊 TEST (global tuned threshold)")
print(classification_report(y_test, pred_test_tuned, digits=3))
print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, pred_test_tuned))

# ---------------- Per-symbol thresholds (tuned on Val) ----------------
thr_map = per_symbol_thresholds(model, val, FEATURES, metric=METRIC_FOR_TUNING)
res = apply_per_symbol_thresholds(model, test, FEATURES, thr_map)
if res is not None:
    acc = accuracy_score(res["y"], res["pred"])
    print(f"\n📊 TEST accuracy with per-symbol thresholds: {acc:.3f}")
    per_sym = res.groupby("symbol").apply(lambda g: (g["y"] == g["pred"]).mean())
    print("\nPer-symbol accuracy (per-symbol tuned):")
    print(per_sym.sort_values(ascending=False).round(3))
else:
    print("\n(No per-symbol results; empty test after cleaning.)")

# ---------------- Save artifacts ----------------
joblib.dump(model, MODEL_PATH)
Path(FEATURES_PATH).write_text("\n".join(FEATURES), encoding="utf-8")
Path(THRESHOLD_PATH).write_text(f"{best_t:.6f}", encoding="utf-8")
Path(PER_SYM_PATH).write_text(json.dumps(thr_map, indent=2), encoding="utf-8")

print(f"\n✅ Saved model → {MODEL_PATH}")
print(f"✅ Saved features → {FEATURES_PATH}")
print(f"✅ Saved global threshold → {THRESHOLD_PATH}")
print(f"✅ Saved per-symbol thresholds → {PER_SYM_PATH}")
