import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Try XGBoost; fallback to sklearn GBM if not available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

CSV_PATH = "training_dataset_binance.csv"
MODEL_PATH = "model_price_direction_xgb.pkl"
FEATURES_PATH = "model_features.txt"

# -------- 1) Load ----------
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], low_memory=False)
df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

# -------- 2) Features & label ----------
FEATURES = [
    "close", "volume",
    "sma_10", "sma_50", "ema_10",
    "rsi_14", "macd", "macd_signal"
]
TARGET = "label_up"
df = df.dropna(subset=FEATURES + [TARGET])

# -------- 3) Time-aware split per symbol (last 20% per symbol -> test) ----------
parts_train, parts_test = [], []
for sym, g in df.groupby("symbol", sort=False):
    n = len(g)
    cut = int(n * 0.8)
    parts_train.append(g.iloc[:cut])
    parts_test.append(g.iloc[cut:])

train = pd.concat(parts_train).reset_index(drop=True)
test  = pd.concat(parts_test).reset_index(drop=True)

X_train, y_train = train[FEATURES], train[TARGET].astype(int)
X_test,  y_test  = test[FEATURES],  test[TARGET].astype(int)

print(f"Train rows: {len(train):,} | Test rows: {len(test):,} | Symbols: {df['symbol'].nunique()}")

# -------- 4) Model ----------
if HAS_XGB:
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
else:
    print("xgboost not found; using sklearn. (pip install xgboost to improve speed/accuracy)")
    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )

model.fit(X_train, y_train)

# -------- 5) Evaluation ----------
proba = model.predict_proba(X_test)[:, 1] if HAS_XGB or hasattr(model, "predict_proba") else model.decision_function(X_test)
pred  = (proba >= 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, pred, digits=3))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, pred))

try:
    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC: {auc:.3f}")
except Exception:
    pass

# Optional: quick per-symbol accuracy to spot weak markets
per_sym = test.assign(pred=pred).groupby("symbol").apply(lambda g: (g["label_up"]==g["pred"]).mean())
print("\nPer-symbol accuracy (last 20%):")
print(per_sym.sort_values(ascending=False).round(3))

# -------- 6) Save ----------
joblib.dump(model, MODEL_PATH)
with open(FEATURES_PATH, "w") as f:
    f.write("\n".join(FEATURES))
print(f"\n✅ Saved model to {MODEL_PATH}")
print(f"✅ Saved features to {FEATURES_PATH}")
