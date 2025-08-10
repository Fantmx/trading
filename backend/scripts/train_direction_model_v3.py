# train_direction_model_v3.py
import os, json, math, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import joblib

# -------- robust .env loader (walk to backend/configs/.env) --------
def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p): return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")
load_dotenv(dotenv_path=_find_env_path())

DB_NAME = os.getenv("MONGO_DB", "CoinCluster")
COLL    = os.getenv("MONGO_COLLECTION", "binance_price_data")
MONGO_URI = os.getenv("MONGO_URI")

OUT_MODEL_PATH = Path("model_price_direction_xgb_v3.pkl")
OUT_THRESH_PATH= Path("model_threshold_v3.txt")
OUT_FEATS_PATH = Path("model_features_v3.json")

# 27-feature schema used by inference
FEATURES_27 = [
    "close","volume",
    "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","atr_norm","bb_upper","bb_lower","bb_middle","bb_width","bb_pctb",
    "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10",
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    "ret_vol_6h","ret_vol_12h","ret_vol_24h"
]

# -------- model: prefer xgboost, fallback to sklearn GBM --------
def _get_model():
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",         # fast CPU; change to 'gpu_hist' if you have GPU
            device='cpu',
            random_state=42,
            n_jobs=0
        ), "xgb"
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        warnings.warn("xgboost not found; falling back to GradientBoostingClassifier.")
        return GradientBoostingClassifier(random_state=42), "gbm"

# -------- feature helpers (same math as inference) --------
def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    # df has basic OHLCV and indicator fields already persisted in Mongo as 'indicators'
    # We recompute derived features to match inference perfectly
    df = df.copy()
    for c in ["open","high","low","close","volume",
              "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
              "atr_14","bb_upper","bb_lower","bb_middle",
              "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived features
    df["atr_norm"] = df["atr_14"] / df["close"]
    bw = (df.get("bb_upper") - df.get("bb_lower"))
    bw_safe = bw.replace(0, np.nan)
    df["bb_width"] = bw_safe / df["close"]
    df["bb_pctb"]  = (df["close"] - df.get("bb_lower")) / bw_safe

    df["ret_1h"]  = df["close"].pct_change(1)
    df["ret_3h"]  = df["close"].pct_change(3)
    df["ret_6h"]  = df["close"].pct_change(6)
    df["ret_12h"] = df["close"].pct_change(12)
    df["ret_24h"] = df["close"].pct_change(24)

    r = df["close"].pct_change()
    df["ret_vol_6h"]  = r.rolling(6).std()
    df["ret_vol_12h"] = r.rolling(12).std()
    df["ret_vol_24h"] = r.rolling(24).std()
    return df

def _flatten_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "indicators" in df.columns:
        ind = pd.json_normalize(df["indicators"])
        df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)
    return df

# -------- data fetch & assemble --------
def fetch_symbol_df(coll, symbol: str, min_rows=200):
    cur = coll.find({"symbol": symbol}, {
        "_id":0,"symbol":1,"timestamp":1,
        "open":1,"high":1,"low":1,"close":1,"volume":1,"indicators":1
    }).sort("timestamp", 1)
    df = pd.DataFrame(list(cur))
    if df.empty or len(df) < min_rows:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _flatten_indicators(df)
    df = _compute_derived(df)

    # build label: next hour up/down
    df["close_fwd1"] = df["close"].shift(-1)
    df["y"] = (df["close_fwd1"] > df["close"]).astype(int)

    # drop rows with any NaNs in features or y missing (last row)
    df = df.dropna(subset=FEATURES_27 + ["y"])
    return df

def build_training_frame(coll, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for s in symbols:
        df = fetch_symbol_df(coll, s)
        if df.empty: continue
        df["sym"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    return big

# -------- threshold search --------
def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    # grid search thresholds 0.30 -> 0.70 for a sensible operating point; widen if you like
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.30, 0.70, 41):
        y_hat = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    y_hat = (y_prob >= best_t).astype(int)
    acc = accuracy_score(y_true, y_hat)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return best_t, {"f1": best_f1, "acc": acc, "prec": pr, "recall": rc, "auc": auc}

# -------- main --------
def main():
    print("[TRAIN] Connecting to Mongo…")
    coll = MongoClient(MONGO_URI)[DB_NAME][COLL]
    symbols = sorted(coll.distinct("symbol"))
    print(f"[TRAIN] Symbols in DB: {len(symbols)} -> sample: {symbols[:10]}{'...' if len(symbols)>10 else ''}")

    # Build dataset
    print("[TRAIN] Building dataset (this may take a moment)…")
    df = build_training_frame(coll, symbols)
    if df.empty:
        raise SystemExit("[TRAIN] No data found. Run ingest + indicators first.")

    # Time-based split: last 15% by timestamp for validation
    df = df.sort_values("timestamp")
    ts_cut = df["timestamp"].quantile(0.85)
    train = df[df["timestamp"] < ts_cut].copy()
    valid = df[df["timestamp"] >= ts_cut].copy()

    X_cols = FEATURES_27
    X_tr, y_tr = train[X_cols].values, train["y"].values.astype(int)
    X_va, y_va = valid[X_cols].values, valid["y"].values.astype(int)

    print(f"[TRAIN] Train rows: {len(train):,}  Valid rows: {len(valid):,}")
    model, mname = _get_model()
    print(f"[TRAIN] Fitting model: {mname} on 27 features…")
    model.fit(X_tr, y_tr)

    # Validation metrics + threshold
    if hasattr(model, "predict_proba"):
        p_va = model.predict_proba(X_va)[:,1]
    else:
        # Some sklearn models expose decision_function
        scores = model.decision_function(X_va)
        p_va = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    thr, metrics = pick_threshold(y_va, p_va)
    print(f"[TRAIN] Validation metrics @best F1:")
    print(f"        AUC={metrics['auc']:.3f}  Acc={metrics['acc']:.3f}  Prec={metrics['prec']:.3f}  Rec={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
    print(f"[TRAIN] Selected threshold: {thr:.3f}")

    # Save artifacts
    joblib.dump(model, OUT_MODEL_PATH)
    OUT_THRESH_PATH.write_text(f"{thr:.6f}")
    OUT_FEATS_PATH.write_text(json.dumps(FEATURES_27, indent=2))
    print(f"[TRAIN] Saved: {OUT_MODEL_PATH}, {OUT_THRESH_PATH}, {OUT_FEATS_PATH}")

if __name__ == "__main__":
    main()
