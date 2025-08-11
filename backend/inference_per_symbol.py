# inference_per_symbol.py (v3, 27-feature, robust imports)

import os
import sys
import json
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import joblib
from datetime import timezone, timedelta  # make sure timedelta is imported

# ---------- Path bootstrap so 'app' is importable from backend/ or backend/scripts ----------
_THIS = Path(__file__).resolve()
BACKEND = _THIS.parent.parent if _THIS.parent.name == "scripts" else _THIS.parent
APP_DIR = BACKEND / "app"
for p in (BACKEND, APP_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Runtime state import with fallback
try:
    from app.utils.runtime_state import get_last_upserts
except ModuleNotFoundError:
    from utils.runtime_state import get_last_upserts


# ---------- Robust .env loader (walk up to backend/configs/.env) ----------
def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p):
            return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")
load_dotenv(dotenv_path=_find_env_path())

# ---------- Config ----------
DB_NAME       = os.getenv("MONGO_DB", "CoinCluster")
PRICE_COLL    = os.getenv("MONGO_COLLECTION", "binance_price_data")
SIGNAL_COLL   = os.getenv("SIGNAL_COLLECTION", "signals")
WRITE_SIGNALS = True
LOOKBACK      = int(os.getenv("LOOKBACK", "36"))

GLOBAL_MODEL_PATH     = Path("model_price_direction_xgb_v3.pkl")
GLOBAL_THRESHOLD_PATH = Path("model_threshold_v3.txt")
FEATURE_ORDER_PATH    = Path("model_features_v3.json")

# Fallback (should match training’s 27-feature list)
FEATURES_27_FALLBACK = [
    "close","volume",
    "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","atr_norm","bb_upper","bb_lower","bb_middle","bb_width","bb_pctb",
    "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10",
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    "ret_vol_6h","ret_vol_12h","ret_vol_24h"
]

# ---------- Mongo ----------
client = MongoClient(os.getenv("MONGO_URI"))
db = client[DB_NAME]
prices = db[PRICE_COLL]
signals = db[SIGNAL_COLL]

# ---------- Health snapshot ----------
def _log_db_health(db, collection_name):
    c = db[collection_name]
    try:
        count = c.estimated_document_count()
        symbols = sorted(c.distinct("symbol"))
    except Exception as e:
        print(f"[HEALTH] Error reading DB health: {e}")
        return
    print(f"[HEALTH] Connected DB='{db.name}', Coll='{collection_name}'  Docs~{count:,}")
    print(f"[HEALTH] Active symbols ({len(symbols)}): {symbols[:8]}{'...' if len(symbols)>8 else ''}")
_log_db_health(db, PRICE_COLL)

# ---------- Helpers ----------
def _load_feature_order(default_order):
    """Load training-time feature order; fallback to provided default."""
    path = FEATURE_ORDER_PATH
    try:
        if path.exists():
            names = json.loads(path.read_text())
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                print(f"[INFER] Loaded feature order from {path} ({len(names)} features)")
                return names
            else:
                print(f"[INFER] {path} malformed; using fallback feature order.")
    except Exception as e:
        print(f"[INFER] Failed reading {path}: {e}; using fallback feature order.")
    return list(default_order)

def load_global_model_and_threshold():
    model = joblib.load(GLOBAL_MODEL_PATH)
    thr = 0.5
    if GLOBAL_THRESHOLD_PATH.exists():
        try:
            thr = float(GLOBAL_THRESHOLD_PATH.read_text().strip())
        except Exception:
            pass
    return model, thr

def fetch_recent(symbol, n=LOOKBACK):
    cur = prices.find({"symbol": symbol}, {
        "_id": 0, "symbol": 1, "timestamp": 1,
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
        "indicators": 1
    }).sort("timestamp", -1).limit(n)
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "indicators" in df.columns:
        ind = pd.json_normalize(df["indicators"])
        df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)
    # ensure numeric
    for c in ["open","high","low","close","volume",
              "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
              "atr_14","bb_upper","bb_lower","bb_middle",
              "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_features_row(df, feature_order):
    """Compute derived features and return last-row vector in training-time order."""
    if df.empty or len(df) < 25:
        return None

    df = df.copy()
    df["atr_norm"] = df["atr_14"] / df["close"]
    bw = (df.get("bb_upper") - df.get("bb_lower"))
    bw_safe = bw.replace(0, np.nan)
    df["bb_width"] = bw_safe / df["close"]
    df["bb_pctb"]  = (df["close"] - df.get("bb_lower")) / (bw_safe.replace(0, np.nan))

    df["ret_1h"]  = df["close"].pct_change(1)
    df["ret_3h"]  = df["close"].pct_change(3)
    df["ret_6h"]  = df["close"].pct_change(6)
    df["ret_12h"] = df["close"].pct_change(12)
    df["ret_24h"] = df["close"].pct_change(24)

    r = df["close"].pct_change()
    df["ret_vol_6h"]  = r.rolling(6).std()
    df["ret_vol_12h"] = r.rolling(12).std()
    df["ret_vol_24h"] = r.rolling(24).std()

    row = df.iloc[-1].copy()

    vec, missing = [], []
    for f in feature_order:
        val = row.get(f, np.nan)
        if isinstance(val, (float, int)) and (np.isinf(val) or np.isnan(val)):
            missing.append(f)
            vec.append(np.nan)
        else:
            vec.append(val)

    if missing:
        return None  # wait for indicators to fill in
    return np.array(vec, dtype=float), row["timestamp"]

def upsert_signal(symbol, ts, prob_up, decision, model_name="global_v3"):
    if not WRITE_SIGNALS:
        return
    ts_aware = pd.to_datetime(ts, utc=True).to_pydatetime()  # tz-aware UTC
    signals.update_one(
        {"symbol": symbol, "timestamp": ts_aware},
        {"$set": {
            "symbol": symbol,
            "timestamp": ts_aware,
            "prob_up": float(prob_up),
            "decision": decision,
            "model": model_name
        }},
        upsert=True
    )



# ---------- Main ----------
if __name__ == "__main__":
    # Start with all symbols in price collection
    all_symbols = sorted(prices.distinct("symbol"))

    # Keep only "fresh" symbols that have a bar within the last N hours
    FRESH_HOURS = int(os.getenv("FRESH_HOURS", "6"))
    fresh_cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=FRESH_HOURS)




    def is_fresh(sym: str) -> bool:
        doc = prices.find_one({"symbol": sym}, sort=[("timestamp", -1)], projection={"timestamp": 1})
        return bool(doc and pd.to_datetime(doc["timestamp"], utc=True) >= fresh_cut)

    symbols = [s for s in all_symbols if is_fresh(s)]

    model, thr = load_global_model_and_threshold()
    feature_order = _load_feature_order(FEATURES_27_FALLBACK)

    n_expected = getattr(model, "n_features_in_", len(feature_order))
    if n_expected != len(feature_order):
        print(f"[INFER] WARNING: Model expects {n_expected} features but feature_order has {len(feature_order)}.")
        print("        Ensure v3 model + v3 feature JSON are in place.")

    results = []
    for sym in symbols:
        try:
            df = fetch_recent(sym, LOOKBACK)
            feat = build_features_row(df, feature_order)
            if feat is None:
                results.append((sym, None, "INSUFFICIENT_DATA"))
                continue

            x, ts = feat
            if x.shape[0] != n_expected:
                results.append((sym, None, f"ERROR: Feature shape mismatch, expected: {n_expected}, got {x.shape[0]}"))
                continue

            proba = model.predict_proba(x.reshape(1, -1))[0, 1].item()
            decision = "UP" if proba >= thr else "DOWN"
            results.append((sym, proba, decision))

            upsert_signal(sym, pd.to_datetime(ts).to_pydatetime().replace(tzinfo=timezone.utc), proba, decision)

        except Exception as e:
            results.append((sym, None, f"ERROR: {e}"))

    rows = []
    for sym, p, dec in results:
        rows.append([sym, (None if p is None else round(p,4)), dec])
    df_out = pd.DataFrame(rows, columns=["symbol","prob_up","decision"]).sort_values(
        by=["prob_up"], ascending=False, na_position="last"
    )
    print(df_out.to_string(index=False))