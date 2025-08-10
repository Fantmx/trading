import os, pandas as pd, numpy as np
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# --- robust .env loader ---
def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p): return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")
load_dotenv(dotenv_path=_find_env_path())

DB_NAME = os.getenv("MONGO_DB", "CoinCluster")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "binance_price_data")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[DB_NAME]
coll = db[COLLECTION_NAME]

# --- health ---
def _log_db_health(db, collection_name):
    c = db[collection_name]
    try:
        count = c.estimated_document_count()
        symbols = sorted(c.distinct("symbol"))
    except Exception as e:
        print(f"[HEALTH] Error reading DB health: {e}")
        return
    print(f"[HEALTH] Connected DB='{db.name}', Coll='{collection_name}'  Docs~{count:,}")
    # or: Docs={count:,}

    print(f"[HEALTH] Active symbols ({len(symbols)}): {symbols[:8]}{'...' if len(symbols)>8 else ''}")
_log_db_health(db, COLLECTION_NAME)

FIELDS = ["sma_10","sma_50","ema_10","rsi_14","macd","macd_signal","atr_14",
          "bb_middle","bb_upper","bb_lower","stoch_rsi_k","stoch_rsi_d",
          "volume_sma_20","volume_roc_10"]

def rsi_wilder(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au / ad
    return 100 - (100/(1+rs))

def stoch_rsi(s, n=14, k=3, d=3):
    rsi = rsi_wilder(s, n)
    rmin = rsi.rolling(n).min(); rmax = rsi.rolling(n).max()
    srsi = (rsi - rmin) / (rmax - rmin)
    kline = srsi.rolling(k).mean()*100
    dline = kline.rolling(d).mean()
    return kline, dline

def process(sym):
    cur = coll.find(
        {"symbol": sym},
        {"_id":0,"symbol":1,"timestamp":1,"open":1,"high":1,"low":1,"close":1,"volume":1,"indicators":1}
    ).sort("timestamp",1)
    df = pd.DataFrame(list(cur))
    if df.empty: return 0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["sma_10"] = c.rolling(10).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["ema_10"] = c.ewm(span=10, adjust=False).mean()
    df["rsi_14"] = rsi_wilder(c,14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df["bb_middle"] = mid; df["bb_upper"] = mid + 2*std; df["bb_lower"] = mid - 2*std
    df["stoch_rsi_k"], df["stoch_rsi_d"] = stoch_rsi(c,14,3,3)
    df["volume_sma_20"] = v.rolling(20).mean()
    df["volume_roc_10"] = v.pct_change(10)

    def needs_update(ind):
        if not isinstance(ind, dict): return True
        return any(ind.get(f) is None for f in FIELDS)

    df["needs"] = df["indicators"].apply(needs_update)
    todo = df[df["needs"]].copy()
    if todo.empty: return 0

    ops = []
    for _, r in todo.iterrows():
        feats = {f: (None if pd.isna(r[f]) else float(round(r[f],6))) for f in FIELDS}
        ops.append(UpdateOne({"symbol": sym, "timestamp": r["timestamp"].to_pydatetime()},
                             {"$set": {"indicators": feats}}))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        return res.modified_count
    return 0

if __name__ == "__main__":
    # Pull active symbols from DB so POL/MATIC/etc. are handled automatically
    symbols = sorted(coll.distinct("symbol"))
    total = 0
    for s in symbols:
        total += process(s)
    print(f"[INDICATORS] updated {total} docs across {len(symbols)} symbols")
