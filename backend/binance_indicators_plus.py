import os
import numpy as np
import pandas as pd
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
coll = client["CoinCluster"]["binance_price_data"]

COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]
BATCH = 2000

def rsi_wilder(s, period=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn
    return 100 - (100 / (1 + rs))

def stoch_rsi(s, period=14, k=3, d=3):
    rsi = rsi_wilder(s, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    srsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    k_line = srsi.rolling(k).mean()*100
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def process(symbol):
    cur = coll.find({"symbol": symbol}, {"_id":0,"timestamp":1,"close":1,"volume":1}).sort("timestamp",1)
    df = pd.DataFrame(list(cur))
    if df.empty:
        print(f"⚠️ {symbol}: no data"); return
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp")

    # Compute features
    df["stoch_rsi_k"], df["stoch_rsi_d"] = stoch_rsi(df["close"], 14, 3, 3)
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_roc_10"] = df["volume"].pct_change(10)  # rate of change

    ops = []
    for _, r in df.iterrows():
        feats = {
            "stoch_rsi_k": float(round(r["stoch_rsi_k"],6)) if pd.notnull(r["stoch_rsi_k"]) else None,
            "stoch_rsi_d": float(round(r["stoch_rsi_d"],6)) if pd.notnull(r["stoch_rsi_d"]) else None,
            "volume_sma_20": float(round(r["volume_sma_20"],6)) if pd.notnull(r["volume_sma_20"]) else None,
            "volume_roc_10": float(round(r["volume_roc_10"],6)) if pd.notnull(r["volume_roc_10"]) else None,
        }
        ops.append(UpdateOne(
            {"symbol": symbol, "timestamp": r["timestamp"].to_pydatetime()},
            {"$set": {f"indicators.{k}": v for k,v in feats.items()}}
        ))
    for i in range(0, len(ops), BATCH):
        coll.bulk_write(ops[i:i+BATCH], ordered=False)
    print(f"✅ {symbol}: updated {len(ops)} docs")

if __name__ == "__main__":
    for s in COINS:
        process(s)
