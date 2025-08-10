import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

OUTPUT_CSV = "training_dataset_binance_v3.csv"
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]

BASE_IND = [
    "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","bb_upper","bb_lower","bb_middle",
    "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10"
]

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
coll = client["CoinCluster"]["binance_price_data"]

def build_symbol(symbol):
    cur = coll.find(
        {"symbol": symbol, "indicators": {"$exists": True}},
        {"_id":0,"symbol":1,"timestamp":1,"open":1,"high":1,"low":1,"close":1,"volume":1,"indicators":1},
        sort=[("timestamp",1)]
    )
    df = pd.DataFrame(list(cur))
    if df.empty: return df
    ind = pd.json_normalize(df["indicators"])
    df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)

    # dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"] + BASE_IND:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derived: ATR normalized & Bollinger widths
    df["atr_norm"] = df["atr_14"] / df["close"]
    bw = (df["bb_upper"] - df["bb_lower"])
    bw_safe = bw.replace(0, np.nan)
    df["bb_width"] = bw_safe / df["close"]
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / bw_safe

    # Momentum returns
    for h in [1,3,6,12,24]:
        df[f"ret_{h}h"] = df["close"].pct_change(h)

    # Rolling volatility of returns
    for w in [6,12,24]:
        df[f"ret_vol_{w}h"] = df["close"].pct_change().rolling(w).std()

    # Label: next hour up?
    df["next_close"] = df["close"].shift(-1)
    df["label_up"] = (df["next_close"] > df["close"]).astype("Int8")

    # Keep cols
    feats = [
        "close","volume",
        "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
        "atr_14","atr_norm","bb_upper","bb_lower","bb_middle","bb_width","bb_pctb",
        "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10",
        "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
        "ret_vol_6h","ret_vol_12h","ret_vol_24h"
    ]
    keep = ["timestamp","symbol"] + feats + ["label_up"]

    # Clean & drop NaNs needed for training
    must_have = [c for c in feats] + ["label_up"]
    df = df[keep].replace([np.inf,-np.inf], np.nan).dropna(subset=must_have).reset_index(drop=True)
    return df

if __name__ == "__main__":
    parts = []
    total = 0
    for s in COINS:
        d = build_symbol(s)
        if d.empty:
            print(f"{s}: 0 rows"); continue
        parts.append(d); total += len(d)
        print(f"{s}: {len(d):,} rows")
    if not parts:
        print("No data"); raise SystemExit
    final = pd.concat(parts, ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(final):,} rows to '{OUTPUT_CSV}'")
