import os
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
collection = db["binance_price_data"]

# Make sure you created this index once:
# db.binance_price_data.createIndex({ symbol: 1, timestamp: 1 })

COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]

BATCH_SIZE = 1000  # bulk write chunk size
MAX_WORKERS = 4    # threads across symbols

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    s = df["close"]

    # SMA / EMA
    df["sma_10"]  = s.rolling(10).mean()
    df["sma_50"]  = s.rolling(50).mean()
    df["ema_10"]  = s.ewm(span=10, adjust=False).mean()

    # RSI(14)
    delta = s.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df

def update_indicators_for_symbol(symbol: str):
    # Pull only what we need
    cursor = collection.find(
        {"symbol": symbol},
        {"_id": 0, "symbol": 1, "timestamp": 1, "close": 1}
    ).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return (symbol, 0)

    # Normalize timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp", "close"], inplace=True)

    # Compute indicators on full series
    df = compute_indicators(df)

    # Find which docs actually need indicators
    missing_ts = set(
        d["timestamp"] for d in collection.find(
            {"symbol": symbol, "indicators": {"$exists": False}},
            {"timestamp": 1, "_id": 0}
        )
    )
    if not missing_ts:
        print(f"⏭ {symbol}: nothing to do (all have indicators).")
        return (symbol, 0)

    # Keep only rows that have indicators computed AND are missing in DB
    mask = (
        df["timestamp"].isin(missing_ts) &
        df[["sma_10","sma_50","ema_10","rsi_14","macd","macd_signal"]].notna().all(axis=1)
    )
    to_update = df.loc[mask]

    if to_update.empty:
        print(f"⏭ {symbol}: no eligible rows (wait for enough history).")
        return (symbol, 0)

    # Build bulk operations
    ops = []
    for _, r in to_update.iterrows():
        features = {
            "sma_10": float(round(r["sma_10"], 6)),
            "sma_50": float(round(r["sma_50"], 6)),
            "ema_10": float(round(r["ema_10"], 6)),
            "rsi_14": float(round(r["rsi_14"], 6)),
            "macd": float(round(r["macd"], 6)),
            "macd_signal": float(round(r["macd_signal"], 6)),
        }
        ops.append(UpdateOne(
            {"symbol": symbol, "timestamp": r["timestamp"].to_pydatetime()},
            {"$set": {"indicators": features}},
            upsert=False
        ))

    # Bulk write in chunks
    written = 0
    for i in range(0, len(ops), BATCH_SIZE):
        chunk = ops[i:i+BATCH_SIZE]
        result = collection.bulk_write(chunk, ordered=False)
        written += (result.modified_count or 0) + (result.upserted_count or 0)

    print(f"✅ {symbol}: updated {written} docs.")
    return (symbol, written)

def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(update_indicators_for_symbol, sym) for sym in COINS]
        for f in as_completed(futures):
            sym, n = f.result()

if __name__ == "__main__":
    main()
