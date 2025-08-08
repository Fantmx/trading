import os
import numpy as np
import pandas as pd
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# ---------- DB ----------
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
coll = db["binance_price_data"]

# Make sure you have this index created once in MDB shell:
# db.binance_price_data.createIndex({ symbol: 1, timestamp: 1 })

COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]

BATCH_SIZE = 2000  # bulk write chunk size

# ---------- Indicator helpers ----------
def rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr_wilder(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    c = df["close"]

    # MA/EMA
    df["sma_10"] = c.rolling(10).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["ema_10"] = c.ewm(span=10, adjust=False).mean()

    # RSI (Wilder)
    df["rsi_14"] = rsi_wilder(c, 14)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR (Wilder)
    df["atr_14"] = atr_wilder(df["high"], df["low"], c, 14)

    # Bollinger (20)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["bb_middle"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    return df

def round_or_none(x):
    return float(round(x, 6)) if pd.notnull(x) else None

# ---------- Process one symbol ----------
def process_symbol(symbol: str):
    cur = coll.find(
        {"symbol": symbol},
        {"_id": 0, "symbol": 1, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ).sort("timestamp", 1)
    df = pd.DataFrame(list(cur))
    if df.empty:
        print(f"⚠️ {symbol}: no data")
        return

    # types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["timestamp","close","high","low"], inplace=True)

    # compute
    df = compute_indicators(df)

    # build UpdateOne ops (overwrite)
    ops = []
    for _, r in df.iterrows():
        feats = {
            "sma_10":      round_or_none(r["sma_10"]),
            "sma_50":      round_or_none(r["sma_50"]),
            "ema_10":      round_or_none(r["ema_10"]),
            "rsi_14":      round_or_none(r["rsi_14"]),
            "macd":        round_or_none(r["macd"]),
            "macd_signal": round_or_none(r["macd_signal"]),
            "atr_14":      round_or_none(r["atr_14"]),
            "bb_middle":   round_or_none(r["bb_middle"]),
            "bb_upper":    round_or_none(r["bb_upper"]),
            "bb_lower":    round_or_none(r["bb_lower"]),
        }
        ops.append(UpdateOne(
            {"symbol": symbol, "timestamp": r["timestamp"].to_pydatetime()},
            {"$set": {"indicators": feats}},
            upsert=False
        ))

    # bulk write in chunks
    total_mod = 0
    for i in range(0, len(ops), BATCH_SIZE):
        chunk = ops[i:i+BATCH_SIZE]
        res = coll.bulk_write(chunk, ordered=False)
        total_mod += (res.modified_count or 0) + (res.upserted_count or 0)

    print(f"✅ {symbol}: updated {total_mod} docs")

# ---------- Run ----------
if __name__ == "__main__":
    for sym in COINS:
        process_symbol(sym)
