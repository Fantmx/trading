import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------- Config ----------
OUTPUT_CSV = "training_dataset_binance.csv"
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]
REQUIRED_INDICTORS = ["sma_10","sma_50","ema_10","rsi_14","macd","macd_signal"]

# ---------- Connect ----------
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
coll = client["CoinCluster"]["binance_price_data"]

def load_symbol_df(symbol: str) -> pd.DataFrame:
    """Load one symbol from Mongo with only needed fields, sorted by time."""
    cur = coll.find(
        {"symbol": symbol, "indicators": {"$exists": True}},
        {"_id": 0, "symbol": 1, "timestamp": 1, "close": 1, "volume": 1, "indicators": 1},
        sort=[("timestamp", 1)]
    )
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df

    # Flatten indicators and normalize dtypes
    ind = pd.json_normalize(df["indicators"])
    df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    for col in REQUIRED_INDICTORS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create label: next-hour close > current close
    df = df.sort_values("timestamp")
    df["next_close"] = df["close"].shift(-1)
    df["label_up"] = (df["next_close"] > df["close"]).astype("Int8")  # 0/1, last row becomes <NA> then dropped

    # Keep only rows with indicators + label available
    keep_cols = ["timestamp","symbol","close","volume"] + REQUIRED_INDICTORS + ["label_up"]
    df = df[keep_cols].dropna(subset=REQUIRED_INDICTORS + ["label_up"]).reset_index(drop=True)

    return df

def main():
    parts = []
    total_rows = 0
    for sym in COINS:
        df_sym = load_symbol_df(sym)
        if df_sym.empty:
            print(f"{sym}: 0 rows (no indicators or data)")
            continue
        parts.append(df_sym)
        total_rows += len(df_sym)
        print(f"{sym}: {len(df_sym):,} rows")

    if not parts:
        print("No data collected. Did you run the indicator script?")
        return

    final = pd.concat(parts, ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(final):,} rows to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
