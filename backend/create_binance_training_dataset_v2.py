import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------- Config ----------
OUTPUT_CSV = "training_dataset_binance_v2.csv"
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]

# Indicators we expect to exist in Mongo (from binance_indicators_pro.py)
BASE_INDICATORS = [
    "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","bb_upper","bb_lower"
]

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
coll = client["CoinCluster"]["binance_price_data"]

def load_symbol_df(symbol: str) -> pd.DataFrame:
    cur = coll.find(
        {"symbol": symbol, "indicators": {"$exists": True}},
        {
            "_id": 0, "symbol": 1, "timestamp": 1,
            "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
            "indicators": 1
        },
        sort=[("timestamp", 1)]
    )
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df

    # Flatten indicators
    ind = pd.json_normalize(df["indicators"])
    df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)

    # Types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    numeric_cols = ["open","high","low","close","volume"] + BASE_INDICATORS
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived features
    # Normalize ATR by price; Bollinger features: width and %B
    df["atr_norm"] = df["atr_14"] / df["close"]
    band_width = (df["bb_upper"] - df["bb_lower"])
    # avoid divide-by-zero in pctb
    df["bb_width"] = band_width / df["close"]
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / band_width

    # Label: next-hour up?
    df = df.sort_values("timestamp")
    df["next_close"] = df["close"].shift(-1)
    df["label_up"] = (df["next_close"] > df["close"]).astype("Int8")

    # Final columns for training
    features = [
        "close","volume",
        "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
        "atr_14","atr_norm","bb_upper","bb_lower","bb_width","bb_pctb"
    ]
    keep_cols = ["timestamp","symbol"] + features + ["label_up"]

    # Drop rows missing required indicators or label
    df = df[keep_cols].dropna(subset=[
        "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
        "atr_14","bb_upper","bb_lower","bb_width","bb_pctb","label_up"
    ]).reset_index(drop=True)

    return df

def main():
    parts = []
    total = 0
    for sym in COINS:
        d = load_symbol_df(sym)
        if d.empty:
            print(f"{sym}: 0 rows (no indicators/data)")
            continue
        parts.append(d)
        total += len(d)
        print(f"{sym}: {len(d):,} rows")

    if not parts:
        print("No data found. Did you run binance_indicators_pro.py?")
        return

    final = pd.concat(parts, ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(final):,} rows to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
