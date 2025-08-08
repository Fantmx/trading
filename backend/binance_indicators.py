import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
collection = db["binance_price_data"]

coin_symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    "XRPUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT",
    "SHIBUSDT", "PEPEUSDT", "APTUSDT", "INJUSDT", "FETUSDT"
]

def add_indicators_to_symbol(symbol):
    cursor = collection.find({"symbol": symbol}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return

    # Convert timestamp to datetime if stored as string/millis
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # --- Technical Indicators ---
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Write results back to MongoDB ---
    updates = 0
    for timestamp, row in df.iterrows():
        features = {
            "sma_10": round(row["sma_10"], 6) if pd.notnull(row["sma_10"]) else None,
            "sma_50": round(row["sma_50"], 6) if pd.notnull(row["sma_50"]) else None,
            "ema_10": round(row["ema_10"], 6) if pd.notnull(row["ema_10"]) else None,
            "rsi_14": round(row["rsi_14"], 6) if pd.notnull(row["rsi_14"]) else None,
            "macd": round(row["macd"], 6) if pd.notnull(row["macd"]) else None,
            "macd_signal": round(row["macd_signal"], 6) if pd.notnull(row["macd_signal"]) else None
        }

        result = collection.update_one(
            {"symbol": symbol, "timestamp": timestamp},
            {"$set": {"indicators": features}}
        )
        updates += result.modified_count

    print(f"✅ {symbol}: Updated {updates} documents with indicators.")

# --- Loop over all coins ---
for coin in coin_symbols:
    add_indicators_to_symbol(coin)
