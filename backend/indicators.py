import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
collection = db["binance_price_data"]

def add_indicators_to_symbol(symbol="BTC/USDT"):
    # Step 1: Load data from MongoDB
    cursor = collection.find({"symbol": symbol}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print(f"No data for {symbol}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Step 2: Add technical indicators
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    # RSI Calculation
    # RSI Calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))


    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Step 3: Write results back to MongoDB
    for timestamp, row in df.iterrows():
        features = {
            "sma_10": round(row["sma_10"], 6) if pd.notnull(row["sma_10"]) else None,
            "sma_50": round(row["sma_50"], 6) if pd.notnull(row["sma_50"]) else None,
            "ema_10": round(row["ema_10"], 6) if pd.notnull(row["ema_10"]) else None,
            "rsi_14": round(row["rsi_14"], 6) if pd.notnull(row["rsi_14"]) else None,
            "macd": round(row["macd"], 6) if pd.notnull(row["macd"]) else None,
            "macd_signal": round(row["macd_signal"], 6) if pd.notnull(row["macd_signal"]) else None
        }

        collection.update_one(
            {"symbol": symbol, "timestamp": timestamp},
            {"$set": {"indicators": features}}
        )

    print(f"Updated {symbol} with indicators.")

# Coin list: symbol -> CoinGecko ID (not used here but ready for future use)
coins = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "BNB/USDT": "binancecoin",
    "SOL/USDT": "solana",
    "ADA/USDT": "cardano",
    "XRP/USDT": "ripple",
    "AVAX/USDT": "avalanche-2",
    "DOGE/USDT": "dogecoin",
    "MATIC/USDT": "matic-network",
    "DOT/USDT": "polkadot",
    "LTC/USDT": "litecoin",
    "SHIB/USDT": "shiba-inu",
    "PEPE/USDT": "pepe",
    "APT/USDT": "aptos",
    "INJ/USDT": "injective-protocol",
    "FET/USDT": "fetch-ai"
}

# Batch processing loop
for symbol in coins.keys():
    try:
        print(f"\nProcessing {symbol}...")
        add_indicators_to_symbol(symbol)
    except Exception as e:
        print(f"[ERROR] Failed to process {symbol}: {e}")
