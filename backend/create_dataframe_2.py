import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
collection = db["price_data"]

coin_symbols = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
    "XRP/USDT", "AVAX/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "LTC/USDT",
    "SHIB/USDT", "PEPE/USDT", "APT/USDT", "INJ/USDT", "FET/USDT"
]

all_data = []

for symbol in coin_symbols:
    cursor = collection.find({"symbol": symbol}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print(f"Skipping {symbol}, no data.")
        continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df.set_index("timestamp", inplace=True)

    # Flatten indicators dictionary
    indicators_df = pd.json_normalize(df["indicators"]).set_index(df.index)
    df = pd.concat([df, indicators_df], axis=1)

    # Create label: price goes up next hour?
    df["next_close"] = df["close"].shift(-1)
    df["label_up"] = (df["next_close"] > df["close"]).astype(int)

    training_df = df[[
        "symbol", "close", "volume", "sma_10", "sma_50", "ema_10",
        "rsi_14", "macd", "macd_signal", "label_up"
    ]]

    # Drop only rows missing required values (not entire row)
    training_df = training_df.dropna(subset=[
        "sma_10", "sma_50", "ema_10", "rsi_14", "macd", "macd_signal", "label_up"
    ])

    print(f"{symbol}: ✅ {len(training_df)} usable rows")
    all_data.append(training_df)

# Combine and export
final_df = pd.concat(all_data).reset_index()
final_df.to_csv("training_dataset.csv", index=False)
print("✅ Final training dataset saved as 'training_dataset.csv'")
