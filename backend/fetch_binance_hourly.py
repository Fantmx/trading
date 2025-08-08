import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["CoinCluster"]
collection = db["binance_price_data"]

# List of symbols to fetch from Binance (USDT pairs)
COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    "XRPUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
    "LTCUSDT", "SHIBUSDT", "PEPEUSDT", "APTUSDT", "INJUSDT", "FETUSDT"
]

BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_symbol_data(symbol):
    """Fetch 2 years of hourly Binance data for a single symbol."""
    end_time = int(datetime.now().timestamp() * 1000)  # Current time in ms
    start_time = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)  # 2 years ago in ms

    all_rows = []
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": start_time,
            "limit": 1000
        }
        try:
            res = requests.get(BASE_URL, params=params)
            res.raise_for_status()
            data = res.json()

            if not data:
                break

            for row in data:
                all_rows.append({
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(row[0] / 1000),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })

            # Move to next batch
            start_time = int(data[-1][0]) + 3600000  # last timestamp + 1 hour in ms

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            break

    if all_rows:
        # Remove duplicates before insert
        existing_timestamps = set(
            doc["timestamp"] for doc in collection.find({"symbol": symbol}, {"timestamp": 1})
        )
        new_rows = [row for row in all_rows if row["timestamp"] not in existing_timestamps]

        if new_rows:
            collection.insert_many(new_rows)
            print(f"{symbol}: ✅ Inserted {len(new_rows)} new rows")
        else:
            print(f"{symbol}: ⚠️ No new rows to insert")
    else:
        print(f"{symbol}: ⚠️ No data fetched")


if __name__ == "__main__":
    start_overall = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:  # Limit to 5 threads
        futures = [executor.submit(fetch_symbol_data, coin) for coin in COINS]

        for future in as_completed(futures):
            future.result()

    print(f"⏱ Done in {round(time.time() - start_overall, 2)} seconds")
