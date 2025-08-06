import os
import requests
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# Load Mongo URI
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["trading"]
collection = db["price_data"]

# All 3 tiers of cryptos with CoinGecko IDs
coins = {
    "tier_1": {
        "BTC/USDT": "bitcoin",
        "ETH/USDT": "ethereum",
        "BNB/USDT": "binancecoin",
        "SOL/USDT": "solana",
        "ADA/USDT": "cardano",
    },
    "tier_2": {
        "XRP/USDT": "ripple",
        "AVAX/USDT": "avalanche-2",
        "DOGE/USDT": "dogecoin",
        "MATIC/USDT": "matic-network",
        "DOT/USDT": "polkadot",
        "LTC/USDT": "litecoin",
    },
    "tier_3": {
        "SHIB/USDT": "shiba-inu",
        "PEPE/USDT": "pepe",
        "APT/USDT": "aptos",
        "INJ/USDT": "injective-protocol",
        "FET/USDT": "fetch-ai",
    }
}

# Fetch CoinGecko hourly data (up to 90 days)
def fetch_hourly_data(coin_id, vs_currency="usd", days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "hourly"
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    return res.json()

# Save to MongoDB with upsert
def insert_hourly_prices(symbol, coin_id):
    print(f"Fetching: {symbol}")
    data = fetch_hourly_data(coin_id)
    prices = data.get("prices", [])
    volumes = {int(x[0]): x[1] for x in data.get("total_volumes", [])}

    ops = []
    for ts_ms, close in prices:
        ts = datetime.utcfromtimestamp(ts_ms / 1000)
        volume = volumes.get(ts_ms, None)
        doc = {
            "symbol": symbol,
            "timestamp": ts,
            "close": close,
            "volume": volume,
            "interval": "1h",
            "source": "coingecko"
        }
        ops.append(UpdateOne(
            {"symbol": symbol, "timestamp": ts},
            {"$set": doc},
            upsert=True
        ))

    if ops:
        result = collection.bulk_write(ops)
        print(f"{symbol}: Inserted: {result.upserted_count}, Updated: {result.modified_count}")

# Main loop
def main():
    for tier, assets in coins.items():
        print(f"\n--- {tier.upper()} ---")
        for symbol, coin_id in assets.items():
            try:
                insert_hourly_prices(symbol, coin_id)
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

if __name__ == "__main__":
    main()
