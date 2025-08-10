import os
import requests
import time
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
print(COINGECKO_API_KEY)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["CoinCluster"]
collection = db["price_data"]

# All 3 tiers of coins
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

# Fetch hourly data from CoinGecko
def fetch_hourly_data(coin_id, vs_currency="usd", days=90, retries=3):
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        f"?vs_currency={vs_currency}&days={days}"
        f"&x_cg_demo_api_key={COINGECKO_API_KEY}"
    )

    for attempt in range(retries):
        try:
            res = requests.get(url)
            print(f"\n[DEBUG] {coin_id} response ({res.status_code}): {res.text[:500]}")
            res.raise_for_status()
            return res.json()
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] {coin_id}: {e}")
            break
        except Exception as e:
            print(f"[ERROR] {coin_id}: {e}")
            break

    return None



# Insert price data into MongoDB
def insert_hourly_prices(symbol, coin_id):
    print(f"Fetching: {symbol}")
    data = fetch_hourly_data(coin_id)
    if not data:
        print(f"[SKIPPED] {symbol}: No data retrieved.")
        return

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
            insert_hourly_prices(symbol, coin_id)
            time.sleep(5)  # prevent rate limits

if __name__ == "__main__":
    main()
