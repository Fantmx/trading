import httpx
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import HTTPException
from collections import defaultdict, deque

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE_URL = os.getenv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "ADA": "cardano"
}

CACHE_FILE = "price_history_cache.json"
price_cache = {}  # { "BTC/USD": [{timestamp, price}, ...] }

async def get_price_history(symbol: str, days: int = 30, interval: str = "daily"):
    is_crypto = "/" in symbol and symbol.endswith("/USD")
    base_symbol = symbol.split("/")[0] if is_crypto else symbol
    market = symbol.split("/")[1] if is_crypto else "USD"
    cache_key = f"{symbol}_{days}_{interval}"

    # Return cached result if available
    if cache_key in price_cache:
        return price_cache[cache_key]

    if is_crypto and base_symbol in COINGECKO_IDS:
        crypto_id = COINGECKO_IDS[base_symbol]
        url = (
            f"{COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
            f"?vs_currency=usd&days={days}&interval={interval}&x_cg_demo_api_key={COINGECKO_API_KEY}"
        )

        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(url)
                res.raise_for_status()
                data = res.json()
                prices = data.get("prices", [])
        except Exception as e:
            print(f"[CoinGecko fetch failed]: {e}")
            return []

        # Format prices
        formatted = [
            {
                "timestamp": datetime.utcfromtimestamp(p[0] / 1000).strftime(
                    "%Y-%m-%d" if interval == "daily" else "%H:%M"
                ),
                "price": round(p[1], 2)
            }
            for p in prices
        ]

        price_cache[cache_key] = formatted
        return formatted

    # Fallback: Alpha Vantage
    function = "DIGITAL_CURRENCY_DAILY" if is_crypto else "TIME_SERIES_INTRADAY"
    url = (
        f"https://www.alphavantage.co/query"
        f"?function={function}&symbol={base_symbol}&market={market}&apikey={ALPHA_VANTAGE_API_KEY}"
        if is_crypto else
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY&symbol={base_symbol}&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            data = res.json()
    except Exception as e:
        print(f"[AlphaVantage fetch failed]: {e}")
        return []

    if "Note" in data or "Error Message" in data:
        raise HTTPException(status_code=502, detail=data.get("Note") or data.get("Error Message") or "Unknown API error")

    if is_crypto:
        series = data.get("Time Series (Digital Currency Daily)", {})
        formatted = [
            {
                "timestamp": ts,
                "price": float(val.get("4a. close (USD)") or val.get("4. close", 0.0))
            }
            for ts, val in list(series.items())[-days:]
            if val.get("4a. close (USD)") or val.get("4. close")
        ]
    else:
        series = data.get("Time Series (5min)", {})
        formatted = [
            {
                "timestamp": ts,
                "price": float(val.get("4. close", 0.0))
            }
            for ts, val in list(series.items())[-30:]
            if "4. close" in val
        ]

    price_cache[cache_key] = formatted
    return formatted

async def get_prediction(symbol: str):
    import random
    return {
        "action": "BUY" if random.random() > 0.5 else "SELL",
        "confidence": round(random.uniform(70, 95), 2)
    }

def load_price_cache():
    global price_cache
    try:
        with open(CACHE_FILE, "r") as f:
            price_cache = json.load(f)
            for k, v in price_cache.items():
                price_cache[k] = [
                    {"timestamp": item["timestamp"], "price": float(item["price"])}
                    for item in v
                ]
            print("[Cache] Loaded price history from file")
    except Exception as e:
        print(f"[Cache] No cache file found or error loading it: {e}")
        price_cache = {}

def save_price_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(price_cache, f, indent=2)
    except Exception as e:
        print(f"[Cache] Failed to save price history: {e}")
