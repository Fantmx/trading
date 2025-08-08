from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
coll = client["CoinCluster"]["binance_price_data"]

# Find slash-format symbols
for sym in coll.distinct("symbol"):
    if "/" in sym:
        new_sym = sym.replace("/", "")
        result = coll.update_many({"symbol": sym}, {"$set": {"symbol": new_sym}})
        print(f"Updated {result.modified_count} docs: {sym} → {new_sym}")
