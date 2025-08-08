from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
collection = db["binance_price_data"]

print("🔍 Scanning for duplicates...")

# Step 1: Aggregate duplicates by symbol + timestamp
pipeline = [
    {
        "$group": {
            "_id": {"symbol": "$symbol", "timestamp": "$timestamp"},
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1}
        }
    },
    {
        "$match": {
            "count": {"$gt": 1}
        }
    }
]

duplicates = list(collection.aggregate(pipeline))
print(f"Found {len(duplicates)} duplicate groups.")

# Step 2: Remove duplicates, keep the first
total_deleted = 0
for group in duplicates:
    ids = group["ids"]
    ids_to_delete = ids[1:]  # Keep the first one, delete the rest
    result = collection.delete_many({"_id": {"$in": ids_to_delete}})
    total_deleted += result.deleted_count

print(f"✅ Deleted {total_deleted} duplicate documents.")
