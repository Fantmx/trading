from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["CoinCluster"]
collection = db["binance_price_data"]

# Check how many documents have bb_upper and bb_lower
missing_bb = collection.count_documents({"bb_upper": {"$exists": False}})
print(f"Documents missing bb_upper and bb_lower: {missing_bb}")
