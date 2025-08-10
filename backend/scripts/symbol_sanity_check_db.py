from pymongo import MongoClient
from dotenv import load_dotenv; import os
load_dotenv(); client = MongoClient(os.getenv("MONGO_URI"))
coll = client["CoinCluster"]["binance_price_data"]

print("Distinct symbols:", coll.distinct("symbol"))
for s in coll.distinct("symbol"):
    print(s, coll.count_documents({"symbol": s}))

