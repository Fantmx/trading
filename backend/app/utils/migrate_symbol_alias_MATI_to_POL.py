# migrate_symbol_alias.py
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import os

SOURCE_SYMBOL = "MATICUSDT"
TARGET_SYMBOL = "POLUSDT"

def find_env_path() -> str:
    """Walk up the directory tree until we find backend/configs/.env"""
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):  # look up to 6 levels just in case
        candidate = os.path.join(cur, "configs", ".env")
        if os.path.isfile(candidate):
            return candidate
        cur = os.path.dirname(cur)
    raise FileNotFoundError("Could not find configs/.env by walking parents.")

def main():
    # Load .env from backend/configs/.env (found by walking up parents)
    env_path = find_env_path()
    load_dotenv(dotenv_path=env_path)

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set. Check configs/.env")

    client = MongoClient(mongo_uri)
    db_name = os.getenv("MONGO_DB", "CoinCluster")
    coll_name = os.getenv("MONGO_COLLECTION", "binance_price_data")
    db = client[db_name]
    col = db[coll_name]

    print(f"[MIGRATE] Using DB='{db_name}' Collection='{coll_name}' URI='{mongo_uri.split('@')[-1]}'")
    print(f"[MIGRATE] Copying {SOURCE_SYMBOL} → {TARGET_SYMBOL} if not present...")

    src_count = col.count_documents({"symbol": SOURCE_SYMBOL})
    print(f"[MIGRATE] Found {src_count} docs for {SOURCE_SYMBOL}")

    if src_count == 0:
        print("[MIGRATE] Nothing to do. If this seems wrong, double-check MONGO_URI/DB/COLLECTION.")
        return

    ops = []
    # Stream to avoid pulling entire collection into RAM
    for doc in col.find({"symbol": SOURCE_SYMBOL}, projection={"_id": 0}):
        ts = doc["timestamp"]
        # only insert if target doesn't already have this timestamp
        if col.count_documents({"symbol": TARGET_SYMBOL, "timestamp": ts}, limit=1) == 0:
            new_doc = dict(doc)
            new_doc["symbol"] = TARGET_SYMBOL
            ops.append(UpdateOne(
                {"symbol": TARGET_SYMBOL, "timestamp": ts},
                {"$setOnInsert": new_doc},
                upsert=True
            ))
            # flush in batches
            if len(ops) >= 1000:
                res = col.bulk_write(ops, ordered=False)
                print(f"[MIGRATE] Batch inserted: {res.upserted_count or 0}")
                ops.clear()

    if ops:
        res = col.bulk_write(ops, ordered=False)
        print(f"[MIGRATE] Final batch inserted: {res.upserted_count or 0}")

    print("[MIGRATE] Done.")

if __name__ == "__main__":
    main()
