import os
from datetime import timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

# robust env loader (same as before)
def _find_env_path():
    import os
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p): return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")
load_dotenv(dotenv_path=_find_env_path())

DB_NAME       = os.getenv("MONGO_DB", "CoinCluster")
PRICE_COLL    = os.getenv("MONGO_COLLECTION", "binance_price_data")
SIGNAL_COLL   = os.getenv("SIGNAL_COLLECTION", "signals")
TRADES_COLL   = os.getenv("TRADES_COLLECTION", "paper_trades")
POS_COLL      = os.getenv("POS_COLLECTION", "positions")
FEE_BP        = float(os.getenv("FEE_BP", "5"))          # per side, bps
USD_PER_TRADE = float(os.getenv("USD_PER_TRADE", "100"))

db = MongoClient(os.getenv("MONGO_URI"))[DB_NAME]
prices  = db[PRICE_COLL]
signals = db[SIGNAL_COLL]
trades  = db[TRADES_COLL]
pos     = db[POS_COLL]

# health (optional)
def _log_db_health(db, collection_name):
    c = db[collection_name]
    try:
        count = c.estimated_document_count()
        symbols = sorted(c.distinct("symbol"))
    except Exception as e:
        print(f"[HEALTH] Error reading DB health: {e}")
        return
    print(f"[HEALTH] Connected DB='{db.name}', Coll='{collection_name}'  Docs~{count:,}")
    print(f"[HEALTH] Active symbols ({len(symbols)}): {symbols[:8]}{'...' if len(symbols)>8 else ''}")
_log_db_health(db, PRICE_COLL)

def execute_for_bar(symbol, ts):
    sig = signals.find_one({"symbol": symbol, "timestamp": ts})
    if not sig: return
    bar = prices.find_one({"symbol": symbol, "timestamp": ts})
    next_bar = prices.find_one({"symbol": symbol, "timestamp": ts + timedelta(hours=1)})
    if not bar or not next_bar: return

    side = "LONG" if sig["decision"] == "UP" else "SHORT"
    px_in, px_out = float(bar["close"]), float(next_bar["close"])
    qty = USD_PER_TRADE / px_in

    gross_ret = (px_out - px_in)/px_in if side=="LONG" else (px_in - px_out)/px_in
    fees = 2 * (FEE_BP/10000.0)  # entry + exit
    net_ret = gross_ret - fees

    trades.insert_one({
        "symbol": symbol, "timestamp": ts, "side": side,
        "px_in": px_in, "px_out": px_out, "qty": qty,
        "gross_ret": gross_ret, "net_ret": net_ret
    })

    pos.update_one({"symbol": symbol},
                   {"$inc": {"equity": USD_PER_TRADE * net_ret},
                    "$setOnInsert": {"equity": 10000}}, upsert=True)

def main():
    # Execute all unexecuted signals that have a next bar available
    todo = signals.find({}, sort=[("timestamp", 1)])
    count = 0
    for sig in todo:
        s, ts = sig["symbol"], sig["timestamp"]
        if trades.find_one({"symbol": s, "timestamp": ts}):
            continue  # already executed
        # require both this bar and next bar
        if not prices.find_one({"symbol": s, "timestamp": ts}):
            continue
        if not prices.find_one({"symbol": s, "timestamp": ts + timedelta(hours=1)}):
            continue
        execute_for_bar(s, ts)
        count += 1
    print(f"[PAPER] executed {count} signals")

if __name__ == "__main__":
    # no runtime_state gating here so backlog can clear even if ingest==0 this run
    main()
