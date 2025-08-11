# ingest_latest_binance.py — catch-up + latest snapshot
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, NamedTuple

import requests
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from pathlib import Path

# --- path bootstrap so 'app' is importable whether we run from backend/ or backend/scripts ---
_THIS = Path(__file__).resolve()
BACKEND = _THIS.parent.parent if _THIS.parent.name == "scripts" else _THIS.parent
APP_DIR = BACKEND / "app"
for p in (BACKEND, APP_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from app.utils.runtime_state import write_state
except ModuleNotFoundError:
    from utils.runtime_state import write_state

# --- robust .env loader: walk up until we find backend/configs/.env ---
def _find_env_path() -> str:
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):  # search up to 6 parent levels
        candidate = os.path.join(cur, "configs", ".env")
        if os.path.isfile(candidate):
            return candidate
        cur = os.path.dirname(cur)
    raise FileNotFoundError("Could not find configs/.env by walking parents.")

ENV_PATH = _find_env_path()
load_dotenv(dotenv_path=ENV_PATH)

# Required envs
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set (check backend/configs/.env).")

# Optional overrides (with sensible defaults)
DB_NAME = os.getenv("MONGO_DB", "CoinCluster")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "binance_price_data")

# Exchange selector: BINANCE_COM (default) or BINANCE_US
EXCHANGE = (os.getenv("EXCHANGE") or "BINANCE_COM").upper().strip()

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# --- health logger ---
def _log_db_health(db, collection_name):
    coll = db[collection_name]
    try:
        count = coll.estimated_document_count()
        symbols = sorted(coll.distinct("symbol"))
    except Exception as e:
        print(f"[HEALTH] Error reading DB health: {e}")
        return []
    sample = symbols[:8]
    print(f"[HEALTH] Connected DB='{db.name}', Coll='{collection_name}'  Docs~{count:,}")
    print(f"[HEALTH] Active symbols ({len(symbols)}): {sample}{'...' if len(symbols)>8 else ''}")
    return symbols

_log_db_health(db, COLLECTION_NAME)

# Your desired modeling universe (request-side intent)
DESIRED_SYMBOLS: List[str] = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","AVAXUSDT","DOGEUSDT",
    "MATICUSDT","DOTUSDT","LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]

# Base endpoints per exchange
if EXCHANGE == "BINANCE_US":
    BINANCE_BASES = ["https://api.binance.us"]
    WS_HINT = "wss://stream.binance.us:9443/stream"
else:
    BINANCE_BASES = ["https://api.binance.com","https://api1.binance.com","https://api3.binance.com"]
    WS_HINT = "wss://stream.binance.com:9443/stream"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "trading-app/1.0"})

class ResolvedSymbols(NamedTuple):
    requested: List[str]
    resolved: List[str]
    alias_map: Dict[str, str]  # e.g. {"MATICUSDT": "POLUSDT"}
    missing: List[str]

def _get_exchange_info(retries: int = 5, base_delay: float = 0.5) -> dict:
    last_exc = None
    for attempt in range(retries):
        for base in BINANCE_BASES:
            url = f"{base}/api/v3/exchangeInfo"
            try:
                r = SESSION.get(url, timeout=15)
                if r.status_code == 451:
                    last_exc = Exception("451 geofenced (exchangeInfo).")
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last_exc = exc
        time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch exchangeInfo: {last_exc}")

def resolve_symbols_for_exchange(desired_symbols: List[str]) -> ResolvedSymbols:
    """
    For BINANCE_US: use exchangeInfo to validate/adjust symbols.
    For BINANCE_COM: pass-through (everything exists on .com).
    """
    if EXCHANGE != "BINANCE_US":
        return ResolvedSymbols(desired_symbols, desired_symbols[:], {}, [])

    info = _get_exchange_info()
    available = {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}

    # Minimal, explicit alias rules for Binance.US
    alias_rules: Dict[str, List[str]] = {
        "MATICUSDT": ["POLUSDT"],  # Polygon on Binance.US
    }

    resolved: List[str] = []
    missing: List[str] = []
    alias_map: Dict[str, str] = {}

    for sym in desired_symbols:
        if sym in available:
            resolved.append(sym)
            continue
        aliased = False
        for candidate in alias_rules.get(sym, []):
            if candidate in available:
                resolved.append(candidate)
                alias_map[sym] = candidate
                aliased = True
                break
        if not aliased:
            missing.append(sym)

    print(f"[SYMBOLS] Exchange={EXCHANGE}  WS={WS_HINT}")
    print(f"[SYMBOLS] Requested: {desired_symbols}")
    if alias_map:
        print(f"[SYMBOLS] Aliased : {alias_map}")
    if missing:
        print(f"[SYMBOLS] Missing : {missing}")
    print(f"[SYMBOLS] Using    : {resolved}")

    return ResolvedSymbols(
        requested=desired_symbols,
        resolved=resolved,
        alias_map=alias_map,
        missing=missing,
    )

def _as_utc(dt):
    # PyMongo returns naive datetimes that are UTC by convention.
    # Make them explicitly UTC-aware so .timestamp() and .astimezone() are correct.
    return dt if (getattr(dt, "tzinfo", None) is not None) else dt.replace(tzinfo=timezone.utc)

def last_ts_ms(symbol: str) -> int:
    doc = col.find_one({"symbol": symbol}, sort=[("timestamp", -1)], projection={"timestamp": 1})
    if not doc:
        return int((datetime.now(timezone.utc).timestamp() - 400*24*3600) * 1000)
    ts = _as_utc(doc["timestamp"])
    return int(ts.timestamp() * 1000)

def fetch_klines(symbol: str, start_ms: int, limit: int = 1000):
    params = {"symbol": symbol, "interval": "1h", "startTime": start_ms, "limit": limit}
    last_err = None
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/klines"
        try:
            r = SESSION.get(url, params=params, timeout=15)
            if r.status_code == 451:
                last_err = "451 geofenced (try another mirror/VPN/node)"
                continue
            if r.status_code == 429:
                # simple backoff and retry next base
                time.sleep(1.0)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last_err = f"{r.status_code} {e}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err or "Unknown error hitting Binance")

def upsert_bars(symbol: str, klines) -> int:
    if not klines:
        return 0
    ops = []
    for k in klines:
        # k = [open_time, open, high, low, close, volume, close_time, ...]
        open_ms = int(k[0])
        doc = {
            "symbol": symbol,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "timestamp": datetime.fromtimestamp(open_ms/1000, tz=timezone.utc),
            # Keep any existing indicators; update later elsewhere
        }
        ops.append(UpdateOne(
            {"symbol": symbol, "timestamp": doc["timestamp"]},
            {"$setOnInsert": doc}, upsert=True
        ))
    if ops:
        res = col.bulk_write(ops, ordered=False)
        return (res.upserted_count or 0)
    return 0

# --- NEW: catch up fully in one run ---
def catch_up_symbol(sym: str) -> int:
    """Fetch until we're within ~1 hour of 'now' for this symbol."""
    total_added = 0
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cur_ms = last_ts_ms(sym) + 1

    # If we're already current, bail early
    if (now_ms - cur_ms) <= 60*60*1000:
        return 0

    while (now_ms - cur_ms) > 60*60*1000:
        gap_hours = max(1, int((now_ms - cur_ms) // (60*60*1000)))
        # Binance limit is 1000; pull bigger chunks if far behind
        limit = 1000 if gap_hours > 1000 else min(1000, max(50, gap_hours))

        try:
            data = fetch_klines(sym, cur_ms, limit=limit)
        except Exception as e:
            print(f"[INGEST] {sym}: {e}")
            break

        if not data:
            break

        added = upsert_bars(sym, data)
        total_added += added

        # Advance to the next bar after the last returned kline
        last_open = int(data[-1][0])
        cur_ms = last_open + 60*60*1000  # next hour open in ms

        # Courtesy pause
        time.sleep(0.1)

        # If Binance returned fewer than requested, we're likely near real-time
        if len(data) < limit:
            break

    return total_added

def _print_latest_snapshot(symbols: List[str]):
    print("[LATEST] Per-symbol newest timestamps (UTC):")
    now_utc = datetime.now(timezone.utc)
    for s in symbols:
        doc = col.find_one({"symbol": s}, sort=[("timestamp", -1)], projection={"timestamp": 1})
        if not doc:
            print(f"  {s}: (none)")
            continue
        ts = _as_utc(doc["timestamp"])
        age_h = (now_utc - ts).total_seconds() / 3600.0
        flag = "ahead" if age_h < 0 else "ago"
        print(f"  {s}: {ts.isoformat()}  (~{abs(age_h):.1f}h {flag})")

def main():
    # Resolve symbols for the chosen exchange (handles US aliasing/missing)
    resolved = resolve_symbols_for_exchange(DESIRED_SYMBOLS)
    active_symbols = resolved.resolved

    if not active_symbols:
        raise SystemExit("[INGEST] No symbols resolved for this exchange. Aborting.")

    total_upserts = 0
    for sym in active_symbols:
        try:
            added = catch_up_symbol(sym)
            print(f"[INGEST] {sym}: +{added} bars")
            total_upserts += added
        except Exception as e:
            print(f"[INGEST] {sym}: {e}")
            continue

    # Persist a simple “ingest done” marker
    write_state(total_upserts, note=f"ingest_complete_{EXCHANGE.lower()}")
    print(f"[INGEST] upserted ~{total_upserts} bars")

    # Snapshot of latest timestamps
    _print_latest_snapshot(active_symbols)

if __name__ == "__main__":
    main()
