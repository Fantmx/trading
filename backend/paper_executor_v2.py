# paper_executor_v2.py — risk-based sizing + multi-hour holds + ATR stops

import os
from datetime import timedelta, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# ---------- .env ----------
def _find_env_path():
    import os
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p): return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")

load_dotenv(dotenv_path=_find_env_path())

# ---------- Config ----------
DB_NAME            = os.getenv("MONGO_DB", "CoinCluster")
PRICE_COLL         = os.getenv("MONGO_COLLECTION", "binance_price_data")
SIGNAL_COLL        = os.getenv("SIGNAL_COLLECTION", "signals")
TRADES_COLL        = os.getenv("TRADES_COLLECTION", "paper_trades")
OPEN_POS_COLL      = os.getenv("OPEN_POS_COLL", "open_positions")
ACCOUNT_COLL       = os.getenv("ACCOUNT_COLL", "account")

# Account / risk
START_EQUITY       = float(os.getenv("START_EQUITY", "1000"))  # total shared account
RISK_PCT           = float(os.getenv("RISK_PCT", "0.015"))     # 1.5% per trade
MAX_CONCURRENT     = int(os.getenv("MAX_CONCURRENT", "5"))     # cap simultaneous positions

# Trade mgmt
HOLD_HOURS         = int(os.getenv("HOLD_HOURS", "6"))         # max holding time
SL_ATR_MULT        = float(os.getenv("SL_ATR_MULT", "1.5"))    # stop = 1.5 * ATR
TP_ATR_MULT        = float(os.getenv("TP_ATR_MULT", "2.0"))    # take = 2.0 * ATR

# Costs
FEE_BP             = float(os.getenv("FEE_BP", "5"))           # per side (bps)
SLIPPAGE_BP        = float(os.getenv("SLIPPAGE_BP", "5"))      # applied once per round-trip (bps)

# ---------- DB ----------
db = MongoClient(os.getenv("MONGO_URI"))[DB_NAME]
prices   = db[PRICE_COLL]
signals  = db[SIGNAL_COLL]
trades   = db[TRADES_COLL]
openpos  = db[OPEN_POS_COLL]
account  = db[ACCOUNT_COLL]

# ---------- Utils ----------
def _utc(ts):
    return ts if getattr(ts, "tzinfo", None) is not None else ts.replace(tzinfo=timezone.utc)

def _fees_roundtrip():
    return 2 * (FEE_BP / 10000.0) + (SLIPPAGE_BP / 10000.0)

def _get_bar(symbol, ts):
    return prices.find_one({"symbol": symbol, "timestamp": ts})

def _get_next_bar(symbol, ts):
    return prices.find_one({"symbol": symbol, "timestamp": ts + timedelta(hours=1)})

def _latest_indicators(bar):
    """Return (close, atr_14) from merged price+indicator doc."""
    if not bar: return None, None
    close = float(bar["close"])
    ind = bar.get("indicators") or {}
    atr = ind.get("atr_14")
    if atr is not None:
        try: atr = float(atr)
        except: atr = None
    return close, atr

def _get_or_init_account():
    doc = account.find_one({"_id": "default"})
    if doc: return doc
    account.insert_one({"_id": "default", "starting_equity": START_EQUITY, "equity": START_EQUITY})
    return account.find_one({"_id": "default"})

def _account_equity():
    return (_get_or_init_account()).get("equity", START_EQUITY)

def _update_equity(delta):
    # atomic increment; initialize if missing
    account.update_one(
        {"_id": "default"},
        [
            {
                "$set": {
                    "starting_equity": {"$ifNull": ["$starting_equity", START_EQUITY]},
                    "equity": { "$add": [ { "$ifNull": ["$equity", {"$ifNull": ["$starting_equity", START_EQUITY]}] }, float(delta) ] }
                }
            }
        ],
        upsert=True
    )

def _concurrent_positions_count():
    return openpos.count_documents({})

def _exit_trade(symbol, entry_ts, side, entry_px, exit_px, qty, reason, ts_fill, fees_bp_roundtrip):
    gross_ret = (exit_px - entry_px)/entry_px if side == "LONG" else (entry_px - exit_px)/entry_px
    net_ret = gross_ret - fees_bp_roundtrip
    pnl_usd = qty * entry_px * net_ret

    trades.insert_one({
        "symbol": symbol,
        "timestamp": entry_ts,         # signal/entry decision bar close
        "side": side,
        "px_in": entry_px,
        "px_out": exit_px,
        "qty": qty,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "reason": reason,
        "fill_time": ts_fill
    })
    _update_equity(pnl_usd)

# ---------- Core logic ----------
def process_signal(symbol, ts):
    """At signal bar close ts, decide open/close actions and fill on next bar close (ts+1h)."""
    ts = _utc(ts)
    sig = signals.find_one({"symbol": symbol, "timestamp": ts})
    if not sig:
        return

    this_bar = _get_bar(symbol, ts)
    next_bar = _get_next_bar(symbol, ts)
    if not this_bar or not next_bar:
        return

    decision = sig["decision"]  # "UP" or "DOWN"
    next_close, next_atr = _latest_indicators(next_bar)
    if next_close is None:
        return

    # 1) Manage any existing open position FIRST (opposite, expiry, SL/TP on next bar H/L)
    pos = openpos.find_one({"symbol": symbol})
    if pos:
        side = pos["side"]
        entry_px = float(pos["entry_px"])
        qty = float(pos["qty"])
        expiry_ts = pos["expiry_ts"]
        stop_px = float(pos["stop_px"])
        take_px = float(pos["take_px"])
        fees_rt = _fees_roundtrip()

        # Evaluate next bar for exits
        hi = float(next_bar["high"])
        lo = float(next_bar["low"])
        exit_reason = None
        exit_px = None

        # Priority: stop-loss / take-profit using next bar high/low
        if side == "LONG":
            if lo <= stop_px:      # stopped
                exit_px = stop_px
                exit_reason = "STOP"
            elif hi >= take_px:    # take-profit
                exit_px = take_px
                exit_reason = "TAKE"
        else:  # SHORT
            if hi >= stop_px:
                exit_px = stop_px
                exit_reason = "STOP"
            elif lo <= take_px:
                exit_px = take_px
                exit_reason = "TAKE"

        # If no SL/TP hit, check expiry or opposite signal; fill at next close
        if exit_reason is None:
            if ts >= _utc(expiry_ts):
                exit_reason = "EXPIRY"
                exit_px = next_close
            elif (decision == "UP" and side == "SHORT") or (decision == "DOWN" and side == "LONG"):
                exit_reason = "OPPOSITE"
                exit_px = next_close

        if exit_reason:
            _exit_trade(symbol, pos["entry_ts"], side, entry_px, float(exit_px), qty, exit_reason, ts + timedelta(hours=1), fees_rt)
            openpos.delete_one({"_id": pos["_id"]})
            # After closing, reload decision logic for opening a new one (fall through)

        else:
            # Still open; nothing else to do for this symbol at this ts
            return

    # 2) Consider opening a new position (only if within risk budget / max concurrent)
    if _concurrent_positions_count() >= MAX_CONCURRENT:
        return

    # Use ATR from next bar to set SL/TP; if no ATR, skip
    _, atr = _latest_indicators(next_bar)
    if atr is None or atr <= 0:
        return

    equity = _account_equity()
    usd_risk = equity * RISK_PCT
    if usd_risk <= 0:
        return

    entry_px = next_close
    qty = usd_risk / entry_px

    # ATR-based levels
    if decision == "UP":
        side = "LONG"
        stop_px = entry_px - SL_ATR_MULT * atr
        take_px = entry_px + TP_ATR_MULT * atr
    else:
        side = "SHORT"
        stop_px = entry_px + SL_ATR_MULT * atr
        take_px = entry_px - TP_ATR_MULT * atr

    openpos.insert_one({
        "symbol": symbol,
        "side": side,
        "entry_ts": ts,
        "entry_px": float(entry_px),
        "qty": float(qty),
        "usd_risk": float(usd_risk),
        "stop_px": float(stop_px),
        "take_px": float(take_px),
        "expiry_ts": ts + timedelta(hours=HOLD_HOURS)
    })

def main():
    # Process all signals chronologically (same as v1), but with new rules
    executed = 0
    for sig in signals.find({}, sort=[("timestamp", 1)]):
        symbol = sig["symbol"]
        ts = _utc(sig["timestamp"])
        process_signal(symbol, ts)
        executed += 1
    print(f"[PAPER_V2] processed {executed} signal bars with risk-based, multi-hour logic.")

if __name__ == "__main__":
    # Health log (optional)
    try:
        c = db[PRICE_COLL].estimated_document_count()
        syms = sorted(db[PRICE_COLL].distinct("symbol"))
        print(f"[HEALTH] DB='{db.name}', Coll='{PRICE_COLL}' Docs~{c:,}  Active symbols={len(syms)}")
    except Exception as e:
        print(f"[HEALTH] DB health error: {e}")
    main()
