# paper_executor.py — unified paper trading (SINGLE / RISK / FOLLOW)

import os
from datetime import timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------- .env ----------
def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p):
            return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")

load_dotenv(dotenv_path=_find_env_path())

# ---------- Mode selection ----------
PAPER_MODE = (os.getenv("PAPER_MODE") or "SINGLE").upper().strip()
# Allowed: SINGLE, RISK, FOLLOW

# ---------- Shared config ----------
DB_NAME        = os.getenv("MONGO_DB", "CoinCluster")
PRICE_COLL     = os.getenv("MONGO_COLLECTION", "binance_price_data")
SIGNAL_COLL    = os.getenv("SIGNAL_COLLECTION", "signals")
TRADES_COLL    = os.getenv("TRADES_COLLECTION", "paper_trades")

# Costs
FEE_BP         = float(os.getenv("FEE_BP", "5"))      # per side (bps)
SLIPPAGE_BP    = float(os.getenv("SLIPPAGE_BP", "0")) # optional extra, bps

# SINGLE‑mode sizing
USD_PER_TRADE  = float(os.getenv("USD_PER_TRADE", "100"))

# RISK‑mode & FOLLOW‑mode state
OPEN_POS_COLL  = os.getenv("OPEN_POS_COLL", "open_positions")
ACCOUNT_COLL   = os.getenv("ACCOUNT_COLL", "account")
START_EQUITY   = float(os.getenv("START_EQUITY", "10000"))
RISK_PCT       = float(os.getenv("RISK_PCT", "0.015"))   # 1.5% per trade
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
HOLD_HOURS     = int(os.getenv("HOLD_HOURS", "6"))       # RISK expiry
SL_ATR_MULT    = float(os.getenv("SL_ATR_MULT", "1.5"))  # RISK stop
TP_ATR_MULT    = float(os.getenv("TP_ATR_MULT", "2.0"))  # RISK take

# FOLLOW‑mode trailing stop options
TRAIL_TYPE     = (os.getenv("TRAIL_TYPE") or "NONE").upper().strip()  # ATR | PCT | NONE
TRAIL_MULT     = float(os.getenv("TRAIL_MULT", "2.0"))   # ATR multiple if TRAIL_TYPE=ATR
TRAIL_PCT      = float(os.getenv("TRAIL_PCT", "0.05"))   # 5% if TRAIL_TYPE=PCT
FOLLOW_MAX_HRS = int(os.getenv("FOLLOW_MAX_HOURS", "0")) # optional max hold; 0 = unlimited

# ---------- DB ----------
db       = MongoClient(os.getenv("MONGO_URI"))[DB_NAME]
prices   = db[PRICE_COLL]
signals  = db[SIGNAL_COLL]
trades   = db[TRADES_COLL]
openpos  = db[OPEN_POS_COLL]
account  = db[ACCOUNT_COLL]

# ---------- Utils ----------
def _utc(ts):
    return ts if getattr(ts, "tzinfo", None) is not None else ts.replace(tzinfo=timezone.utc)

def _fees_roundtrip_bp():
    # entry + exit + optional slippage once
    return 2 * FEE_BP + SLIPPAGE_BP

def _fees_roundtrip_frac():
    return _fees_roundtrip_bp() / 10000.0

def _get_bar(symbol, ts):
    return prices.find_one({"symbol": symbol, "timestamp": ts})

def _get_next_bar(symbol, ts):
    return prices.find_one({"symbol": symbol, "timestamp": ts + timedelta(hours=1)})

def _close_and_atr(bar):
    if not bar:
        return None, None
    close = float(bar["close"])
    ind = bar.get("indicators") or {}
    atr = ind.get("atr_14")
    try:
        atr = None if atr is None else float(atr)
    except:
        atr = None
    return close, atr

# ---------- Account helpers (RISK/FOLLOW) ----------
def _get_or_init_account():
    doc = account.find_one({"_id": "default"})
    if doc:
        return doc
    account.insert_one({"_id": "default", "starting_equity": START_EQUITY, "equity": START_EQUITY})
    return account.find_one({"_id": "default"})

def _equity():
    return (_get_or_init_account()).get("equity", START_EQUITY)

def _equity_add(delta):
    account.update_one(
        {"_id": "default"},
        [
            {
                "$set": {
                    "starting_equity": {"$ifNull": ["$starting_equity", START_EQUITY]},
                    "equity": {
                        "$add": [
                            {"$ifNull": ["$equity", {"$ifNull": ["$starting_equity", START_EQUITY]}]},
                            float(delta)
                        ]
                    }
                }
            }
        ],
        upsert=True
    )

def _concurrent_positions():
    return openpos.count_documents({})

# ---------- Trade booking ----------
def _book_trade(symbol, entry_ts, side, px_in, px_out, qty, reason, fill_time):
    fees_rt = _fees_roundtrip_frac()
    gross_ret = (px_out - px_in) / px_in if side == "LONG" else (px_in - px_out) / px_in
    net_ret = gross_ret - fees_rt
    pnl = qty * px_in * net_ret

    trades.insert_one({
        "symbol": symbol,
        "timestamp": entry_ts,      # signal/entry bar close
        "side": side,
        "px_in": float(px_in),
        "px_out": float(px_out),
        "qty": float(qty),
        "gross_ret": float(gross_ret),
        "net_ret": float(net_ret),
        "reason": reason,
        "fill_time": fill_time
    })
    return pnl

# ---------- Mode: SINGLE (v1) ----------
def run_single_mode():
    executed = 0
    skipped = 0
    for sig in signals.find({}, sort=[("timestamp", 1)]):
        symbol = sig["symbol"]
        ts = _utc(sig["timestamp"])

        bar = _get_bar(symbol, ts)
        next_bar = _get_next_bar(symbol, ts)
        if not bar or not next_bar:
            skipped += 1
            continue

        side = "LONG" if sig["decision"] == "UP" else "SHORT"
        px_in = float(bar["close"])
        px_out = float(next_bar["close"])
        qty = USD_PER_TRADE / px_in

        pnl = _book_trade(symbol, ts, side, px_in, px_out, qty, reason="SINGLE_NEXT_CLOSE", fill_time=ts + timedelta(hours=1))
        # In SINGLE mode we don't track account; it’s a per-trade notional
        executed += 1
    print(f"[PAPER:SINGLE] executed {executed} signals (skipped no bars: {skipped}).")

# ---------- Mode: RISK (v2) ----------
def _risk_exit_if_needed(symbol, pos, next_bar, decision, ts):
    """Check SL/TP/expiry/opposite on next bar; return (exited, pnl)."""
    side   = pos["side"]
    px_in  = float(pos["entry_px"])
    qty    = float(pos["qty"])
    stop   = float(pos["stop_px"])
    take   = float(pos["take_px"])
    expiry = _utc(pos["expiry_ts"])
    hi = float(next_bar["high"])
    lo = float(next_bar["low"])
    next_close = float(next_bar["close"])

    reason = None
    exit_px = None
    # SL/TP priority using next bar H/L
    if side == "LONG":
        if lo <= stop:
            reason, exit_px = "STOP", stop
        elif hi >= take:
            reason, exit_px = "TAKE", take
    else:
        if hi >= stop:
            reason, exit_px = "STOP", stop
        elif lo <= take:
            reason, exit_px = "TAKE", take

    # If neither hit, check expiry/opposite → fill at next close
    if reason is None:
        if ts >= expiry:
            reason, exit_px = "EXPIRY", next_close
        elif (decision == "UP" and side == "SHORT") or (decision == "DOWN" and side == "LONG"):
            reason, exit_px = "OPPOSITE", next_close

    if reason:
        pnl = _book_trade(symbol, pos["entry_ts"], side, px_in, float(exit_px), qty, reason, fill_time=ts + timedelta(hours=1))
        openpos.delete_one({"_id": pos["_id"]})
        _equity_add(pnl)
        return True, pnl
    return False, 0.0

def run_risk_mode():
    processed = 0
    for sig in signals.find({}, sort=[("timestamp", 1)]):
        symbol = sig["symbol"]
        ts = _utc(sig["timestamp"])

        this_bar = _get_bar(symbol, ts)
        next_bar = _get_next_bar(symbol, ts)
        if not this_bar or not next_bar:
            continue

        decision = sig["decision"]
        # 1) manage existing position first
        pos = openpos.find_one({"symbol": symbol})
        if pos:
            exited, _ = _risk_exit_if_needed(symbol, pos, next_bar, decision, ts)
            if exited:
                # after closing, we may consider opening a new one below
                pass
            else:
                # still open → skip any new open
                processed += 1
                continue

        # 2) possibly open new position (risk budget + ATR needed)
        if _concurrent_positions() >= MAX_CONCURRENT:
            processed += 1
            continue

        _, atr = _close_and_atr(next_bar)
        if atr is None or atr <= 0:
            processed += 1
            continue

        equity = _equity()
        usd_risk = equity * RISK_PCT
        if usd_risk <= 0:
            processed += 1
            continue

        entry_px = float(next_bar["close"])
        qty = usd_risk / entry_px

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
            "expiry_ts": ts + timedelta(hours=HOLD_HOURS),
            "mode": "RISK"
        })
        processed += 1

    print(f"[PAPER:RISK] processed {processed} signal bars.")

# ---------- Mode: FOLLOW (follow until opposite + optional trailing stop) ----------
def _update_trailing(side, trail_type, trail_mult, trail_pct, pos, next_bar):
    """
    Update trailing stop given next bar high/low/close.
    For LONG: move stop up when price makes new highs.
    For SHORT: move stop down when price makes new lows.
    """
    hi = float(next_bar["high"])
    lo = float(next_bar["low"])
    close = float(next_bar["close"])
    _, atr = _close_and_atr(next_bar)

    stop = float(pos.get("stop_px")) if pos.get("stop_px") is not None else None

    if trail_type == "NONE":
        return stop

    if side == "LONG":
        if trail_type == "ATR" and atr is not None and atr > 0:
            new_stop = close - trail_mult * atr
        elif trail_type == "PCT":
            new_stop = close * (1 - trail_pct)
        else:
            return stop
        # only ratchet upward
        return new_stop if (stop is None or new_stop > stop) else stop

    else:  # SHORT
        if trail_type == "ATR" and atr is not None and atr > 0:
            new_stop = close + trail_mult * atr
        elif trail_type == "PCT":
            new_stop = close * (1 + trail_pct)
        else:
            return stop
        # only ratchet downward
        return new_stop if (stop is None or new_stop < stop) else stop

def _follow_exit_if_needed(symbol, pos, next_bar, incoming_decision, ts):
    side   = pos["side"]
    px_in  = float(pos["entry_px"])
    qty    = float(pos["qty"])
    stop   = pos.get("stop_px")
    next_close = float(next_bar["close"])
    hi = float(next_bar["high"])
    lo = float(next_bar["low"])

    # 1) trailing stop check on next bar (H/L)
    if stop is not None:
        stop = float(stop)
        if side == "LONG" and lo <= stop:
            pnl = _book_trade(symbol, pos["entry_ts"], side, px_in, stop, qty, "TRAIL_STOP", ts + timedelta(hours=1))
            openpos.delete_one({"_id": pos["_id"]})
            _equity_add(pnl)
            return True

        if side == "SHORT" and hi >= stop:
            pnl = _book_trade(symbol, pos["entry_ts"], side, px_in, stop, qty, "TRAIL_STOP", ts + timedelta(hours=1))
            openpos.delete_one({"_id": pos["_id"]})
            _equity_add(pnl)
            return True

    # 2) opposite signal → exit at next close
    if (incoming_decision == "UP" and side == "SHORT") or (incoming_decision == "DOWN" and side == "LONG"):
        pnl = _book_trade(symbol, pos["entry_ts"], side, px_in, next_close, qty, "OPPOSITE", ts + timedelta(hours=1))
        openpos.delete_one({"_id": pos["_id"]})
        _equity_add(pnl)
        return True

    # 3) optional max hold
    if FOLLOW_MAX_HRS > 0 and ts >= _utc(pos["entry_ts"]) + timedelta(hours=FOLLOW_MAX_HRS):
        pnl = _book_trade(symbol, pos["entry_ts"], side, px_in, next_close, qty, "MAX_HOLD", ts + timedelta(hours=1))
        openpos.delete_one({"_id": pos["_id"]})
        _equity_add(pnl)
        return True

    return False

def run_follow_mode():
    processed = 0
    for sig in signals.find({}, sort=[("timestamp", 1)]):
        symbol = sig["symbol"]
        ts = _utc(sig["timestamp"])
        this_bar = _get_bar(symbol, ts)
        next_bar = _get_next_bar(symbol, ts)
        if not this_bar or not next_bar:
            continue

        decision = sig["decision"]

        # Manage existing first
        pos = openpos.find_one({"symbol": symbol, "mode": "FOLLOW"})
        if pos:
            exited = _follow_exit_if_needed(symbol, pos, next_bar, decision, ts)
            if not exited:
                # Update trailing stop (ratchet only)
                new_stop = _update_trailing(pos["side"], TRAIL_TYPE, TRAIL_MULT, TRAIL_PCT, pos, next_bar)
                if new_stop is not None and (pos.get("stop_px") is None or new_stop != float(pos.get("stop_px"))):
                    openpos.update_one({"_id": pos["_id"]}, {"$set": {"stop_px": float(new_stop)}})
                processed += 1
                continue
            # if exited, we may choose to open a new one below

        # If we reach here, no open FOLLOW position → consider opening
        if _concurrent_positions() >= MAX_CONCURRENT:
            processed += 1
            continue

        entry_px = float(next_bar["close"])
        # Reuse risk sizing so FOLLOW also respects account risk
        equity = _equity()
        usd_risk = equity * RISK_PCT
        if usd_risk <= 0:
            processed += 1
            continue
        qty = usd_risk / entry_px

        side = "LONG" if decision == "UP" else "SHORT"

        # Seed an initial stop if trailing is active
        _, atr = _close_and_atr(next_bar)
        init_stop = None
        if TRAIL_TYPE == "ATR" and atr and atr > 0:
            init_stop = entry_px - TRAIL_MULT * atr if side == "LONG" else entry_px + TRAIL_MULT * atr
        elif TRAIL_TYPE == "PCT":
            init_stop = entry_px * (1 - TRAIL_PCT) if side == "LONG" else entry_px * (1 + TRAIL_PCT)

        openpos.insert_one({
            "symbol": symbol,
            "mode": "FOLLOW",
            "side": side,
            "entry_ts": ts,
            "entry_px": float(entry_px),
            "qty": float(qty),
            "usd_risk": float(usd_risk),
            "stop_px": (None if init_stop is None else float(init_stop))
        })
        processed += 1

    print(f"[PAPER:FOLLOW] processed {processed} signal bars.")

# ---------- Main ----------
def main():
    if PAPER_MODE == "SINGLE":
        run_single_mode()
    elif PAPER_MODE == "RISK":
        run_risk_mode()
    elif PAPER_MODE == "FOLLOW":
        run_follow_mode()
    else:
        raise SystemExit(f"Unknown PAPER_MODE={PAPER_MODE}. Use SINGLE | RISK | FOLLOW.")

if __name__ == "__main__":
    main()
