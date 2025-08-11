# backtest_executor_v5.py
# v5 adds:
# - Pyramiding per symbol: up to MAX_PYRAMIDS_PER_SYMBOL positions per symbol (default 3)
# - Exposure cap still enforced: MAX_GROSS_EXPOSURE (fraction of equity)
# - Breakeven trigger configurable: BE_ATR_TRIGGER (default 1.0 x ATR)
# - ATR trailing stop continues after breakeven: ATR_TRAIL_MULT
# - Optional single partial take-profit: PARTIAL_TP_ENABLE, PARTIAL_TP_FRAC, PARTIAL_TP_ATR_MULT
#
# Keeps from v4:
# - Confidence band, trend filter, per-symbol thresholds, allowlist/blacklist
# - Hourly signal re-check to extend expiry and optional EXIT_ON_DECAY
# - Risk sizing by % of current equity
#
# ENV (typical):
#   START_EQUITY=1000
#   RISK_PCT=0.03
#   MAX_CONCURRENT=8
#   MAX_GROSS_EXPOSURE=0.30
#   MAX_PYRAMIDS_PER_SYMBOL=3
#   PYR_REQUIRE_IMPROVE=1
#   PYR_ADD_ATR_STEP=0.75
#   THRESH_LONG=0.56
#   THRESH_SHORT=0.44
#   TREND_FILTER=1
#   TREND_LEN=200
#   HOLD_HOURS=4
#   RECHECK_SIGNAL=1
#   EXIT_ON_DECAY=0
#   BREAKEVEN_TRAIL=1
#   BE_ATR_TRIGGER=1.0
#   ATR_TRAIL_MULT=1.0
#   PARTIAL_TP_ENABLE=1
#   PARTIAL_TP_FRAC=0.5
#   PARTIAL_TP_ATR_MULT=1.5
#
import os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv

def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p): return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")
load_dotenv(dotenv_path=_find_env_path())

DB_NAME            = os.getenv("MONGO_DB", "CoinCluster")
COLL               = os.getenv("MONGO_COLLECTION", "binance_price_data")
MONGO_URI          = os.getenv("MONGO_URI")

BACKTEST_START_S   = os.getenv("BACKTEST_START")
BACKTEST_END_S     = os.getenv("BACKTEST_END")

UNIVERSE           = [s.strip() for s in (os.getenv("UNIVERSE","").split(",")) if s.strip()]
SYMBOL_BLACKLIST   = {s.strip().upper() for s in os.getenv("SYMBOL_BLACKLIST","").split(",") if s.strip()}
SYMBOL_ALLOWLIST   = {s.strip().upper() for s in os.getenv("SYMBOL_ALLOWLIST","").split(",") if s.strip()}

GLOBAL_MODEL_PATH  = Path("model_price_direction_xgb_v3.pkl")
FEATURE_ORDER_PATH = Path("model_features_v3.json")

FEATURES_27 = [
    "close","volume","sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","atr_norm","bb_upper","bb_lower","bb_middle","bb_width","bb_pctb",
    "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10",
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    "ret_vol_6h","ret_vol_12h","ret_vol_24h"
]

START_EQUITY       = float(os.getenv("START_EQUITY", "1000"))
RISK_PCT           = float(os.getenv("RISK_PCT", "0.03"))
HOLD_HOURS         = int(os.getenv("HOLD_HOURS", "4"))
SL_ATR_MULT        = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT        = float(os.getenv("TP_ATR_MULT", "2.0"))
FEE_BP             = float(os.getenv("FEE_BP", "5"))
SLIPPAGE_BP        = float(os.getenv("SLIPPAGE_BP", "5"))
MAX_CONCURRENT     = int(os.getenv("MAX_CONCURRENT", "8"))
MAX_GROSS_EXPOSURE = float(os.getenv("MAX_GROSS_EXPOSURE", "0.30"))

THRESH_LONG        = float(os.getenv("THRESH_LONG", "0.56"))
THRESH_SHORT       = float(os.getenv("THRESH_SHORT", "0.44"))
PER_SYMBOL_THRESH_PATH = os.getenv("PER_SYMBOL_THRESH_PATH", "per_symbol_thresholds.json")

TREND_FILTER       = os.getenv("TREND_FILTER", "1") not in ("0","false","False","no","NO")
TREND_LEN          = int(os.getenv("TREND_LEN", "200"))

RECHECK_SIGNAL     = os.getenv("RECHECK_SIGNAL", "1") not in ("0","false","False","no","NO")
EXIT_ON_DECAY      = os.getenv("EXIT_ON_DECAY", "0") not in ("0","false","False","no","NO")

BREAKEVEN_TRAIL    = os.getenv("BREAKEVEN_TRAIL", "1") not in ("0","false","False","no","NO")
BE_ATR_TRIGGER     = float(os.getenv("BE_ATR_TRIGGER", "1.0"))
ATR_TRAIL_MULT     = float(os.getenv("ATR_TRAIL_MULT", "1.0"))

# Pyramiding controls
MAX_PYRAMIDS_PER_SYMBOL = int(os.getenv("MAX_PYRAMIDS_PER_SYMBOL", "3"))
PYR_REQUIRE_IMPROVE     = os.getenv("PYR_REQUIRE_IMPROVE", "1") not in ("0","false","False","no","NO")
PYR_ADD_ATR_STEP        = float(os.getenv("PYR_ADD_ATR_STEP", "0.75"))

# Partial take-profit
PARTIAL_TP_ENABLE   = os.getenv("PARTIAL_TP_ENABLE", "1") not in ("0","false","False","no","NO")
PARTIAL_TP_FRAC     = float(os.getenv("PARTIAL_TP_FRAC", "0.5"))
PARTIAL_TP_ATR_MULT = float(os.getenv("PARTIAL_TP_ATR_MULT", "1.5"))

def _utc(dt):
    return dt if getattr(dt, "tzinfo", None) else dt.replace(tzinfo=timezone.utc)

def _fees_rt():
    return 2*(FEE_BP/10000.0) + (SLIPPAGE_BP/10000.0)

def _load_feature_order():
    if FEATURE_ORDER_PATH.exists():
        try:
            names = json.loads(FEATURE_ORDER_PATH.read_text())
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return names
        except Exception:
            pass
    return FEATURES_27[:]

def _attach_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_norm"] = df["atr_14"] / df["close"]
    bw = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = bw / df["close"]
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / bw
    df["ret_1h"]  = df["close"].pct_change(1)
    df["ret_3h"]  = df["close"].pct_change(3)
    df["ret_6h"]  = df["close"].pct_change(6)
    df["ret_12h"] = df["close"].pct_change(12)
    df["ret_24h"] = df["close"].pct_change(24)
    r = df["close"].pct_change()
    df["ret_vol_6h"]  = r.rolling(6).std()
    df["ret_vol_12h"] = r.rolling(12).std()
    df["ret_vol_24h"] = r.rolling(24).std()
    df[f"sma_{TREND_LEN}"] = df["close"].rolling(TREND_LEN).mean()
    return df

def _build_row_vec(row, feature_order):
    vec = []
    for f in feature_order:
        val = row.get(f, np.nan)
        if isinstance(val, (float, int)) and (np.isnan(val) or np.isinf(val)):
            return None
        vec.append(float(val))
    return np.array(vec, dtype=float)

def _load_per_symbol_thresholds(path: str) -> Dict[str, Tuple[float,float]]:
    m = {}
    p = Path(path)
    if not p.exists():
        return m
    try:
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            for k,v in raw.items():
                k2 = k.upper().strip()
                if isinstance(v, (int,float)):
                    m[k2] = (float(v), 1.0 - float(v))
                elif isinstance(v, dict):
                    lo = float(v.get("long", THRESH_LONG))
                    sh = float(v.get("short", THRESH_SHORT))
                    m[k2] = (lo, sh)
    except Exception as e:
        print(f"[WARN] Failed to read per-symbol thresholds: {e}")
    return m

# ---------- Load data ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLL]

if BACKTEST_START_S and BACKTEST_END_S:
    START = _utc(datetime.fromisoformat(BACKTEST_START_S))
    END   = _utc(datetime.fromisoformat(BACKTEST_END_S))
else:
    END = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    START = END - timedelta(days=90)

if not UNIVERSE:
    UNIVERSE = sorted(col.distinct("symbol"))
if SYMBOL_ALLOWLIST:
    UNIVERSE = [s for s in UNIVERSE if s.upper() in SYMBOL_ALLOWLIST]
if SYMBOL_BLACKLIST:
    UNIVERSE = [s for s in UNIVERSE if s.upper() not in SYMBOL_BLACKLIST]

print(f"[BT] Window: {START.isoformat()} -> {END.isoformat()}")
print(f"[BT] Symbols: {len(UNIVERSE)} -> {UNIVERSE[:10]}{'...' if len(UNIVERSE)>10 else ''}")

model = joblib.load(GLOBAL_MODEL_PATH)
feature_order = _load_feature_order()
n_expected = getattr(model, "n_features_in_", len(feature_order))
if n_expected != len(feature_order):
    print(f"[WARN] Model expects {n_expected} features; feature list has {len(feature_order)}.")

PER_THR = _load_per_symbol_thresholds(PER_SYMBOL_THRESH_PATH)

def load_symbol_df(sym: str) -> pd.DataFrame:
    cur = col.find(
        {"symbol": sym, "timestamp": {"$gte": START, "$lte": END}},
        {"_id":0,"timestamp":1,"open":1,"high":1,"low":1,"close":1,"volume":1,"indicators":1}
    ).sort("timestamp",1)
    df = pd.DataFrame(list(cur))
    if df.empty: return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ind = pd.json_normalize(df["indicators"])
    df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)
    num_cols = ["open","high","low","close","volume","sma_10","sma_50","ema_10","rsi_14",
                "macd","macd_signal","atr_14","bb_upper","bb_lower","bb_middle",
                "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10"]
    for c in num_cols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _attach_derived_features(df)
    return df

data: Dict[str, pd.DataFrame] = {s: load_symbol_df(s) for s in UNIVERSE}
for s in list(data.keys()):
    if data[s].empty: del data[s]
UNIVERSE = [s for s in UNIVERSE if s in data]

all_ts = sorted({ts for df in data.values() for ts in df["timestamp"]})
index = pd.DatetimeIndex(all_ts, tz="UTC").unique().sort_values()

# ---------- Backtest state ----------
equity = START_EQUITY
fees_rt = _fees_rt()
open_pos: Dict[str, List[dict]] = {}  # symbol -> list of positions
trades: List[dict] = []
equity_curve = []

def _thr_for(sym: str):
    return PER_THR.get(sym.upper(), (THRESH_LONG, THRESH_SHORT))

def _gross_exposure_frac():
    if equity <= 0: return 0.0
    notional = 0.0
    for lst in open_pos.values():
        for p in lst:
            notional += p["qty"] * p["entry_px"]
    return notional / equity

def _open_positions_count():
    return sum(len(lst) for lst in open_pos.values())

def _signal_ok(sym: str, row) -> Tuple[bool, str, float, float]:
    vec = _build_row_vec(row, feature_order)
    if vec is None or vec.shape[0] != n_expected:
        return (False, "", float("nan"), float("nan"))
    proba_up = float(model.predict_proba(vec.reshape(1,-1))[0,1])
    thr_long, thr_short = _thr_for(sym)
    direction = "NEUTRAL"
    if proba_up >= thr_long: direction = "UP"
    elif proba_up <= thr_short: direction = "DOWN"
    if TREND_FILTER:
        sma = row.get(f"sma_{TREND_LEN}")
        price = float(row["close"])
        if not (isinstance(sma, (float, int)) and np.isfinite(sma)):
            return (False, direction, proba_up, float("nan"))
        if direction == "UP" and not (price > sma): direction = "NEUTRAL"
        if direction == "DOWN" and not (price < sma): direction = "NEUTRAL"
    return (direction != "NEUTRAL", direction, proba_up, row.get(f"sma_{TREND_LEN}", float("nan")))

def _update_trailing(pos: dict, cur_row):
    """Update trailing (breakeven and ATR trail) for a single position using current bar."""
    side = pos["side"]
    entry_px = pos["entry_px"]
    stop_px = pos["stop_px"]
    take_px = pos["take_px"]

    atr = cur_row.get("atr_14")
    if (atr is None) or (not np.isfinite(atr)) or (atr <= 0):
        return

    hi = float(cur_row.get("high", cur_row.get("close", entry_px)))
    lo = float(cur_row.get("low",  cur_row.get("close", entry_px)))
    close = float(cur_row.get("close", entry_px))

    if side == "LONG":
        pos["high_water"] = max(pos.get("high_water", entry_px), hi)
        if BREAKEVEN_TRAIL and not pos.get("be_done", False):
            if close >= entry_px + BE_ATR_TRIGGER * atr:
                stop_px = max(stop_px, entry_px)
                pos["be_done"] = True
        if pos.get("be_done", False) and ATR_TRAIL_MULT > 0:
            trail_px = pos["high_water"] - ATR_TRAIL_MULT * atr
            stop_px = max(stop_px, trail_px)
        stop_px = min(stop_px, take_px)
        pos["stop_px"] = stop_px
    else:
        pos["low_water"] = min(pos.get("low_water", entry_px), lo)
        if BREAKEVEN_TRAIL and not pos.get("be_done", False):
            if close <= entry_px - BE_ATR_TRIGGER * atr:
                stop_px = min(stop_px, entry_px)
                pos["be_done"] = True
        if pos.get("be_done", False) and ATR_TRAIL_MULT > 0:
            trail_px = pos["low_water"] + ATR_TRAIL_MULT * atr
            stop_px = min(stop_px, trail_px)
        stop_px = max(stop_px, take_px)
        pos["stop_px"] = stop_px

def maybe_exit_for_symbol(sym: str, t_idx: int):
    """Process exits (and optional decay exit) for all positions in a symbol at bar index t_idx."""
    global equity
    if sym not in open_pos or not open_pos[sym]: return
    if t_idx+1 >= len(index): return
    ts = index[t_idx]
    t_next = index[t_idx+1]
    df = data[sym]

    # Re-check signal and update trailing on current bar
    if RECHECK_SIGNAL:
        cur = df[df["timestamp"] == ts]
        if not cur.empty:
            row_cur = cur.iloc[0]
            ok, direction, _, _ = _signal_ok(sym, row_cur)
            # Extend expiry or decay exit for EACH position
            new_list = []
            for pos in open_pos[sym]:
                if ok and ((pos["side"]=="LONG" and direction=="UP") or (pos["side"]=="SHORT" and direction=="DOWN")):
                    pos["expiry_ts"] = ts + timedelta(hours=HOLD_HOURS)
                elif EXIT_ON_DECAY:
                    # Close at current close
                    exit_price = float(row_cur["close"])
                    side = pos["side"]; entry_px = pos["entry_px"]; qty = pos["qty"]
                    gross = (exit_price - entry_px)/entry_px if side=="LONG" else (entry_px - exit_price)/entry_px
                    net = gross - fees_rt
                    pnl = qty * entry_px * net
                    equity += pnl
                    trades.append({
                        "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": ts,
                        "side": side, "px_in": entry_px, "px_out": exit_price, "qty": qty,
                        "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": "DECAY_EXIT"
                    })
                    continue  # don't keep this pos
                # Update trailing
                _update_trailing(pos, row_cur)
                new_list.append(pos)
            open_pos[sym] = new_list

    # Normal exit & partial TP on next bar's ranges
    row_next = df[df["timestamp"] == t_next]
    if row_next.empty or sym not in open_pos or not open_pos[sym]:
        return
    r = row_next.iloc[0]
    hi, lo, close = float(r["high"]), float(r["low"]), float(r["close"])

    new_list = []
    for pos in open_pos[sym]:
        side = pos["side"]
        entry_px = pos["entry_px"]
        qty = pos["qty"]
        atr_entry = pos["atr_entry"]  # ATR used at entry
        exit_reason, exit_px = None, None
        realized_partial = False

        # Check stop vs take first (conservative ordering: STOP -> TAKE -> PARTIAL)
        if side == "LONG":
            if lo <= pos["stop_px"]:
                exit_reason, exit_px = "STOP", pos["stop_px"]
            elif hi >= pos["take_px"]:
                exit_reason, exit_px = "TAKE", pos["take_px"]
            elif PARTIAL_TP_ENABLE and (not pos.get("partial_done", False)):
                ppx = pos.get("partial_px", None)
                if ppx is not None and hi >= ppx:
                    # Close PARTIAL_TP_FRAC of position
                    close_qty = qty * PARTIAL_TP_FRAC
                    gross = (ppx - entry_px)/entry_px
                    net = gross - fees_rt
                    pnl = close_qty * entry_px * net
                    equity += pnl
                    trades.append({
                        "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": t_next,
                        "side": side, "px_in": entry_px, "px_out": ppx, "qty": close_qty,
                        "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": "PARTIAL_TP"
                    })
                    # Keep remainder open
                    pos["qty"] = qty - close_qty
                    pos["partial_done"] = True
                    realized_partial = True
        else:  # SHORT
            if hi >= pos["stop_px"]:
                exit_reason, exit_px = "STOP", pos["stop_px"]
            elif lo <= pos["take_px"]:
                exit_reason, exit_px = "TAKE", pos["take_px"]
            elif PARTIAL_TP_ENABLE and (not pos.get("partial_done", False)):
                ppx = pos.get("partial_px", None)
                if ppx is not None and lo <= ppx:
                    close_qty = qty * PARTIAL_TP_FRAC
                    gross = (entry_px - ppx)/entry_px
                    net = gross - fees_rt
                    pnl = close_qty * entry_px * net
                    equity += pnl
                    trades.append({
                        "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": t_next,
                        "side": side, "px_in": entry_px, "px_out": ppx, "qty": close_qty,
                        "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": "PARTIAL_TP"
                    })
                    pos["qty"] = qty - close_qty
                    pos["partial_done"] = True
                    realized_partial = True

        # Expiry check if still open and not partially handled this bar
        if exit_reason is None and not realized_partial and t_next >= pos["expiry_ts"]:
            exit_reason, exit_px = "EXPIRY", close

        if exit_reason is not None:
            gross = (exit_px - entry_px)/entry_px if side=="LONG" else (entry_px - exit_px)/entry_px
            net = gross - fees_rt
            pnl = qty * entry_px * net
            equity += pnl
            trades.append({
                "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": t_next,
                "side": side, "px_in": entry_px, "px_out": exit_px, "qty": qty,
                "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": exit_reason
            })
        else:
            new_list.append(pos)

    open_pos[sym] = new_list

def maybe_open_for_symbol(sym: str, t_idx: int):
    """Entry decision on bar t, fill at t+1 close; allows pyramiding."""
    global equity
    if t_idx+1 >= len(index): return
    ts = index[t_idx]; t_next = index[t_idx+1]
    df = data[sym]
    cur = df[df["timestamp"] == ts]
    nxt = df[df["timestamp"] == t_next]
    if cur.empty or nxt.empty: return
    row = cur.iloc[0]; nxt_row = nxt.iloc[0]

    ok, direction, proba_up, _ = _signal_ok(sym, row)
    if not ok: return

    # Pyramiding limits
    sym_count = len(open_pos.get(sym, []))
    if sym_count >= MAX_PYRAMIDS_PER_SYMBOL: return
    if _open_positions_count() >= MAX_CONCURRENT: return

    # Exposure cap
    if _gross_exposure_frac() + RISK_PCT > MAX_GROSS_EXPOSURE: return

    # Require favorable price improvement before adding more (optional)
    if PYR_REQUIRE_IMPROVE and sym_count > 0:
        atr = nxt_row.get("atr_14")
        if atr is None or not np.isfinite(atr) or atr <= 0: return
        last_entry_px = open_pos[sym][-1]["entry_px"]
        price_now = float(row["close"])
        if direction == "UP":
            if price_now < last_entry_px + PYR_ADD_ATR_STEP * atr: return
        else:
            if price_now > last_entry_px - PYR_ADD_ATR_STEP * atr: return

    atr = nxt_row.get("atr_14")
    if (atr is None) or (not np.isfinite(atr)) or (atr <= 0): return

    entry_px = float(nxt_row["close"])
    usd_risk = equity * RISK_PCT
    if usd_risk <= 0: return
    qty = usd_risk / entry_px

    if direction == "UP":
        side = "LONG"
        stop_px = entry_px - SL_ATR_MULT*atr
        take_px = entry_px + TP_ATR_MULT*atr
        partial_px = entry_px + PARTIAL_TP_ATR_MULT*atr if PARTIAL_TP_ENABLE else None
        init_high, init_low = entry_px, None
    else:
        side = "SHORT"
        stop_px = entry_px + SL_ATR_MULT*atr
        take_px = entry_px - TP_ATR_MULT*atr
        partial_px = entry_px - PARTIAL_TP_ATR_MULT*atr if PARTIAL_TP_ENABLE else None
        init_high, init_low = None, entry_px

    pos = {
        "side": side, "entry_ts": ts, "entry_px": entry_px, "qty": qty,
        "stop_px": stop_px, "take_px": take_px,
        "partial_px": partial_px, "partial_done": False,
        "expiry_ts": ts + timedelta(hours=HOLD_HOURS),
        "be_done": False,
        "high_water": init_high, "low_water": init_low,
        "atr_entry": atr
    }
    open_pos.setdefault(sym, []).append(pos)

# ---- Run loop ----
trades_path = Path("backtest_trades.csv")
equity_path = Path("backtest_equity.csv")

for i, ts in enumerate(index[:-1]):
    # Exits first for all symbols
    for s in list(open_pos.keys()):
        maybe_exit_for_symbol(s, i)
    # Entries for all symbols
    for s in UNIVERSE:
        maybe_open_for_symbol(s, i)
    # Record equity
    equity_curve.append({
        "timestamp": ts,
        "equity": equity,
        "open_positions": _open_positions_count(),
        "exposure_frac": _gross_exposure_frac()
    })

# Close remaining positions at final bar
last_ts = index[-1]
for sym, lst in list(open_pos.items()):
    df = data[sym]
    last_row = df[df["timestamp"] == last_ts]
    if last_row.empty: continue
    close_px = float(last_row.iloc[0]["close"])
    new_list = []
    for pos in lst:
        side = pos["side"]; entry_px = pos["entry_px"]; qty = pos["qty"]
        gross = (close_px - entry_px)/entry_px if side=="LONG" else (entry_px - close_px)/entry_px
        net = gross - _fees_rt()
        pnl = qty * entry_px * net
        equity += pnl
        trades.append({
            "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": last_ts,
            "side": side, "px_in": entry_px, "px_out": close_px, "qty": qty,
            "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": "END_CLOSED"
        })

# ---- Save ----
trades_df = pd.DataFrame(trades)
eq_df = pd.DataFrame(equity_curve)
trades_df.to_csv(trades_path, index=False)
eq_df.to_csv(equity_path, index=False)

total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
wins = (trades_df["pnl"] > 0).sum() if not trades_df.empty else 0
losses = (trades_df["pnl"] <= 0).sum() if not trades_df.empty else 0
win_rate = (wins / max(1, wins+losses)) * 100.0

print(f"[BT] Trades: {len(trades_df)} | Win%: {win_rate:.1f}% | PnL: ${total_pnl:,.2f}")
print(f"[BT] Final equity: ${equity:,.2f}  (start ${START_EQUITY:,.2f})")
print(f"[BT] Files written: {trades_path} , {equity_path}")
