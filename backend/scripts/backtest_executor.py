# backtest_executor.py — risk-based multi-hour backtest using your v3 model + Mongo data
# Outputs: backtest_trades.csv and backtest_equity.csv (in working dir)

import os, sys, json, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------- .env ----------
def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p): return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError("configs/.env not found")
load_dotenv(dotenv_path=_find_env_path())

# ---------- Config (env-overridable) ----------
DB_NAME            = os.getenv("MONGO_DB", "CoinCluster")
COLL               = os.getenv("MONGO_COLLECTION", "binance_price_data")
MONGO_URI          = os.getenv("MONGO_URI")

# Backtest window (UTC). Example: BACKTEST_START=2024-06-01 BACKTEST_END=2024-09-01
BACKTEST_START_S   = os.getenv("BACKTEST_START")  # "YYYY-MM-DD"
BACKTEST_END_S     = os.getenv("BACKTEST_END")

# Universe: if empty -> use all symbols found in collection
UNIVERSE           = [s.strip() for s in (os.getenv("UNIVERSE","").split(",")) if s.strip()]

# Model + features (same as inference_per_symbol.py)
GLOBAL_MODEL_PATH  = Path("model_price_direction_xgb_v3.pkl")
FEATURE_ORDER_PATH = Path("model_features_v3.json")

FEATURES_27 = [
    "close","volume",
    "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
    "atr_14","atr_norm","bb_upper","bb_lower","bb_middle","bb_width","bb_pctb",
    "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10",
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    "ret_vol_6h","ret_vol_12h","ret_vol_24h"
]

# Trading rules (match paper_executor_v2)
START_EQUITY       = float(os.getenv("START_EQUITY", "1000"))
RISK_PCT           = float(os.getenv("RISK_PCT", "0.015"))
HOLD_HOURS         = int(os.getenv("HOLD_HOURS", "6"))
SL_ATR_MULT        = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT        = float(os.getenv("TP_ATR_MULT", "2.0"))
FEE_BP             = float(os.getenv("FEE_BP", "5"))     # per side
SLIPPAGE_BP        = float(os.getenv("SLIPPAGE_BP", "5"))
MAX_CONCURRENT     = int(os.getenv("MAX_CONCURRENT", "5"))

# ---------- Helpers ----------
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
    # Derived features (match inference_per_symbol)
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
    return df

def _build_row_vec(row, feature_order):
    vec = []
    for f in feature_order:
        val = row.get(f, np.nan)
        if isinstance(val, (float, int)) and (np.isnan(val) or np.isinf(val)):
            return None
        vec.append(float(val))
    return np.array(vec, dtype=float)

# ---------- Load data ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLL]

if BACKTEST_START_S and BACKTEST_END_S:
    START = _utc(datetime.fromisoformat(BACKTEST_START_S))
    END   = _utc(datetime.fromisoformat(BACKTEST_END_S))
else:
    # default to last 90 days
    END = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    START = END - timedelta(days=90)

if not UNIVERSE:
    UNIVERSE = sorted(col.distinct("symbol"))

print(f"[BT] Window: {START.isoformat()} -> {END.isoformat()}")
print(f"[BT] Symbols: {len(UNIVERSE)} -> {UNIVERSE[:10]}{'...' if len(UNIVERSE)>10 else ''}")

# Load model
model = joblib.load(GLOBAL_MODEL_PATH)
feature_order = _load_feature_order()
n_expected = getattr(model, "n_features_in_", len(feature_order))
if n_expected != len(feature_order):
    print(f"[WARN] Model expects {n_expected} features; feature list has {len(feature_order)}.")

# ---------- Build per-symbol DataFrames ----------
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
    # enforce numeric
    for c in ["open","high","low","close","volume","sma_10","sma_50","ema_10","rsi_14",
              "macd","macd_signal","atr_14","bb_upper","bb_lower","bb_middle",
              "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _attach_derived_features(df)
    return df

data: Dict[str, pd.DataFrame] = {s: load_symbol_df(s) for s in UNIVERSE}
# Build the master hourly index over window, then align fills to that
all_ts = sorted({ts for df in data.values() if not df.empty for ts in df["timestamp"]})
if not all_ts:
    raise SystemExit("[BT] No data in the window; expand BACKTEST_START/BACKTEST_END.")
index = pd.DatetimeIndex(all_ts, tz="UTC").unique().sort_values()

# ---------- Backtest loop ----------
equity = START_EQUITY
fees_rt = _fees_rt()
open_pos: Dict[str, dict] = {}  # symbol -> position
trades: List[dict] = []
equity_curve = []

def maybe_exit(sym: str, t_idx: int):
    """Check exits on 'next bar' of timestamp index[t_idx] using high/low/close of that bar."""
    global equity
    if sym not in open_pos: return
    pos = open_pos[sym]
    # Next bar timestamp for fills
    if t_idx+1 >= len(index): return
    t_next = index[t_idx+1]
    df = data[sym]
    row = df[df["timestamp"] == t_next]
    if row.empty: return
    r = row.iloc[0]
    hi, lo, close = float(r["high"]), float(r["low"]), float(r["close"])
    # Exit logic
    exit_reason, exit_px = None, None
    if pos["side"] == "LONG":
        if lo <= pos["stop_px"]:
            exit_reason, exit_px = "STOP", pos["stop_px"]
        elif hi >= pos["take_px"]:
            exit_reason, exit_px = "TAKE", pos["take_px"]
    else:
        if hi >= pos["stop_px"]:
            exit_reason, exit_px = "STOP", pos["stop_px"]
        elif lo <= pos["take_px"]:
            exit_reason, exit_px = "TAKE", pos["take_px"]

    # Expiry check
    if exit_reason is None and t_next >= pos["expiry_ts"]:
        exit_reason, exit_px = "EXPIRY", close

    if exit_reason:
        entry_px = pos["entry_px"]
        qty = pos["qty"]
        side = pos["side"]
        gross = (exit_px - entry_px)/entry_px if side=="LONG" else (entry_px - exit_px)/entry_px
        net = gross - fees_rt
        pnl = qty * entry_px * net
        equity += pnl
        trades.append({
            "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": t_next,
            "side": side, "px_in": entry_px, "px_out": exit_px, "qty": qty,
            "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": exit_reason
        })
        del open_pos[sym]

def maybe_open(sym: str, t_idx: int):
    """Decide entry at index[t_idx] (decision at t, fill at t+1 close)."""
    global equity
    if sym in open_pos: return
    if len(open_pos) >= MAX_CONCURRENT: return
    # Need t and t+1 rows
    if t_idx+1 >= len(index): return
    ts = index[t_idx]
    t_next = index[t_idx+1]

    df = data[sym]
    cur = df[df["timestamp"] == ts]
    nxt = df[df["timestamp"] == t_next]
    if cur.empty or nxt.empty: return
    row = cur.iloc[0]
    nxt_row = nxt.iloc[0]

    # Build feature vector from 'cur' bar (decision at close of t)
    vec = _build_row_vec(row, feature_order)
    if vec is None or vec.shape[0] != n_expected: return
    proba_up = float(model.predict_proba(vec.reshape(1,-1))[0,1])
    decision = "UP" if proba_up >= 0.5 else "DOWN"

    # Require ATR on next bar for SL/TP sizing
    atr = nxt_row.get("atr_14")
    if (atr is None) or (not np.isfinite(atr)) or (atr <= 0): return

    # Entry at next close
    entry_px = float(nxt_row["close"])
    usd_risk = equity * RISK_PCT
    if usd_risk <= 0: return
    qty = usd_risk / entry_px

    if decision == "UP":
        side = "LONG"
        stop_px = entry_px - SL_ATR_MULT*atr
        take_px = entry_px + TP_ATR_MULT*atr
    else:
        side = "SHORT"
        stop_px = entry_px + SL_ATR_MULT*atr
        take_px = entry_px - TP_ATR_MULT*atr

    open_pos[sym] = {
        "side": side, "entry_ts": ts, "entry_px": entry_px, "qty": qty,
        "stop_px": stop_px, "take_px": take_px,
        "expiry_ts": ts + timedelta(hours=HOLD_HOURS)
    }

# Iterate over every hour in the window
for i, ts in enumerate(index[:-1]):  # we look at next bar, so stop at len-1
    # 1) process exits first (fair across symbols)
    for s in UNIVERSE:
        if not data[s].empty and s in open_pos:
            maybe_exit(s, i)

    # 2) process entries
    for s in UNIVERSE:
        if data[s].empty: continue
        maybe_open(s, i)

    # 3) record equity
    equity_curve.append({"timestamp": ts, "equity": equity, "open_positions": len(open_pos)})

# If anything left open at the very end, close at last available close
last_ts = index[-1]
for s, pos in list(open_pos.items()):
    df = data[s]
    last_row = df[df["timestamp"] == last_ts]
    if last_row.empty: continue
    exit_px = float(last_row.iloc[0]["close"])
    side = pos["side"]; entry_px = pos["entry_px"]; qty = pos["qty"]
    gross = (exit_px - entry_px)/entry_px if side=="LONG" else (entry_px - exit_px)/entry_px
    net = gross - fees_rt
    pnl = qty * entry_px * net
    equity += pnl
    trades.append({
        "symbol": s, "entry_ts": pos["entry_ts"], "exit_ts": last_ts,
        "side": side, "px_in": entry_px, "px_out": exit_px, "qty": qty,
        "gross_ret": gross, "net_ret": net, "pnl": pnl, "reason": "END_CLOSED"
    })
    del open_pos[s]

# ---------- Save outputs ----------
trades_df = pd.DataFrame(trades)
eq_df = pd.DataFrame(equity_curve)
trades_path = Path("backtest_trades.csv")
equity_path = Path("backtest_equity.csv")
trades_df.to_csv(trades_path, index=False)
eq_df.to_csv(equity_path, index=False)

# Basic summary
total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
wins = (trades_df["pnl"] > 0).sum() if not trades_df.empty else 0
losses = (trades_df["pnl"] <= 0).sum() if not trades_df.empty else 0
win_rate = (wins / max(1, wins+losses)) * 100.0

print(f"[BT] Trades: {len(trades_df)} | Win%: {win_rate:.1f}% | PnL: ${total_pnl:,.2f}")
print(f"[BT] Final equity: ${equity:,.2f}  (start ${START_EQUITY:,.2f})")
print(f"[BT] Files written: {trades_path} , {equity_path}")
