import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
import joblib

# ------------ Config ------------
DB_NAME = "CoinCluster"
PRICE_COLL = "binance_price_data"
OUT_DIR = Path("backtest_out")
OUT_DIR.mkdir(exist_ok=True)

DAYS = 90
LOOKBACK_MIN = 36  # need enough to compute 24h features

COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
    "XRPUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","SHIBUSDT","PEPEUSDT","APTUSDT","INJUSDT","FETUSDT"
]

# Features must match v3 training
# REPLACE these lines in your backtest:
GLOBAL_MODEL_PATH     = Path("model_price_direction_xgb_v3.pkl")
GLOBAL_THRESHOLD_PATH = Path("model_threshold_v3.txt")
PER_SYM_THRESH_PATH   = Path("model_thresholds_per_symbol_v3.json")

# And instead of hardcoding FEATURES, load them:
FEATURES_PATH = Path("model_features_v3.txt")
FEATURES = [f.strip() for f in FEATURES_PATH.read_text().splitlines() if f.strip()]
print(f"Using {len(FEATURES)} features from {FEATURES_PATH}")


# ------------ Setup ------------
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[DB_NAME]
prices = db[PRICE_COLL]

model = joblib.load(GLOBAL_MODEL_PATH)
global_thr = 0.5
if GLOBAL_THRESHOLD_PATH.exists():
    try:
        global_thr = float(GLOBAL_THRESHOLD_PATH.read_text().strip())
    except:
        pass

per_sym_thr = {}
if PER_SYM_THRESH_PATH.exists():
    try:
        per_sym_thr = json.loads(PER_SYM_THRESH_PATH.read_text())
    except:
        per_sym_thr = {}

# ------------ Helpers ------------
def fetch_last_days(symbol: str, days: int) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    cur = prices.find(
        {"symbol": symbol, "timestamp": {"$gte": since}},
        {"_id":0,"symbol":1,"timestamp":1,"open":1,"high":1,"low":1,"close":1,"volume":1,"indicators":1}
    ).sort("timestamp", 1)
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Flatten indicators
    if "indicators" in df.columns:
        ind = pd.json_normalize(df["indicators"])
        df = pd.concat([df.drop(columns=["indicators"]), ind], axis=1)
    # Cast numerics
    for c in ["open","high","low","close","volume",
              "sma_10","sma_50","ema_10","rsi_14","macd","macd_signal",
              "atr_14","bb_upper","bb_lower","bb_middle",
              "stoch_rsi_k","stoch_rsi_d","volume_sma_20","volume_roc_10"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived v3 features in place and return cleaned DF."""
    if df.empty or len(df) < LOOKBACK_MIN:
        return pd.DataFrame()

    # Derived: ATR normalized & Bollinger
    df["atr_norm"] = df["atr_14"] / df["close"]
    bw = (df.get("bb_upper") - df.get("bb_lower"))
    bw_safe = bw.replace(0, np.nan)
    df["bb_width"] = bw_safe / df["close"]
    df["bb_pctb"]  = (df["close"] - df.get("bb_lower")) / bw_safe

    # Momentum returns
    df["ret_1h"]  = df["close"].pct_change(1)
    df["ret_3h"]  = df["close"].pct_change(3)
    df["ret_6h"]  = df["close"].pct_change(6)
    df["ret_12h"] = df["close"].pct_change(12)
    df["ret_24h"] = df["close"].pct_change(24)

    # Rolling volatility of returns
    r = df["close"].pct_change()
    df["ret_vol_6h"]  = r.rolling(6).std()
    df["ret_vol_12h"] = r.rolling(12).std()
    df["ret_vol_24h"] = r.rolling(24).std()

    # Actual label for evaluation
    df["actual_up"] = (df["close"].shift(-1) > df["close"]).astype("Int8")

    # Clean/sanitize
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def valid_mask(df: pd.DataFrame):
    """Rows that have all FEATURES AND a known next-close (actual_up not NA)."""
    need = FEATURES + ["actual_up"]
    present = [c for c in need if c in df.columns]
    mask = df[present].notna().all(axis=1)
    # last row has no next_close label; mask removes it automatically via actual_up NaN
    return mask

def equity_curve(returns: np.ndarray, start_equity: float = 1.0) -> np.ndarray:
    # cumulative product of (1+ret), no leverage
    eq = np.cumprod(1.0 + returns)
    return start_equity * eq

# ------------ Backtest ------------
all_trades = []
summ_rows = []

for sym in COINS:
    df = fetch_last_days(sym, DAYS)
    if df.empty:
        summ_rows.append([sym, 0, 0, 0.0, 0.0, 0.0])
        continue

    df = build_features(df)
    if df.empty or not set(FEATURES).issubset(df.columns):
        summ_rows.append([sym, 0, 0, 0.0, 0.0, 0.0])
        continue

    m = valid_mask(df)
    df_use = df.loc[m].copy()
    if df_use.empty:
        summ_rows.append([sym, 0, 0, 0.0, 0.0, 0.0])
        continue

    X = df_use[FEATURES].values
    proba = model.predict_proba(X)[:, 1]
    thr = float(per_sym_thr.get(sym, global_thr))
    pred_up = (proba >= thr).astype(int)

    # Shift next close for PnL calc (trade from current close to next close)
    next_close = df_use["close"].shift(-1)
    close_now  = df_use["close"]
    # Align lengths: last row will be NaN next_close; drop it
    valid_trade = next_close.notna()
    close_now  = close_now[valid_trade]
    next_close = next_close[valid_trade]
    pred_up_tr = pred_up[valid_trade.values]
    actual_up  = df_use.loc[valid_trade, "actual_up"].astype(int).values
    ts_tr      = df_use.loc[valid_trade, "timestamp"].values

    # PnL per trade
    # long if pred_up=1: (next - now)/now ; short if pred_up=0: (now - next)/now
    long_ret  = (next_close.values - close_now.values) / close_now.values
    short_ret = (close_now.values - next_close.values) / close_now.values
    trade_ret = np.where(pred_up_tr == 1, long_ret, short_ret)

    # Metrics
    acc = (pred_up_tr == actual_up).mean() if len(actual_up) else 0.0
    win_rate = (trade_ret > 0).mean() if len(trade_ret) else 0.0
    pnl_pct = trade_ret.sum() * 100.0

    summ_rows.append([sym, int(len(df_use)), int(len(trade_ret)), float(acc), float(win_rate), float(pnl_pct)])

    # Save per-trade detail
    per_trades_df = pd.DataFrame({
        "symbol": sym,
        "timestamp": ts_tr,
        "prob_up": proba[valid_trade.values],
        "pred_up": pred_up_tr,
        "actual_up": actual_up,
        "close_now": close_now.values,
        "next_close": next_close.values,
        "trade_ret": trade_ret
    })
    per_trades_df.to_csv(OUT_DIR / f"{sym}_trades_90d.csv", index=False)

    # Plot equity curve
    eq = equity_curve(trade_ret, 1.0)
    plt.figure()
    plt.plot(eq)
    plt.title(f"{sym} Equity Curve (90d)")
    plt.xlabel("Trades (hourly)")
    plt.ylabel("Equity (start=1.0)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{sym}_equity_90d.png")
    plt.close()

# Summary CSV
summary = pd.DataFrame(summ_rows, columns=["symbol","rows_used","trades","accuracy","win_rate","pnl_percent"])
summary = summary.sort_values("accuracy", ascending=False)
summary.to_csv(OUT_DIR / "summary_90d.csv", index=False)

# Overall equity curve (equal-weight, by concatenating returns across coins in time order)
# Simple: average trade return across all coins at each timestamp where they exist.
# Build a panel keyed by timestamp: mean across symbols
panel = []
for sym in COINS:
    path = OUT_DIR / f"{sym}_trades_90d.csv"
    if path.exists():
        df_tr = pd.read_csv(path, parse_dates=["timestamp"])
        panel.append(df_tr[["timestamp","trade_ret"]].rename(columns={"trade_ret": sym}))
if panel:
    merged = panel[0]
    for p in panel[1:]:
        merged = pd.merge(merged, p, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    # mean across symbols per timestamp (ignore NaN)
    merged["ret_mean"] = merged.drop(columns=["timestamp"]).mean(axis=1, skipna=True)
    merged["equity"] = (1.0 + merged["ret_mean"].fillna(0.0)).cumprod()
    merged.to_csv(OUT_DIR / "overall_equity_series_90d.csv", index=False)

    plt.figure()
    plt.plot(merged["equity"].values)
    plt.title("Overall Equity Curve (Equal-weight, 90d)")
    plt.xlabel("Time steps (hourly)")
    plt.ylabel("Equity (start=1.0)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "overall_equity_90d.png")
    plt.close()

print("\n=== Backtest done ===")
print(f"Summary: {OUT_DIR/'summary_90d.csv'}")
print(f"Per-symbol trades: {OUT_DIR}/<SYMBOL>_trades_90d.csv")
print(f"Equity plots: {OUT_DIR}/<SYMBOL>_equity_90d.png and overall_equity_90d.png")
