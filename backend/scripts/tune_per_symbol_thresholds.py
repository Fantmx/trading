#!/usr/bin/env python
"""
tune_per_symbol_thresholds.py

Grid-search per-symbol confidence thresholds for your v5 backtester.
Strategy:
- Tune each symbol in isolation (SYMBOL_ALLOWLIST=<that symbol>) over a grid of THRESH_LONG in [0.55 .. 0.62].
  THRESH_SHORT is set to 1 - THRESH_LONG.
- Score each run by total PnL (fallback to EV/trade if equal).
- Write per_symbol_thresholds.tuned.json with the best threshold per symbol.
- Then run a final multi-symbol backtest using the tuned file to show the aggregate result.

Usage (run this from your repo's scripts folder):
    python tune_per_symbol_thresholds.py --symbols "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOTUSDT" \
        --bt-path .\backtest_executor_v5.py --out thresholds.tuned.json --start 2025-05-01 --end 2025-08-01

Notes:
- Relies on your existing configs/.env for DB and defaults.
- Safe to CTRL+C; partial results remain in memory but final file won't be written.
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

def run_backtest(bt_path: str, extra_env: dict) -> tuple[float, float, float]:
    """
    Run the backtester and return (total_pnl, win_rate, ev_per_trade).
    Expects it to write backtest_trades.csv in cwd.
    """
    env = os.environ.copy()
    env.update(extra_env or {})
    # Clear path overrides that could conflict
    env.pop("PER_SYMBOL_THRESH_PATH", None)

    # Call the backtester
    cmd = [sys.executable, bt_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Optional: print summarized stdout for debugging
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    trades_path = Path("backtest_trades.csv")
    if not trades_path.exists():
        return (float("-inf"), 0.0, float("-inf"))
    df = pd.read_csv(trades_path)
    if df.empty:
        return (0.0, 0.0, 0.0)
    total_pnl = float(df["pnl"].sum())
    win_rate = float((df["pnl"] > 0).mean() * 100.0)
    ev = float(df["pnl"].mean())
    return (total_pnl, win_rate, ev)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols to tune (e.g., BTCUSDT,ETHUSDT,BNBUSDT,...)")
    ap.add_argument("--bt-path", default="backtest_executor_v5.py", help="Path to v5 backtester")
    ap.add_argument("--out", default="per_symbol_thresholds.tuned.json", help="Output JSON for tuned thresholds")
    ap.add_argument("--start", default=None, help="Override BACKTEST_START (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Override BACKTEST_END (YYYY-MM-DD)")
    ap.add_argument("--min", type=float, default=0.55, help="Grid min for THRESH_LONG")
    ap.add_argument("--max", type=float, default=0.62, help="Grid max for THRESH_LONG")
    ap.add_argument("--step", type=float, default=0.01, help="Grid step for THRESH_LONG")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"[TUNE] Symbols: {symbols}")
    grid = []
    t = args.min
    while t <= args.max + 1e-9:
        grid.append(round(t, 3))
        t += args.step
    print(f"[TUNE] Threshold grid: {grid} (short = 1 - long)")

    best = {}
    leaderboard = []

    for sym in symbols:
        print(f"\n[TUNE] >>> {sym} <<<")
        best_score = float("-inf")
        best_thr = None
        best_wr = 0.0
        best_ev = float("-inf")

        for thr in grid:
            extra_env = {
                "SYMBOL_ALLOWLIST": sym,
                "THRESH_LONG": str(thr),
                "THRESH_SHORT": str(round(1.0 - thr, 6)),
                "MAX_CONCURRENT": "1",     # isolate symbol
                "MAX_GROSS_EXPOSURE": "1", # no cap binding while single-symbol
            }
            if args.start:
                extra_env["BACKTEST_START"] = args.start
            if args.end:
                extra_env["BACKTEST_END"] = args.end

            pnl, wr, ev = run_backtest(args.bt_path, extra_env)
            print(f"[{sym}] thr={thr:.3f}  pnl={pnl:.2f}  wr={wr:.1f}%  ev={ev:.4f}")
            # Rank by PnL first, then EV
            score = (pnl, ev)
            if score > (best_score, best_ev):
                best_score = pnl
                best_ev = ev
                best_thr = thr
                best_wr = wr

        # Fallback if nothing ran
        if best_thr is None:
            best_thr = 0.56

        best[sym] = round(float(best_thr), 3)
        leaderboard.append((sym, best_thr, best_score, best_wr, best_ev))
        print(f"[TUNE] {sym} -> best thr {best_thr:.3f}  pnl={best_score:.2f}  wr={best_wr:.1f}%  ev={best_ev:.4f}")

    # Write tuned file
    out_path = Path(args.out)
    out_path.write_text(json.dumps(best, indent=2))
    print(f"\n[TUNE] Wrote tuned thresholds to: {out_path.resolve()}")

    # Final evaluation on full set with exposure cap restored
    print("\n[TUNE] Running final multi-symbol evaluation with tuned thresholds ...")
    extra_env = {
        "SYMBOL_ALLOWLIST": ",".join(symbols),
        "PER_SYMBOL_THRESH_PATH": str(out_path.name),
        # Restore defaults so your .env governs the rest
    }
    if args.start:
        extra_env["BACKTEST_START"] = args.start
    if args.end:
        extra_env["BACKTEST_END"] = args.end

    pnl, wr, ev = run_backtest(args.bt_path, extra_env)
    print(f"[TUNE][FINAL] pnl={pnl:.2f}  wr={wr:.1f}%  ev/trade={ev:.4f}")
    print("[TUNE] Done.")

if __name__ == "__main__":
    main()
