# backend/utils/binance_us_symbols.py
from __future__ import annotations
import time
import logging
import requests
from typing import List, Dict, NamedTuple

BINANCE_US_REST = "https://api.binance.us"

class ResolvedSymbols(NamedTuple):
    requested: List[str]
    resolved: List[str]
    alias_map: Dict[str, str]  # e.g. {"MATICUSDT": "POLUSDT"}
    missing: List[str]

def _get_exchange_info(retries: int = 5, backoff: float = 0.8) -> dict:
    url = f"{BINANCE_US_REST}/api/v3/exchangeInfo"
    last_exc = None
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            sleep = backoff * (2 ** i)
            time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch exchangeInfo from Binance.US: {last_exc}")

def resolve_us_symbols(desired_symbols: List[str]) -> ResolvedSymbols:
    """
    Given a list like ["BTCUSDT","ETHUSDT","MATICUSDT",...], produce a Binance.US-safe list.
    - If a symbol is missing but a known alias exists (e.g., POLUSDT for MATICUSDT), use the alias.
    - Otherwise drop the symbol and report it in `missing`.
    """
    info = _get_exchange_info()
    available = {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}

    # Known alias rules (kept small & explicit). Checked only if the original symbol is absent.
    # We prefer to derive aliases by existence rather than assuming rebrands.
    alias_rules: Dict[str, List[str]] = {
        "MATICUSDT": ["POLUSDT"],  # Polygon rebrand on Binance.US
        # Optional future-proofing: if ever needed, add more here, e.g. "FETUSDT": ["ASIUSDT"]
    }

    resolved: List[str] = []
    missing: List[str] = []
    alias_map: Dict[str, str] = {}

    for sym in desired_symbols:
        if sym in available:
            resolved.append(sym)
            continue

        # try aliases in order
        aliased = False
        for candidate in alias_rules.get(sym, []):
            if candidate in available:
                resolved.append(candidate)
                alias_map[sym] = candidate
                aliased = True
                break

        if not aliased:
            missing.append(sym)

    # Log a concise summary (you can swap to your logger)
    logging.info("Binance.US symbol resolution: requested=%s resolved=%s alias_map=%s missing=%s",
                 desired_symbols, resolved, alias_map, missing)

    return ResolvedSymbols(
        requested=desired_symbols,
        resolved=resolved,
        alias_map=alias_map,
        missing=missing,
    )
