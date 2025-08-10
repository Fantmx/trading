# utils/runtime_state.py
import json
from pathlib import Path
from typing import Any, Dict

STATE_FILE = Path("runtime_state.json")

def write_state(upserts: int, note: str = "", extra: Dict[str, Any] | None = None) -> None:
    data = {"upserts": int(upserts), "note": note}
    if extra:
        data.update(extra)
    STATE_FILE.write_text(json.dumps(data))

def read_state(default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return default or {}

def get_last_upserts() -> int:
    try:
        return int(read_state().get("upserts", -1))
    except Exception:
        return -1
