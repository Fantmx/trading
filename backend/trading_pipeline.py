import os
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import subprocess
import sys

# ---------- Config ----------
TZ = pytz.timezone("America/Chicago")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

# If you'd rather import and call Python functions directly, flip this to True
USE_SUBPROCESS = True

BASE = Path(__file__).parent

def _resolve_script(name: str) -> str:
    # Try common locations in order
    candidates = [
        BASE / name,
        BASE / "scripts" / name,
        BASE / "app" / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # last resort: return as-is (subprocess will error with a clear path)
    return str(BASE / name)

SCRIPTS = {
    "ingest": _resolve_script("ingest_latest_binance.py"),
    "indicators": _resolve_script("indicators_incremental.py"),
    "inference": _resolve_script("inference_per_symbol.py"),
    "paper": _resolve_script("paper_executor.py"),
}
# Stagger minutes within the hour to ensure sequential steps
# :01 ingest, :02 indicators, :03 inference, :04 paper exec
CRON_MINUTES = {
    "ingest": 1,
    "indicators": 6,
    "inference": 8,
    "paper": 10,
}

# APScheduler job options
MISFIRE_GRACE = 900       # seconds a job can be "late" and still run
MAX_INSTANCES = 1         # don't let jobs overlap
COALESCE = True           # if multiple runs are due, merge into one

# ---------- Logging ----------
logger = logging.getLogger("trading_pipeline")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# Also echo to console
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(console)

# ---------- Env ----------
def _find_env_path():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        p = os.path.join(cur, "configs", ".env")
        if os.path.isfile(p):
            return p
        cur = os.path.dirname(cur)
    return None

envp = _find_env_path()
if envp:
    load_dotenv(dotenv_path=envp)
else:
    logger.warning("Could not find configs/.env")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.warning("MONGO_URI not found in environment. Make sure your configs/.env is set.")


if not MONGO_URI:
    logger.warning("MONGO_URI not found in environment. Make sure your .env is set.")

# ---------- Helpers ----------
def run_script(path: str, name: str):
    """Run a script via subprocess so we don't fight imports/paths."""
    try:
        logger.info(f"[{name}] starting…")
        result = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=str(Path(__file__).parent)  # run from this folder
        )
        if result.returncode == 0:
            logger.info(f"[{name}] done:\n{result.stdout.strip()}")
        else:
            logger.error(f"[{name}] exited with code {result.returncode}:\n{result.stdout.strip()}")
    except Exception as e:
        logger.exception(f"[{name}] failed: {e}")

def task_ingest():
    if USE_SUBPROCESS:
        run_script(SCRIPTS["ingest"], "INGEST")
    else:
        # from ingest_latest_binance import main as ingest_main
        # ingest_main()
        pass

def task_indicators():
    if USE_SUBPROCESS:
        run_script(SCRIPTS["indicators"], "INDICATORS")
    else:
        # from indicators_incremental import main as indicators_main
        # indicators_main()
        pass

def task_inference():
    if USE_SUBPROCESS:
        run_script(SCRIPTS["inference"], "INFERENCE")
    else:
        # from inference_per_symbol import main as inference_main
        # inference_main()
        pass

def task_paper():
    if USE_SUBPROCESS:
        run_script(SCRIPTS["paper"], "PAPER")
    else:
        # from paper_executor import main as paper_main
        # paper_main()
        pass

def add_cron_job(sched, func, minute: int, name: str):
    trigger = CronTrigger(minute=minute, timezone=TZ)
    sched.add_job(
        func,
        trigger=trigger,
        name=name,
        misfire_grace_time=MISFIRE_GRACE,
        max_instances=MAX_INSTANCES,
        coalesce=COALESCE,
        replace_existing=True,
    )
    logger.info(f"Scheduled {name} at minute :{minute:02d} (America/Chicago)")

# ---------- Main ----------
def main():
    logger.info("Starting trading pipeline scheduler…")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working dir: {Path.cwd()}")
    logger.info(f"Timezone: {TZ}")

    sched = BlockingScheduler(timezone=TZ)

    add_cron_job(sched, task_ingest,     CRON_MINUTES["ingest"],     "INGEST")
    add_cron_job(sched, task_indicators, CRON_MINUTES["indicators"], "INDICATORS")
    add_cron_job(sched, task_inference,  CRON_MINUTES["inference"],  "INFERENCE")
    add_cron_job(sched, task_paper,      CRON_MINUTES["paper"],      "PAPER")

    # Optional: kick off a first run immediately
    if USE_SUBPROCESS:
        run_script(SCRIPTS["ingest"], "INGEST (startup)")
        run_script(SCRIPTS["indicators"], "INDICATORS (startup)")
        run_script(SCRIPTS["inference"], "INFERENCE (startup)")
        run_script(SCRIPTS["paper"], "PAPER (startup)")


    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler.")
    except Exception as e:
        logger.exception(f"Scheduler crashed: {e}")
        time.sleep(2)

if __name__ == "__main__":
    main()
