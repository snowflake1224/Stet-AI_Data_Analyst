#!/usr/bin/env python3
"""
KDD Cup 2026 — Data Agent v5
==============================

Local development (Gemini free):
  Copy .env.example → .env, fill in your Gemini API key, then:
  python main.py

Official evaluation (auto-injected by organizers):
  MODEL_API_URL, MODEL_API_KEY, MODEL_NAME are set automatically.
  Same code, no changes needed.
"""
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use system env vars directly


# ── Configuration from environment variables ────────────────────────────────
MODEL_API_URL = os.environ.get("MODEL_API_URL",  "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY",  "")
MODEL_NAME    = os.environ.get("MODEL_NAME",     "gemini-3.5-flash-lite")
MAX_WORKERS   = int(os.environ.get("MAX_WORKERS",              "2"))
TASK_TIMEOUT  = int(os.environ.get("TASK_TIMEOUT_SECONDS",  "900"))
INPUT_ROOT    = Path(os.environ.get("INPUT_ROOT",  "/input"))
OUTPUT_ROOT   = Path(os.environ.get("OUTPUT_ROOT", "/output"))
LOG_DIR       = Path(os.environ.get("LOG_DIR",     "/logs"))

# ── Logging setup ────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "runtime.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("agent")

# ── Set env vars so ModelClient picks them up ────────────────────────────────
os.environ["MODEL_API_URL"] = MODEL_API_URL
os.environ["MODEL_API_KEY"] = MODEL_API_KEY
os.environ["MODEL_NAME"]    = MODEL_NAME

from agent.agents.agent import DataAgent
from agent.agents.model import ModelClient
from agent.benchmark.dataset import load_tasks


def make_agent() -> DataAgent:
    """Create a fresh agent (one per task to avoid state leaks)."""
    client = ModelClient()
    return DataAgent(client=client, task_timeout=TASK_TIMEOUT)


def write_prediction(task_id: str, columns: list, rows: list) -> None:
    out_dir = OUTPUT_ROOT / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prediction.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([columns] + rows)


def write_empty_prediction(task_id: str) -> None:
    """Write an empty CSV so the evaluator doesn't crash on a missing file."""
    out_dir = OUTPUT_ROOT / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction.csv").write_text("", encoding="utf-8")


def run_one_task(task) -> dict:
    start = time.time()
    log.info(f"▶ {task.task_id} [{task.difficulty}]: {task.question[:80]}...")
    try:
        agent  = make_agent()
        result = agent.run(task)
        elapsed = round(time.time() - start, 1)

        if result.success and result.answer:
            write_prediction(task.task_id, result.answer.columns, result.answer.rows)
            log.info(
                f"  ✓ {task.task_id}  {elapsed}s  "
                f"{result.steps} steps  "
                f"{len(result.answer.columns)} cols  "
                f"{len(result.answer.rows)} rows"
            )
            return {"task_id": task.task_id, "difficulty": task.difficulty,
                    "success": True, "elapsed": elapsed, "steps": result.steps}
        else:
            write_empty_prediction(task.task_id)
            log.warning(f"  ✗ {task.task_id}  {elapsed}s  FAILED: {result.failure}")
            return {"task_id": task.task_id, "difficulty": task.difficulty,
                    "success": False, "elapsed": elapsed, "reason": result.failure}

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        write_empty_prediction(task.task_id)
        log.error(f"  ✗ {task.task_id}  {elapsed}s  EXCEPTION: {e}", exc_info=True)
        return {"task_id": task.task_id, "difficulty": task.difficulty,
                "success": False, "elapsed": elapsed, "reason": str(e)}


def main() -> None:
    log.info("=" * 60)
    log.info("KDD Cup 2026 — Data Agent v5")
    log.info(f"  Model  : {MODEL_NAME}")
    log.info(f"  API URL: {MODEL_API_URL}")
    log.info(f"  Input  : {INPUT_ROOT}")
    log.info(f"  Output : {OUTPUT_ROOT}")
    log.info(f"  Workers: {MAX_WORKERS}")
    log.info("=" * 60)

    if not MODEL_API_KEY:
        log.error("MODEL_API_KEY is not set! Copy .env.example → .env and add your key.")
        sys.exit(1)

    if not INPUT_ROOT.exists():
        log.error(f"INPUT_ROOT does not exist: {INPUT_ROOT}")
        sys.exit(1)

    tasks = load_tasks(INPUT_ROOT)
    log.info(f"Found {len(tasks)} tasks")
    if not tasks:
        log.warning("No tasks found — exiting.")
        return

    overall_start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(run_one_task, t): t for t in tasks}
        for future in as_completed(future_map):
            try:
                r = future.result(timeout=TASK_TIMEOUT + 30)
                results.append(r)
            except Exception as e:
                t = future_map[future]
                log.error(f"  ✗ {t.task_id} future error: {e}")
                write_empty_prediction(t.task_id)
                results.append({"task_id": t.task_id, "difficulty": t.difficulty,
                                 "success": False, "reason": str(e)})

    total   = len(results)
    success = sum(1 for r in results if r.get("success"))
    elapsed = round(time.time() - overall_start, 1)

    summary = {
        "total": total, "success": success,
        "rate": round(success / total, 4) if total else 0,
        "elapsed_seconds": elapsed,
        "tasks": results,
    }
    (LOG_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"Done: {success}/{total} tasks  ({elapsed}s total)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
