from __future__ import annotations
import json
import logging
from pathlib import Path
from agent.benchmark.schema import Task

log = logging.getLogger("agent.dataset")


def load_tasks(input_root: Path) -> list[Task]:
    tasks = []
    for task_dir in sorted(input_root.iterdir()):
        task_json = task_dir / "task.json"
        if not task_dir.is_dir() or not task_json.exists():
            continue
        try:
            data = json.loads(task_json.read_text(encoding="utf-8"))
            tasks.append(Task(
                task_id=data["task_id"],
                difficulty=data.get("difficulty", "unknown"),
                question=data["question"],
                task_dir=task_dir,
            ))
        except Exception as e:
            log.warning(f"Could not load {task_dir.name}: {e}")

    # Sort easy → extreme: fast tasks write CSVs early, safe under the 12-hour wall
    order = {"easy": 0, "medium": 1, "hard": 2, "extreme": 3, "unknown": 4}
    tasks.sort(key=lambda t: order.get(t.difficulty.lower(), 4))
    log.info(
        f"Loaded {len(tasks)} tasks: "
        + ", ".join(
            f"{d}={sum(1 for t in tasks if t.difficulty == d)}"
            for d in ["easy", "medium", "hard", "extreme"]
        )
    )
    return tasks
