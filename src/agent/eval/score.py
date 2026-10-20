from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?")
_EMPTY = {"", "nan", "null", "none"}


@dataclass(frozen=True)
class Score:
    """Column-level score: recall minus a penalty for extra columns."""

    score: float
    recall: float
    matched_columns: int
    gold_columns: int
    extra_columns: int
    predicted_columns: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def normalize_cell(value: Any) -> str:
    """Normalize one cell so row order and number formatting do not matter."""
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{round(float(value), 2):.2f}"

    text = str(value).strip()
    if text.lower() in _EMPTY:
        return ""
    if _NUMBER.fullmatch(text):
        return f"{round(float(text), 2):.2f}"
    return text


def _signatures(columns: list[Any], rows: list[list[Any]]) -> list[tuple[str, ...]]:
    width = len(columns)
    buckets: list[list[str]] = [[] for _ in range(width)]
    for row in rows:
        cells = list(row) + [""] * max(0, width - len(row))
        for index in range(width):
            buckets[index].append(normalize_cell(cells[index]))
    return [tuple(sorted(bucket)) for bucket in buckets]


def score_answer(
    gold_columns: list[Any],
    gold_rows: list[list[Any]],
    pred_columns: list[Any],
    pred_rows: list[list[Any]],
    *,
    lambda_: float = 1.0,
) -> Score:
    """Score a prediction using recall minus lambda * extra / predicted.

    Column headers are ignored. A gold column matches one predicted column
    when the multisets of their normalized values are equal. Each predicted
    column can match at most once. Unmatched predicted columns are extra.
    """
    gold_sigs = _signatures(gold_columns, gold_rows)
    pred_sigs = _signatures(pred_columns, pred_rows)
    used: set[int] = set()
    matched = 0
    for gold in gold_sigs:
        for index, predicted in enumerate(pred_sigs):
            if index in used or predicted != gold:
                continue
            used.add(index)
            matched += 1
            break

    gold_count = len(gold_sigs)
    pred_count = len(pred_sigs)
    extra = pred_count - len(used)
    recall = 1.0 if gold_count == 0 else matched / gold_count
    penalty = 0.0 if pred_count == 0 else extra / pred_count
    return Score(
        score=recall - lambda_ * penalty,
        recall=recall,
        matched_columns=matched,
        gold_columns=gold_count,
        extra_columns=extra,
        predicted_columns=pred_count,
    )


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def score_benchmark(
    benchmark_root: Path,
    predictions_root: Path,
    *,
    lambda_: float = 1.0,
) -> dict[str, Any]:
    """Score ``predictions/<task_id>/prediction.csv`` against each gold.csv."""
    tasks: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
        task_json = task_dir / "task.json"
        gold_path = task_dir / "gold.csv"
        if not task_json.exists() or not gold_path.exists():
            continue
        meta = json.loads(task_json.read_text(encoding="utf-8"))
        task_id = str(meta["task_id"])
        gold_columns, gold_rows = load_csv(gold_path)
        prediction = predictions_root / task_id / "prediction.csv"
        if not prediction.exists():
            result = Score(0.0, 0.0, 0, len(gold_columns), 0, 0)
            tasks.append({"task_id": task_id, "missing": True, **result.as_dict()})
            continue
        pred_columns, pred_rows = load_csv(prediction)
        result = score_answer(
            gold_columns, gold_rows, pred_columns, pred_rows, lambda_=lambda_
        )
        tasks.append({"task_id": task_id, "missing": False, **result.as_dict()})

    mean = sum(item["score"] for item in tasks) / len(tasks) if tasks else 0.0
    return {"lambda": lambda_, "mean_score": mean, "tasks": tasks}
