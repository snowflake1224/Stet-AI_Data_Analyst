#!/usr/bin/env python3
"""Ask one question about a folder of files and write a CSV answer."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.agents.agent import DataAgent
from agent.agents.model import ModelClient
from agent.benchmark.schema import Task


def build_task(context: Path, question: str, difficulty: str) -> Task:
    """Point a task at an arbitrary folder, not only a contest context/ directory."""
    folder = context.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Context folder not found: {folder}")
    return Task(
        task_id="ask",
        difficulty=difficulty,
        question=question,
        task_dir=folder.parent,
        context_path=folder,
    )


def write_csv(path: Path, columns: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([columns] + rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer one question about a folder of data files."
    )
    parser.add_argument("--context", type=Path, required=True, help="Folder of data files")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/ask/prediction.csv"),
        help="Where to write the CSV answer",
    )
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard", "extreme"],
    )
    args = parser.parse_args(argv)

    if not os.environ.get("MODEL_API_KEY"):
        print("MODEL_API_KEY is not set. Copy .env.example to .env and add a key.", file=sys.stderr)
        return 1

    try:
        task = build_task(args.context, args.question, args.difficulty)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    timeout = float(os.environ.get("TASK_TIMEOUT_SECONDS", "900"))
    agent = DataAgent(client=ModelClient(), task_timeout=timeout)
    result = agent.run(task)
    if result.success and result.answer is not None:
        write_csv(args.output, result.answer.columns, result.answer.rows)
        print(f"Wrote {args.output} ({len(result.answer.rows)} rows, {result.steps} steps)")
        return 0

    write_csv(args.output, [], [])
    print(f"No answer: {result.failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
