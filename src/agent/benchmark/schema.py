from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    difficulty: str
    question: str
    task_dir: Path
    context_path: Path | None = None

    @property
    def context_dir(self) -> Path:
        if self.context_path is not None:
            return self.context_path
        return self.task_dir / "context"


@dataclass
class Answer:
    columns: list[str]
    rows: list[list[Any]]

    def to_csv_rows(self) -> list[list]:
        return [self.columns] + self.rows
