from pathlib import Path

import pytest

from ask import build_task


def test_build_task_uses_the_given_folder(tmp_path):
    folder = tmp_path / "drop"
    folder.mkdir()
    task = build_task(folder, "How many rows?", "easy")
    assert task.context_dir == folder.resolve()
    assert task.question == "How many rows?"
    assert task.context_path == folder.resolve()


def test_build_task_rejects_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_task(tmp_path / "missing", "q", "easy")


def test_module_path_is_the_project_root():
    assert Path(__file__).resolve().parents[1].joinpath("ask.py").exists()
