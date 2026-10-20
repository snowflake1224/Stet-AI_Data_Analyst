import json

from agent.benchmark.dataset import load_tasks


def test_load_tasks_sorts_easy_before_hard(tmp_path):
    for name, difficulty in (("b_hard", "hard"), ("a_easy", "easy")):
        task_dir = tmp_path / name
        (task_dir / "context").mkdir(parents=True)
        (task_dir / "task.json").write_text(
            json.dumps({"task_id": name, "difficulty": difficulty, "question": "q?"}),
            encoding="utf-8",
        )
    loaded = load_tasks(tmp_path)
    assert [task.task_id for task in loaded] == ["a_easy", "b_hard"]
    assert loaded[0].context_dir == tmp_path / "a_easy" / "context"
