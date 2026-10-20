import json
from pathlib import Path

from agent.agents.agent import DataAgent
from agent.benchmark.schema import Task


class ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, temperature=0.0, max_tokens=4096) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("model was called after the script ended")
        return self.responses.pop(0)


def _answer(columns, rows) -> str:
    return json.dumps({
        "thought": "submit",
        "action": "answer",
        "action_input": {"columns": columns, "rows": rows},
    })


def test_executor_answers_on_the_first_step(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    client = ScriptedClient([
        json.dumps({"approach": "answer from the question"}),
        _answer(["region"], [["North"]]),
        json.dumps({"ok": True, "remove_columns": []}),
    ])
    agent = DataAgent(client=client, task_timeout=30)
    result = agent.run(Task("demo", "easy", "Which region?", tmp_path))
    assert result.success is True
    assert result.answer.columns == ["region"]
    assert result.answer.rows == [["North"]]
    assert result.steps == 1
    assert client.calls == 3


def test_execute_loop_stops_when_the_timeout_is_already_spent(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    client = ScriptedClient([json.dumps({"approach": "too late"})])
    agent = DataAgent(client=client, task_timeout=0)
    result = agent.run(Task("demo", "easy", "Which region?", tmp_path))
    assert result.success is False
    assert "timed out" in result.failure.lower()
    assert client.calls == 1
    assert Path(context).is_dir()
