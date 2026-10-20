import json

import pytest

from agent.agents.agent import (
    _extract_json,
    _inject_user,
    _trim_history,
)


def test_extract_json_from_fence_think_tag_and_trailing_comma():
    text = """<think>scratch</think>
```json
{"thought": "ok", "action": "answer", "action_input": {"columns": ["a"],},}
```
"""
    payload = _extract_json(text)
    assert payload["action"] == "answer"
    assert payload["action_input"]["columns"] == ["a"]


def test_extract_json_rejects_prose():
    with pytest.raises(ValueError):
        _extract_json("I cannot answer that.")


def test_inject_user_merges_consecutive_user_messages():
    history = [{"role": "user", "content": "Observation: {}"}]
    _inject_user(history, "Try a different tool.")
    assert len(history) == 1
    assert "Try a different tool." in history[0]["content"]


def test_trim_history_keeps_the_question_and_alternates():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "FIRST QUESTION"},
    ]
    for index in range(10):
        history.append({"role": "assistant", "content": f"a{index}"})
        history.append({"role": "user", "content": f"u{index}"})

    trimmed = _trim_history(history, trim_turns=2)
    roles = [message["role"] for message in trimmed]
    assert roles[0] == "system"
    assert trimmed[1]["content"] == "FIRST QUESTION"
    assert roles[2] == "assistant"
    assert "u9" in trimmed[-1]["content"]
    for left, right in zip(roles, roles[1:]):
        if left != "system":
            assert left != right
    assert json.dumps(trimmed)
