import json
import sqlite3

from agent.benchmark.schema import Task
from agent.tools.tools import execute_tool, tool_read_csv, tool_read_excel


def _task(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    return Task("t", "easy", "q", tmp_path), context


def test_path_escape_is_blocked(tmp_path):
    task, _context = _task(tmp_path)
    secret = tmp_path / "secret.csv"
    secret.write_text("value\n1\n", encoding="utf-8")
    obs, terminal, answer = execute_tool(
        "read_csv", {"path": "../secret.csv"}, task, client=None
    )
    assert terminal is False
    assert answer is None
    assert obs["error"] == "ValueError"
    assert "escape" in obs["message"].lower()


def test_read_csv_returns_rows(tmp_path):
    task, context = _task(tmp_path)
    (context / "sales.csv").write_text("region,amount\nNorth,5\n", encoding="utf-8")
    obs = tool_read_csv(task, "sales.csv", max_rows=10)
    assert obs["columns"] == ["region", "amount"]
    assert obs["rows"] == [["North", "5"]]


def test_sql_is_read_only(tmp_path):
    task, context = _task(tmp_path)
    db = context / "shop.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (n INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
    select_obs, _, _ = execute_tool(
        "execute_sql",
        {"path": "shop.sqlite", "sql": "SELECT n FROM t"},
        task,
        client=None,
    )
    assert select_obs["rows"] == [[1]]

    insert_obs, terminal, _ = execute_tool(
        "execute_sql",
        {"path": "shop.sqlite", "sql": "INSERT INTO t VALUES (2)"},
        task,
        client=None,
    )
    assert terminal is False
    assert insert_obs["error"] == "ValueError"

    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 1


def test_read_excel_selects_a_sheet(tmp_path):
    pandas = __import__("pandas")
    task, context = _task(tmp_path)
    path = context / "report.xlsx"
    with pandas.ExcelWriter(path) as writer:
        pandas.DataFrame({"item": ["Pen"], "price": ["1.5"]}).to_excel(
            writer, sheet_name="prices", index=False
        )
        pandas.DataFrame({"item": ["Pen"], "quantity": ["4"]}).to_excel(
            writer, sheet_name="qty", index=False
        )
    obs = tool_read_excel(task, "report.xlsx", sheet="qty", max_rows=10)
    assert obs["sheet"] == "qty"
    assert obs["columns"] == ["item", "quantity"]
    assert obs["rows"] == [["Pen", "4"]]


def test_answer_normalizes_nan(tmp_path):
    task, _context = _task(tmp_path)
    _obs, terminal, answer = execute_tool(
        "answer",
        {"columns": ["region"], "rows": [["North"], [float("nan")]]},
        task,
        client=None,
    )
    assert terminal is True
    assert answer.rows[1] == [None]
    assert json.dumps(answer.rows)
