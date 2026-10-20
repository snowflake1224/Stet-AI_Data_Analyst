import json
import shutil
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import serve
from agent.agents.agent import AgentResult, DataAgent
from agent.agents.model import config_for_provider, redact_secret
from agent.benchmark.schema import Answer, Task
from agent.tools.tools import execute_tool, tool_query_tables


def test_health_endpoint_is_available_for_the_host():
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/health"
        ) as response:
            assert response.status == 200
            assert json.load(response)["ok"] is True
    finally:
        server.shutdown()


def test_provider_configs_are_explicit_and_secret_is_redacted():
    secret = "sk-test-" + "x" * 30
    gemini = config_for_provider("gemini", secret)
    openai = config_for_provider("openai", secret)
    assert gemini.model == "gemini-3.5-flash-lite"
    assert gemini.min_interval == 7
    assert openai.model == "auto"
    assert openai.api_url == "https://api.openai.com/v1"
    assert secret not in redact_secret(f"failed for {secret}", secret)
    with pytest.raises(ValueError):
        config_for_provider("unknown", secret)


def test_upload_validation_sanitizes_names_and_keeps_only_text():
    folder, files = serve.prepare_uploads(
        [{"name": "../../sales report.csv", "content": "region,amount\nEast,10\n"}],
        pasted_markdown="# Definitions\nAmount is gross.",
    )
    try:
        assert {path.name for path in folder.iterdir()} == {
            "sales_report.csv", "knowledge.md",
        }
        assert files[0]["preview"].startswith("region,amount")
        assert not (folder.parent / "sales report.csv").exists()
    finally:
        shutil.rmtree(folder)


@pytest.mark.parametrize(
    "files,message",
    [
        ([{"name": "bad.exe", "content": "x"}], "CSV file is required"),
        ([{"name": "bad.csv", "content": "\x00"}], "binary"),
        ([{"name": "bad.csv", "content": ",amount\nEast,1"}], "header"),
    ],
)
def test_upload_validation_rejects_unsafe_input(files, message):
    with pytest.raises(ValueError, match=message):
        serve.prepare_uploads(files)


def test_query_tables_supports_joins_and_numeric_aggregation(tmp_path):
    (tmp_path / "orders.csv").write_text(
        "customer_id,amount\n1,10\n1,2.5\n2,7\n", encoding="utf-8"
    )
    (tmp_path / "customers.csv").write_text(
        "id,name\n1,Alice\n2,Bob\n", encoding="utf-8"
    )
    task = Task("upload", "easy", "totals", tmp_path, context_path=tmp_path)
    result = tool_query_tables(
        task,
        """
        SELECT c.name, SUM(CAST(o.amount AS REAL)) AS total
        FROM customers c JOIN orders o ON c.id = o.customer_id
        GROUP BY c.name ORDER BY c.name
        """,
    )
    assert result["columns"] == ["name", "total"]
    assert result["rows"] == [["Alice", 12.5], ["Bob", 7.0]]
    assert result["tables"][0]["table"] == "customers"


def test_query_tables_treats_blanks_as_null_and_compares_numbers(tmp_path):
    (tmp_path / "HR_Analytics.csv").write_text(
        "YearsWithCurrManager,YearsInCurrentRole,Attrition,EmployeeCode\n"
        ",0,Yes,001\n"
        " ,1,No,002\n"
        "0,0,No,003\n"
        "1,2,Yes,004\n"
        "10,9,No,005\n",
        encoding="utf-8",
    )
    task = Task("upload", "easy", "tenure", tmp_path, context_path=tmp_path)
    result = tool_query_tables(
        task,
        """
        SELECT 'manager_years_missing' AS grp, COUNT(*) AS n,
               AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) AS attrition_rate
        FROM HR_Analytics WHERE YearsWithCurrManager IS NULL
        UNION ALL
        SELECT 'manager_years_known', COUNT(*),
               AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END)
        FROM HR_Analytics WHERE YearsWithCurrManager IS NOT NULL
        UNION ALL
        SELECT 'manager_years_lt_role_years', COUNT(*),
               AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END)
        FROM HR_Analytics
        WHERE YearsWithCurrManager IS NOT NULL
          AND YearsWithCurrManager < YearsInCurrentRole
        UNION ALL
        SELECT 'manager_years_gte_role_years', COUNT(*),
               AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END)
        FROM HR_Analytics
        WHERE YearsWithCurrManager IS NOT NULL
          AND YearsWithCurrManager >= YearsInCurrentRole
        """,
    )
    assert result["rows"] == [
        ["manager_years_missing", 2, 0.5],
        ["manager_years_known", 3, 1 / 3],
        ["manager_years_lt_role_years", 1, 1.0],
        ["manager_years_gte_role_years", 2, 0.0],
    ]
    assert result["tables"][0]["types"] == ["INTEGER", "INTEGER", "TEXT", "TEXT"]
    codes = tool_query_tables(
        task,
        "SELECT EmployeeCode FROM HR_Analytics "
        "WHERE YearsWithCurrManager IS NULL ORDER BY EmployeeCode",
    )
    assert codes["rows"] == [["001"], ["002"]]


def test_public_tool_policy_blocks_generated_python(tmp_path):
    task = Task("upload", "easy", "q", tmp_path, context_path=tmp_path)
    observation, terminal, answer = execute_tool(
        "run_python",
        {"code": "print('unsafe')"},
        task,
        client=None,
        allowed_tools=DataAgent.PUBLIC_TOOLS,
    )
    assert observation["error"] == "PermissionError"
    assert terminal is False
    assert answer is None


def test_job_snapshot_and_csv_are_isolated():
    job_id, job = serve._new_job("upload", [{"name": "sales.csv"}])
    with serve._state_lock:
        job.update({
            "status": "done",
            "columns": ["region", "total"],
            "rows": [["East", 10]],
        })
    try:
        assert serve.job_snapshot(job_id)["rows"] == [["East", 10]]
        assert serve.job_csv(job_id) == b"region,total\r\nEast,10\r\n"
        with pytest.raises(KeyError):
            serve.job_snapshot("some-other-job")
    finally:
        with serve._state_lock:
            serve._jobs.pop(job_id, None)


def test_owner_cooldown_is_enforced(monkeypatch):
    monkeypatch.setattr(serve, "OWNER_COOLDOWN_SECONDS", 60)
    monkeypatch.setattr(serve, "OWNER_DAILY_LIMIT", 20)
    with serve._state_lock:
        serve._owner_last_by_ip.clear()
        serve._owner_runs = 0
    serve._check_owner_limit("127.0.0.99")
    with pytest.raises(RuntimeError, match="wait"):
        serve._check_owner_limit("127.0.0.99")


def test_temporary_context_is_deleted_when_a_job_finishes(tmp_path, monkeypatch):
    class StubAgent:
        def run(self, task):
            return AgentResult(
                task.task_id, Answer(["value"], [[1]]), True, 1
            )

    class StubFactory:
        @classmethod
        def for_public_upload(cls, client, task_timeout):
            return StubAgent()

    class Release:
        released = False

        def release(self):
            self.released = True

    context = tmp_path / "temporary"
    context.mkdir()
    (context / "data.csv").write_text("value\n1\n", encoding="utf-8")
    _job_id, job = serve._new_job("upload", [])
    release = Release()
    monkeypatch.setattr(serve, "ModelClient", lambda config: object())
    monkeypatch.setattr(serve, "DataAgent", StubFactory)
    config = config_for_provider("gemini", "AIza" + "x" * 30)
    serve._run_job(job, "value?", context, config, release, temporary=True)
    assert job["status"] == "done"
    assert not context.exists()
    assert release.released is True
    with serve._state_lock:
        serve._jobs.pop(job["id"], None)
