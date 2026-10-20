#!/usr/bin/env python3
"""Deployable CSV/Markdown demo; no uploaded file or BYOK secret is persisted."""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from agent.agents.agent import DataAgent
from agent.agents.model import (
    GEMINI_MODEL,
    ModelClient,
    ModelConfig,
    config_for_provider,
    redact_secret,
)
from agent.benchmark.schema import Task

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
DEMO_DIR = ROOT / "examples" / "benchmark" / "knowledge_revenue"

MAX_CSV_FILES = 3
MAX_MD_FILES = 1
MAX_FILE_BYTES = 512_000
MAX_TOTAL_BYTES = 1_500_000
MAX_QUESTION_CHARS = 2_000
JOB_TTL_SECONDS = 30 * 60
OWNER_DAILY_LIMIT = int(os.environ.get("OWNER_DAILY_LIMIT", "20"))
OWNER_COOLDOWN_SECONDS = int(os.environ.get("OWNER_COOLDOWN_SECONDS", "60"))
BYOK_CONCURRENCY = int(os.environ.get("BYOK_CONCURRENCY", "2"))

_PLACEHOLDERS = {"", "your_gemini_api_key_here", "EMPTY"}
_jobs: dict[str, dict] = {}
_state_lock = threading.RLock()
_example_lock = threading.Lock()
_byok_slots = threading.BoundedSemaphore(max(1, BYOK_CONCURRENCY))
_owner_day = date.today()
_owner_runs = 0
_owner_last_by_ip: dict[str, float] = {}


def key_is_usable(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip()
    return (
        text not in _PLACEHOLDERS
        and "\r" not in text
        and "\n" not in text
        and len(text) >= 20
    )


def _validate_question(value: object) -> str:
    question = str(value or "").strip()
    if not question:
        raise ValueError("Enter a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question must be at most {MAX_QUESTION_CHARS} characters.")
    return question


def _safe_filename(value: object, extension: str) -> str:
    raw = Path(str(value or "")).name
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(raw).stem).strip("_")[:60]
    if not stem:
        stem = "upload"
    return f"{stem}{extension}"


def _validate_csv(content: str, name: str) -> None:
    try:
        rows = csv.reader(io.StringIO(content))
        header = next(rows)
    except (csv.Error, StopIteration) as exc:
        raise ValueError(f"{name} is not a valid CSV with a header row.") from exc
    if not header or any(not item.strip() for item in header):
        raise ValueError(f"{name} must have a non-empty CSV header.")
    if len(header) > 100:
        raise ValueError(f"{name} has too many columns.")


def prepare_uploads(
    files: object,
    pasted_csv: object = "",
    pasted_markdown: object = "",
) -> tuple[Path, list[dict]]:
    """Validate text uploads and copy them into a new temporary context."""
    if not isinstance(files, list):
        raise ValueError("files must be a list.")
    entries: list[tuple[str, str]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each upload must include a name and text content.")
        entries.append((str(item.get("name", "")), str(item.get("content", ""))))
    if str(pasted_csv or "").strip():
        entries.append(("pasted.csv", str(pasted_csv)))
    if str(pasted_markdown or "").strip():
        entries.append(("knowledge.md", str(pasted_markdown)))

    csv_count = sum(Path(name).suffix.lower() == ".csv" for name, _ in entries)
    md_count = sum(Path(name).suffix.lower() == ".md" for name, _ in entries)
    if not entries:
        raise ValueError("Upload or paste at least one CSV file.")
    if csv_count < 1:
        raise ValueError("At least one CSV file is required.")
    if csv_count > MAX_CSV_FILES or md_count > MAX_MD_FILES:
        raise ValueError(
            f"Use at most {MAX_CSV_FILES} CSV files and {MAX_MD_FILES} Markdown file."
        )
    if csv_count + md_count != len(entries):
        raise ValueError("Only .csv and .md files are accepted.")

    total = 0
    normalized: list[tuple[str, str]] = []
    used: set[str] = set()
    for raw_name, content in entries:
        extension = Path(raw_name).suffix.lower()
        encoded = content.encode("utf-8")
        if b"\x00" in encoded:
            raise ValueError(f"{raw_name or 'upload'} appears to be binary.")
        if len(encoded) > MAX_FILE_BYTES:
            raise ValueError(f"{raw_name} exceeds the 500 KB file limit.")
        total += len(encoded)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Uploads exceed the 1.5 MB total limit.")
        name = _safe_filename(raw_name, extension)
        candidate = name
        index = 2
        while candidate.lower() in used:
            candidate = f"{Path(name).stem}_{index}{extension}"
            index += 1
        used.add(candidate.lower())
        if extension == ".csv":
            _validate_csv(content, candidate)
        normalized.append((candidate, content))

    folder = Path(tempfile.mkdtemp(prefix="kdd-upload-"))
    summaries: list[dict] = []
    try:
        for name, content in normalized:
            (folder / name).write_text(content, encoding="utf-8")
            summaries.append({
                "name": name,
                "bytes": len(content.encode("utf-8")),
                "preview": content[:600],
            })
        return folder, summaries
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _cached_example() -> tuple[list[str], list[list[str]]]:
    with (DEMO_DIR / "gold.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def demo_payload() -> dict:
    task = json.loads((DEMO_DIR / "task.json").read_text(encoding="utf-8"))
    columns, rows = _cached_example()
    files = []
    for path in sorted((DEMO_DIR / "context").iterdir()):
        if path.is_file():
            files.append({"name": path.name, "preview": path.read_text(encoding="utf-8")[:1000]})
    return {
        "owner_configured": key_is_usable(os.environ.get("MODEL_API_KEY")),
        "model": GEMINI_MODEL,
        "question": task["question"],
        "files": files,
        "cached_columns": columns,
        "cached_rows": rows,
        "limits": {
            "csv_files": MAX_CSV_FILES,
            "markdown_files": MAX_MD_FILES,
            "file_bytes": MAX_FILE_BYTES,
        },
    }


def _append_trace(job: dict, message: str) -> None:
    with _state_lock:
        job["trace"].append(message)
        job["trace"] = job["trace"][-80:]
        job["updated"] = time.time()


class _JobLog(logging.Handler):
    _action = re.compile(r"step \d+/\d+: ([A-Za-z_]+)")
    _labels = {
        "list_context": "Inspecting uploaded files",
        "read_csv": "Reading CSV data",
        "read_doc": "Reading Markdown context",
        "search_doc": "Searching Markdown context",
        "query_tables": "Running a read-only table query",
        "answer": "Preparing the answer table",
    }

    def __init__(self, job: dict, secret: str) -> None:
        super().__init__(level=logging.INFO)
        self.job = job
        self.secret = secret

    def emit(self, record: logging.LogRecord) -> None:
        text = redact_secret(record.getMessage(), self.secret)
        if f"[{self.job['id']}]" not in text:
            return
        match = self._action.search(text)
        if match:
            action = match.group(1)
            _append_trace(self.job, self._labels.get(action, f"Using {action}"))
        elif "answered at step" in text:
            _append_trace(self.job, "Verifying the result")


def _new_job(mode: str, files: list[dict]) -> tuple[str, dict]:
    job_id = secrets.token_urlsafe(18)
    now = time.time()
    job = {
        "id": job_id,
        "mode": mode,
        "status": "running",
        "trace": ["Planning how to answer the question"],
        "columns": [],
        "rows": [],
        "error": "",
        "steps": 0,
        "elapsed": 0.0,
        "files": files,
        "created": now,
        "updated": now,
    }
    with _state_lock:
        _jobs[job_id] = job
    return job_id, job


def _clean_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _state_lock:
        expired = [
            job_id for job_id, job in _jobs.items()
            if job["status"] != "running" and job["updated"] < cutoff
        ]
        for job_id in expired:
            del _jobs[job_id]


def _run_job(
    job: dict,
    question: str,
    context: Path,
    config: ModelConfig,
    release: threading.Lock | threading.BoundedSemaphore,
    temporary: bool,
) -> None:
    started = time.monotonic()
    handler = _JobLog(job, config.api_key)
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        task = Task(
            task_id=job["id"],
            difficulty="easy",
            question=question,
            task_dir=context.parent,
            context_path=context,
        )
        agent = DataAgent.for_public_upload(
            client=ModelClient(config),
            task_timeout=float(os.environ.get("PUBLIC_TASK_TIMEOUT_SECONDS", "180")),
        )
        result = agent.run(task)
        with _state_lock:
            if result.success and result.answer is not None:
                job["status"] = "done"
                job["columns"] = result.answer.columns
                job["rows"] = result.answer.rows[:1000]
                job["steps"] = result.steps
                job["trace"].append("Complete")
            else:
                job["status"] = "error"
                job["error"] = result.failure or "The agent did not produce an answer."
            job["elapsed"] = round(time.monotonic() - started, 2)
            job["updated"] = time.time()
    except Exception as exc:
        with _state_lock:
            job["status"] = "error"
            job["error"] = redact_secret(str(exc), config.api_key)
            job["elapsed"] = round(time.monotonic() - started, 2)
            job["updated"] = time.time()
    finally:
        logger.removeHandler(handler)
        if temporary:
            shutil.rmtree(context, ignore_errors=True)
        release.release()
        # Drop the last reference owned by this thread as soon as the run ends.
        config = ModelConfig("discarded", "", "", "")


def _check_owner_limit(ip: str) -> None:
    global _owner_day, _owner_runs
    now = time.monotonic()
    with _state_lock:
        today = date.today()
        if today != _owner_day:
            _owner_day = today
            _owner_runs = 0
            _owner_last_by_ip.clear()
        if _owner_runs >= OWNER_DAILY_LIMIT:
            raise RuntimeError("Today's live example limit has been reached. The cached result is still available.")
        wait = OWNER_COOLDOWN_SECONDS - (now - _owner_last_by_ip.get(ip, 0.0))
        if wait > 0:
            raise RuntimeError(f"Please wait {int(wait) + 1} seconds before another live example.")
        _owner_runs += 1
        _owner_last_by_ip[ip] = now


def start_example(question: object, ip: str) -> dict:
    question_text = _validate_question(question)
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key_is_usable(api_key):
        raise ValueError("The owner Gemini key is not configured. View the cached sample result instead.")
    if not _example_lock.acquire(blocking=False):
        raise RuntimeError("The live example is already running. Try again shortly.")
    try:
        _check_owner_limit(ip)
    except Exception:
        _example_lock.release()
        raise
    files = [{"name": path.name, "bytes": path.stat().st_size}
             for path in sorted((DEMO_DIR / "context").iterdir()) if path.is_file()]
    job_id, job = _new_job("example", files)
    config = config_for_provider("gemini", api_key)
    threading.Thread(
        target=_run_job,
        args=(job, question_text, DEMO_DIR / "context", config, _example_lock, False),
        daemon=True,
    ).start()
    return {"job_id": job_id}


def start_upload(data: dict) -> dict:
    question = _validate_question(data.get("question"))
    provider = str(data.get("provider", "")).strip().lower()
    api_key = str(data.get("api_key", "")).strip()
    if not key_is_usable(api_key):
        raise ValueError("Paste a valid provider API key. It is used only for this run.")
    config = config_for_provider(provider, api_key)
    if not _byok_slots.acquire(blocking=False):
        raise RuntimeError("The demo is busy. Wait for another upload run to finish.")

    context: Path | None = None
    try:
        context, files = prepare_uploads(
            data.get("files", []),
            data.get("pasted_csv", ""),
            data.get("pasted_markdown", ""),
        )
        job_id, job = _new_job("upload", files)
        threading.Thread(
            target=_run_job,
            args=(job, question, context, config, _byok_slots, True),
            daemon=True,
        ).start()
        return {"job_id": job_id}
    except Exception:
        if context is not None:
            shutil.rmtree(context, ignore_errors=True)
        _byok_slots.release()
        raise


def job_snapshot(job_id: str) -> dict:
    _clean_jobs()
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise KeyError("Job not found or expired.")
        return {
            key: job[key] for key in (
                "id", "mode", "status", "trace", "columns", "rows",
                "error", "steps", "elapsed", "files",
            )
        }


def job_csv(job_id: str) -> bytes:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] != "done":
            raise KeyError("Completed job not found.")
        output = io.StringIO(newline="")
        csv.writer(output).writerows([job["columns"]] + job["rows"])
        return output.getvalue().encode("utf-8")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Data Analyst</title>
  <style>
    :root{color-scheme:light;--ink:#1c1915;--muted:#655e54;--line:#ded7cc;--paper:#f7f4ee;--card:#fff;--accent:#0c6b4b}
    *{box-sizing:border-box} body{margin:0;font:15px/1.5 Segoe UI,sans-serif;color:var(--ink);background:var(--paper)}
    main{max-width:920px;margin:auto;padding:30px 18px 60px} h1{margin:0;font-size:1.9rem} h2{font-size:1.15rem}
    section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}
    .muted{color:var(--muted)} .tabs,.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .tabs button{background:transparent;color:var(--ink);border:1px solid var(--line)} .tabs button.active{background:var(--ink);color:white}
    button,.download{background:var(--accent);color:white;border:0;border-radius:8px;padding:10px 15px;font:inherit;text-decoration:none;cursor:pointer}
    button:disabled{opacity:.5;cursor:wait} label{display:block;font-weight:600;margin:12px 0 5px}
    input,select,textarea{width:100%;font:inherit;padding:10px;border:1px solid var(--line);border-radius:8px;background:white}
    textarea{min-height:85px;resize:vertical}.drop{border:2px dashed var(--line);padding:16px;border-radius:9px}
    pre{white-space:pre-wrap;background:#f2eee7;padding:11px;border-radius:8px;max-height:230px;overflow:auto}
    table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid var(--line);text-align:left}
    .hidden{display:none}.error{color:#922}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    @media(max-width:650px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body><main>
  <h1>AI Data Analyst</h1>
  <p class="muted">Ask a question over CSV and Markdown. The agent plans, reads, queries and returns a verified table.</p>
  <div class="tabs"><button id="tab-example" class="active">Try example</button><button id="tab-upload">Use your files</button></div>

  <div id="example-pane">
    <section><h2>Curated business example</h2><div id="example-files"></div>
      <label for="example-question">Question</label><textarea id="example-question"></textarea>
      <div class="row"><button id="run-example">Run live</button><span id="example-note" class="muted"></span></div>
      <p class="muted">Cached sample result is shown below so the portfolio still works when the free quota is unavailable.</p>
    </section>
  </div>

  <div id="upload-pane" class="hidden">
    <section><h2>Your CSV and Markdown</h2>
      <div class="drop"><label for="files">Choose or drop files</label><input id="files" type="file" multiple accept=".csv,.md">
      <p class="muted">Up to 3 CSV files and 1 Markdown file, 500 KB each. Files are deleted after the run.</p></div>
      <div class="grid"><div><label for="paste-csv">Or paste CSV</label><textarea id="paste-csv" placeholder="region,amount&#10;East,100"></textarea></div>
      <div><label for="paste-md">Optional Markdown context</label><textarea id="paste-md" placeholder="# Definitions&#10;Net revenue equals..."></textarea></div></div>
      <div id="file-list" class="muted"></div>
      <label for="upload-question">Question</label><textarea id="upload-question" placeholder="What is the total amount for each region?"></textarea>
      <div class="grid"><div><label for="provider">Provider</label><select id="provider"><option value="gemini">Gemini</option><option value="openai">OpenAI</option></select></div>
      <div><label for="user-key">API key for this run</label><input id="user-key" type="password" autocomplete="off" placeholder="Never stored"></div></div>
      <p class="muted">The key is held in server memory only until this job ends. A deployed site must use HTTPS.</p>
      <button id="run-upload">Run agent</button>
    </section>
  </div>

  <section><h2>Result</h2><p id="result-note" class="muted">Cached example result.</p>
    <div id="result-table"></div><p><a id="download" class="download hidden">Download CSV</a></p>
    <h2>Progress</h2><pre id="trace">Ready.</pre>
  </section>
</main>
<script>
const $=id=>document.getElementById(id); let currentJob=null,pollTimer=null,statusData=null;
const esc=v=>String(v??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function table(cols,rows){$("result-table").innerHTML=cols.length?"<table><tr>"+cols.map(x=>"<th>"+esc(x)+"</th>").join("")+"</tr>"+rows.map(r=>"<tr>"+r.map(x=>"<td>"+esc(x)+"</td>").join("")+"</tr>").join("")+"</table>":""}
function mode(name){$("example-pane").classList.toggle("hidden",name!=="example");$("upload-pane").classList.toggle("hidden",name!=="upload");$("tab-example").classList.toggle("active",name==="example");$("tab-upload").classList.toggle("active",name==="upload")}
$("tab-example").onclick=()=>mode("example");$("tab-upload").onclick=()=>mode("upload");
$("files").onchange=()=>{$("file-list").textContent=[...$("files").files].map(f=>f.name+" ("+f.size+" bytes)").join(", ")};
async function loadStatus(){statusData=await(await fetch("/api/status")).json();$("example-question").value=statusData.question;$("example-files").innerHTML=statusData.files.map(f=>"<p><strong>"+esc(f.name)+"</strong></p><pre>"+esc(f.preview)+"</pre>").join("");table(statusData.cached_columns,statusData.cached_rows);$("run-example").disabled=!statusData.owner_configured;$("example-note").textContent=statusData.owner_configured?"Uses the deployment's "+statusData.model+".":"Owner key not configured; cached result only."}
async function poll(){if(!currentJob)return;const r=await fetch("/api/jobs/"+currentJob),j=await r.json();if(!r.ok){finishError(j.error);return}$("trace").textContent=j.trace.join("\n");if(j.status==="done"){clearInterval(pollTimer);pollTimer=null;table(j.columns,j.rows);$("result-note").textContent="Finished in "+j.steps+" steps and "+j.elapsed+" seconds.";$("download").href="/api/jobs/"+currentJob+"/download";$("download").classList.remove("hidden");enableRuns(true)}else if(j.status==="error"){clearInterval(pollTimer);pollTimer=null;finishError(j.error)}}
function enableRuns(on){$("run-example").disabled=!on||!statusData.owner_configured;$("run-upload").disabled=!on}
function finishError(message){$("result-note").textContent=message||"Run failed.";$("result-note").className="error";enableRuns(true)}
function begin(id){currentJob=id;$("result-note").textContent="Running…";$("result-note").className="muted";$("trace").textContent="Starting…";$("download").classList.add("hidden");enableRuns(false);pollTimer=setInterval(poll,1000);poll()}
$("run-example").onclick=async()=>{const r=await fetch("/api/example",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:$("example-question").value})}),d=await r.json();r.ok?begin(d.job_id):finishError(d.error)};
$("run-upload").onclick=async()=>{enableRuns(false);const files=await Promise.all([...$("files").files].map(async f=>({name:f.name,content:await f.text()})));const payload={files,pasted_csv:$("paste-csv").value,pasted_markdown:$("paste-md").value,question:$("upload-question").value,provider:$("provider").value,api_key:$("user-key").value};const r=await fetch("/api/upload",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});$("user-key").value="";const d=await r.json();r.ok?begin(d.job_id):finishError(d.error)};
loadStatus();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(
        self, status: int, body: bytes, content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_TOTAL_BYTES * 2 + 100_000:
            raise ValueError("Request is too large.")
        data = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/health":
                self._json(200, {"ok": True})
            elif path == "/api/status":
                self._json(200, demo_payload())
            elif re.fullmatch(r"/api/jobs/[^/]+", path):
                self._json(200, job_snapshot(path.rsplit("/", 1)[-1]))
            elif re.fullmatch(r"/api/jobs/[^/]+/download", path):
                job_id = path.split("/")[-2]
                self._send(
                    200, job_csv(job_id), "text/csv; charset=utf-8",
                    {"Content-Disposition": 'attachment; filename="prediction.csv"'},
                )
            else:
                self._json(404, {"error": "Not found."})
        except KeyError as exc:
            self._json(404, {"error": str(exc).strip("'")})
        except Exception:
            self._json(500, {"error": "The server could not complete that request."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            data = self._read_json()
            if path == "/api/example":
                self._json(202, start_example(data.get("question"), self.client_address[0]))
            elif path == "/api/upload":
                self._json(202, start_upload(data))
            else:
                self._json(404, {"error": "Not found."})
        except RuntimeError as exc:
            self._json(409, {"error": str(exc)})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            self._json(500, {"error": "The server could not start that run."})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}", flush=True)
    print("Public uploads: CSV + Markdown; generated Python is disabled.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
