# Engineering FAQ

This guide prepares technical walkthroughs on Stet's agent loop, sandbox design, SQL CSV ingestion engine, and KDD evaluation framing.

---

## 1. 60-Second Elevator Pitch

> Stet started as a KDD Cup 2026 tabular data-agent system and evolved into a public-safe analytics web service (`serve.py`). Given a folder of CSV/Markdown/SQLite/Excel/PDF files and a natural-language question, it runs a three-phase **Planner → Executor → Verifier** loop with strict safety separation: local mode unlocks full Python/SQL/Excel/PDF/vision tools, while public web uploads run a strict allowlist that compiles uploaded CSVs into an isolated, query-only in-memory SQLite engine and wipes keys and temporary files immediately when the run ends.

---

## 2. Key Technical Questions & Concrete Answers

### Q1: Why use a three-phase Planner → Executor → Verifier loop instead of a single ReAct prompt?
- **Planning up front** forces the model to read the file tree and notice `knowledge.md` business definitions before wasting tool calls.
- **Verification at the end (`_verify`)** directly targets the competition rubric's extra-column penalty: LLMs naturally dump helper columns (`id`, `count`, intermediate subtotals) that lower `score = recall - lambda * (extra / predicted)`.

### Q2: How do you make public BYOK uploads safe without Docker-per-request overhead?
- **Disable arbitrary code execution (`for_public_upload`)**: `PUBLIC_TOOLS` strictly restricts the agent to `list_context`, `read_csv`, `read_doc`, `search_doc`, `query_tables`, and `answer` (blocking `run_python`, arbitrary DB paths, PDF, Excel, and vision).
- **In-memory SQLite (`tool_query_tables`)**: Rather than executing generated pandas code, CSVs are loaded with strict cell type inference into `:memory:` SQLite under `PRAGMA query_only = ON`, single-statement enforcement, and a `10,000`-step `set_progress_handler` timeout interrupt.
- **Ephemeral secret lifecycle**: The visitor key never touches `.env`, disk, cookies, logs (`redact_secret`), or responses — `ModelConfig` is held only by the worker thread and replaced with `"discarded"` inside the `finally` block as temp upload directories are deleted.

### Q3: What hard LLM quirks did you solve inside `agent.py` and `model.py`?
1. **Consecutive user turns breaking chat APIs (`BUG-1 / BUG-2 / BUG-3`)**: Injecting error hints, step-limit warnings, trimming history, or appending a forced-answer prompt can produce `user -> user` message sequences that trigger HTTP `400` errors. `_inject_user` and `_trim_history` enforce strict role alternation via `_ensure_alternating`.
2. **Reasoning tokens & noisy JSON (`BUG-6`)**: `_strip_thinking` removes `<think>`, `<|thinking|>`, and `<reasoning>` blocks, repairs trailing commas, and falls back to brace-depth scanning.
3. **NaN / numpy serialization crashes (`BUG-5`)**: `_normalise_rows` converts `NaN`, `Inf`, and `.item()` numpy scalars cleanly before emitting `Answer`.
4. **Provider rate limits (`ModelClient`)**: Per-key pacing (`_pace_lock`), `Retry-After` honoring, immediate failure on zero/daily/billing quota errors, and OpenAI project model probing (`gpt-5-mini -> gpt-4.1-mini -> gpt-4o-mini`).

### Q4: How does the column-multiset scorer work without matching column header names?
- Headers vary wildly (`net_rev` vs `total_net_revenue`). `score_answer` converts every column into a sorted tuple of rounded strings (`f"{round(float(val), 2):.2f}"` for numbers, `""` for missing values) and performs greedy 1-to-1 matching against gold columns. Recall rewards finding every required data series, while `extra / predicted` penalizes returning superfluous columns.
