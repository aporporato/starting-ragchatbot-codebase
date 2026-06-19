# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (Python >=3.13). The ChromaDB store at `backend/chroma_db/` is gitignored and populated on first startup.

Install dependencies (same on every platform):

```
uv sync
```

Requires `ANTHROPIC_API_KEY` in `.env` at repo root (see `.env.example`).

### Start the server — Windows (PowerShell)

`run.sh` is a Bash script and will not run in PowerShell. Use the uvicorn command directly:

```powershell
uv run uvicorn --app-dir backend app:app --reload --port 8000
```

### Start the server — Linux / macOS (or Windows + Git Bash)

```bash
./run.sh
```

which is equivalent to:

```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

`--app-dir backend` (or `cd backend`) is required so the relative `../docs` path in `app.py`'s startup hook resolves correctly (see Startup ingestion below).

App serves at `http://localhost:8000` (UI) and `/docs` (Swagger). There is no test suite, linter config, or build step.

## Architecture

### Tool-based retrieval (not classic RAG)

`RAGSystem.query` (`backend/rag_system.py`) does **not** retrieve chunks and stuff them into the prompt. Instead it gives Claude a `search_course_content` tool definition and lets the model decide whether to call it. The flow inside `AIGenerator.generate_response` (`backend/ai_generator.py`):

1. First `messages.create` call includes `tools=[...]` and `tool_choice={"type":"auto"}`.
2. If `stop_reason == "tool_use"`, `_handle_tool_execution` runs the tool, appends the result as a `tool_result` message, and makes a **second** `messages.create` call **without** tools to get the final answer.
3. Otherwise the first response is returned directly.

The `SYSTEM_PROMPT` in `ai_generator.py` enforces "one search per query maximum" — there is no multi-step tool loop. Changing this prompt is the lever for retrieval behaviour, not the orchestration code.

### Sources are passed via a side-channel, not the LLM

The tool's `_format_results` stashes human-readable sources on `CourseSearchTool.last_sources` (`backend/search_tools.py`). After the answer is generated, `RAGSystem.query` reads them via `ToolManager.get_last_sources()` and then calls `reset_sources()`. This is why sources appear in the API response separately from `answer`, and why a stale-source bug would manifest if `reset_sources()` were skipped.

### Two ChromaDB collections, different purposes

`VectorStore` (`backend/vector_store.py`) maintains:

- **`course_catalog`** — one document per course, embedded text is the course title. Used by `_resolve_course_name` to fuzzy-match a user-supplied `course_name` (e.g. "MCP") to a canonical title before content search. Lessons are stored as a JSON-stringified metadata field (`lessons_json`) because ChromaDB metadata doesn't accept nested objects.
- **`course_content`** — the chunks. Filtered by `course_title` and/or `lesson_number` once the catalog has resolved the name.

When searching, the resolved title is used as an exact `where` filter on `course_content`. Adding metadata fields means writing them in `add_course_metadata` **and** parsing them back in `get_all_courses_metadata` / `get_lesson_link`.

### Document format is load-bearing

`DocumentProcessor.process_course_document` (`backend/document_processor.py`) parses a strict structure:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<body...>
Lesson 1: ...
```

Header lines are matched by regex prefix; lesson markers are `^Lesson\s+(\d+):\s*(.+)$`. If no `Lesson N:` markers exist the entire body becomes one unnamed chunk block (`lesson_number=None`). Course title is the **primary key** in both Chroma collections — duplicate titles are skipped in `add_course_folder`.

Chunking quirk: for non-final lessons, only the first chunk is prefixed with `"Lesson N content: "`. For the final lesson, **every** chunk is prefixed with `"Course <title> Lesson N content: "`. This inconsistency lives in `document_processor.py` around the two `chunk_text` call sites and affects embedding quality of later chunks of non-final lessons.

### Startup ingestion

`backend/app.py`'s `startup_event` calls `rag_system.add_course_folder("../docs")`. That path is **relative to the backend working directory**, which is why `run.sh` does `cd backend && uvicorn ...` and why `--app-dir backend` is needed when invoking uvicorn from the repo root. Existing courses are detected by title and skipped, so re-ingestion is idempotent unless the title changes.

### Session state is in-memory

`SessionManager` (`backend/session_manager.py`) stores messages in a plain dict keyed by `session_<n>`. Restarting the server drops all history. History is capped at `MAX_HISTORY * 2 = 4` messages (`config.py`) and injected into the system prompt as `"Previous conversation:\n<formatted>"` — not as separate `messages` entries.

### Config is a frozen dataclass

`backend/config.py` exposes a single `config` instance. Model, chunk size, embedding model, max results, and Chroma path live here — there is no per-request override.

## Frontend

Static `frontend/index.html` + `script.js` + `style.css`, mounted at `/` by `app.py` via `StaticFiles`. `DevStaticFiles` injects `Cache-Control: no-cache` headers so reloads pick up edits without a hard refresh. The UI is two endpoints' worth of behaviour: `POST /api/query` and `GET /api/courses`. Markdown rendering on the assistant side uses `marked` loaded via the HTML.
