# Frontend Changes — Dark / Light Theme Toggle

Added a theme toggle that lets users switch between the existing dark theme and a
new light theme. The choice persists across reloads via `localStorage`.

## Files changed

### `frontend/index.html`
- Added a fixed-position **theme toggle button** (`#themeToggle`) just inside
  `<body>`, before `.container`. It contains two inline SVG icons (a sun and a
  moon) that cross-fade depending on the active theme.
- The button is a native `<button>` with `aria-label`/`title` for accessibility,
  so it is keyboard-focusable and activates with Enter/Space out of the box.
- Bumped the CSS/JS cache-busting query params from `?v=10` to `?v=11` so the
  edits are picked up on reload.

### `frontend/style.css`
- Renamed the `:root` block comment to **"Dark theme (default)"** and added a
  `--code-bg` variable so code-block backgrounds adapt per theme.
- Added a **`[data-theme="light"]`** block overriding all color variables with a
  light palette: white/`#f1f5f9` surfaces, dark `#0f172a` text, adjusted
  borders/shadows, and a light welcome background — all chosen for good contrast.
- Added a **smooth transition** rule (`background-color`, `color`, `border-color`,
  `box-shadow`, 0.3s) on the major surfaces so switching themes animates.
- Added **`.theme-toggle`** styles: a 44px circular button pinned to the
  top-right (`position: fixed`), with hover lift, `:active` press, and
  `:focus-visible` focus ring.
- Added **icon crossfade** styles: the sun/moon icons rotate and fade in/out
  based on the `data-theme` attribute.
- Switched the two hardcoded `rgba(0,0,0,0.2)` code backgrounds to the new
  `var(--code-bg)`.

### `frontend/script.js`
- Added `themeToggle` to the tracked DOM elements.
- Added an early `initTheme()` call (before `DOMContentLoaded`) that reads the
  saved theme from `localStorage` (defaulting to `dark`) and applies it
  immediately to avoid a flash of the wrong theme.
- Added theme helpers:
  - `initTheme()` — loads and applies the stored preference.
  - `applyTheme(theme)` — sets/removes the `data-theme="light"` attribute on
    `<html>`.
  - `toggleTheme()` — flips the theme and persists it to `localStorage`.
- Wired a `click` listener on the toggle button. Keyboard activation is handled
  natively by the `<button>` element (Enter/Space).

## Behavior
- **Default:** dark theme (unchanged for existing users).
- **Toggle:** click (or focus + Enter/Space) the top-right button to switch.
- **Icon:** moon shows in dark mode, sun shows in light mode, with a smooth
  rotate/fade transition.
- **Persistence:** the selected theme is saved and restored on the next visit.
- **Implementation:** purely via CSS custom properties switched by a
  `data-theme` attribute on the `<html>` element — no per-element overrides
  needed, preserving the existing visual hierarchy.

---

# Testing Framework Enhancements

This change adds API-level testing infrastructure for the RAG system, on top of
the existing component unit tests. The frontend talks to the backend exclusively
through the HTTP API (`POST /api/query`, `GET /api/courses`, `POST
/api/session/clear`, and `GET /`), so these tests lock down the request/response
contract the frontend depends on.

## What changed

### 1. `pyproject.toml` — pytest configuration + dev dependencies

- Added a `[dependency-groups] dev` group with `pytest` and `httpx` (the latter
  is required by FastAPI's `TestClient`).
- Added `[tool.pytest.ini_options]`:
  - `testpaths = ["backend/tests"]` so `pytest` can be run from the repo root.
  - `python_files`/`python_classes`/`python_functions` patterns that collect the
    new pytest-style classes **and** the existing `unittest.TestCase` classes.
  - `addopts = "-ra --tb=short --strict-markers"` for concise, useful output.
  - `norecursedirs` to skip `chroma_db`, `__pycache__`, `.venv`.
  - Because collection is restricted to `test_*.py`, the ad-hoc
    `_diagnose_*.py` / `_inspect_*.py` / `_verify_*.py` live-environment probe
    scripts are no longer picked up as tests.

### 2. `backend/tests/conftest.py` — shared fixtures (new file)

The real `backend/app.py` can't be imported cleanly under test because:
- it mounts `StaticFiles(directory="../frontend")` at import time, which raises
  unless the cwd is `backend/`; and
- it constructs a real `RAGSystem` (ChromaDB + embedding model + Anthropic
  client) at import time.

Per the task guidance, `conftest.py` therefore **rebuilds the same API surface
inline** — identical Pydantic models, routes, and error handling — wired to a
mocked `RAGSystem`, with the static mount omitted and `/` stubbed. Fixtures:

- `sample_sources`, `sample_query_result`, `sample_analytics` — canned test data
  shaped exactly like the real return values.
- `mock_rag_system` — a `MagicMock` with sensible default returns for `query`,
  `get_course_analytics`, and `session_manager.create_session/clear_session`.
- `test_app` — the inline FastAPI app bound to the mock.
- `client` — a `TestClient` for the inline app.

### 3. `backend/tests/test_api_endpoints.py` — API tests (new file)

16 tests covering:
- **`POST /api/query`**: explicit vs. auto-created session id, source shape,
  empty-sources case, `422` on missing/malformed body, and `500` when the RAG
  system raises.
- **`GET /api/courses`**: stats payload, response shape, empty catalog, and
  `500` on failure.
- **`POST /api/session/clear`**: success, `422` on missing `session_id`, and
  `500` on failure.
- **`GET /`**: root reachability without the frontend directory.

### 4. `backend/tests/test_rag_system.py` — fixture fix

`FakeConfig` was missing `MAX_TOOL_ROUNDS`, which `RAGSystem.__init__` now reads
(it exists in `config.py`). This had been breaking 7 existing tests at collection
time; added the attribute so the whole suite is green again.

## Result

```
41 passed
```

Run the suite from the repo root with:

```powershell
uv run pytest            # once `uv sync --group dev` has installed dev deps
# or directly against the venv:
.venv\Scripts\python.exe -m pytest
```

---

# Code Quality Tooling (Prettier)

Added automated code quality tooling for the frontend and applied consistent
formatting across the entire frontend codebase. **Prettier** is the de-facto
formatter for JS/CSS/HTML — the front-end equivalent of Python's `black`.

## What was added

### 1. Prettier formatter

- **`package.json`** (new, repo root) — declares `prettier` as a dev dependency and
  exposes npm scripts:
  - `npm run format` — format all frontend files in place.
  - `npm run format:check` — verify formatting without writing (CI-friendly,
    non-zero exit on violations).
  - `npm run quality` — alias for `format:check`, the single quality-gate entry point.
- **`.prettierrc.json`** (new) — shared config so every editor and CI produce
  identical output: 100-char print width, 2-space indent, single quotes, ES5
  trailing commas, semicolons, `lf` line endings.
- **`.prettierignore`** (new) — keeps Prettier scoped to the frontend by excluding
  the Python backend, `.venv/`, `node_modules/`, ChromaDB data, `docs/`, and
  Markdown files.

### 2. Developer quality-check scripts

Cross-platform wrappers in a new **`scripts/`** directory:

- **`scripts/format.ps1`** / **`scripts/format.sh`** — auto-format the frontend in
  place.
- **`scripts/quality-check.ps1`** / **`scripts/quality-check.sh`** — read-only
  formatting check that fails on any unformatted file. Suitable for CI or a
  pre-commit hook. PowerShell scripts use colored output and propagate the exit
  code; Bash scripts use `set -euo pipefail`.

### 3. `.gitignore` update

Added `node_modules/` so the installed Prettier dependency is not committed.

## Formatting applied to existing code

Ran `npx prettier --write` across all frontend files to establish a consistent
baseline. These are formatting-only changes (indentation, quote style, line
wrapping, spacing) — **no behavior was modified**:

| File                  | Result      |
| --------------------- | ----------- |
| `frontend/index.html` | reformatted |
| `frontend/script.js`  | reformatted |
| `frontend/style.css`  | reformatted |

All files now pass `npm run format:check`.

## How to use

```bash
# One-time: install the tooling (Prettier)
npm install

# Auto-fix formatting
npm run format            # or: ./scripts/format.ps1   (Windows)
                          #     ./scripts/format.sh     (Linux/macOS)

# Check formatting without changing files (CI / pre-commit)
npm run format:check      # or: ./scripts/quality-check.ps1
                          #     ./scripts/quality-check.sh
```

## Notes

- If `npm install` fails with `UNABLE_TO_VERIFY_LEAF_SIGNATURE` behind a corporate
  proxy, run it with the system certificate store:
  `$env:NODE_OPTIONS="--use-system-ca"; npm install` (PowerShell) or
  `NODE_OPTIONS=--use-system-ca npm install` (Bash).
