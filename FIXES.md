# FIXES.md — prioritized work list (external code scan, 2026-07-07)

Findings below were verified empirically (tests run, bugs reproduced). Work them in order.
**Out of scope for now: PyPI publication** — the owner will handle the actual release at the end.
Preparing packaging metadata (`py.typed`, `[project.urls]`) is in scope; uploading is not.

## P0 — the failing test (blocks everything else)

**`provenance_memory.py:194-196` — the guard against too-deep values crashes inside its own
error message.** `_json_native` correctly detects a value nested past the recursion limit, but the
`ValueError` message formats `{fv!r}`, and `repr()` on a ~2000-level-deep list raises
`RecursionError` before the `ValueError` is ever constructed.

- Repro: `pytest test_provenance_memory.py::test_too_deep_value_fails_cleanly_not_recursionerror`
  → currently **1 failed, 42 passed** on Python 3.11 (recursion limit 1000).
- Fix: don't repr the full value in the message. Use the type name only, or a bounded repr
  (`reprlib.repr`, or wrap the f-string interpolation in `try/except RecursionError`).
- Acceptance: `pytest -q` → **43 passed**, which makes README.md:8 ("43 tests pass") true again.
  This repo's whole pitch is "verify, don't trust" — the first verify command must not fail.

## P1 — silent data-loss bug + dangling references

1. **`mem0_shim.py:118-121` — `reset()` silently drops persistence.** It recreates
   `ProvenanceStore(path=None)` while `self._path` is kept, so every write after `reset()`
   silently stops being persisted. Fix: recreate with the original path (truncating the file), or
   clear `self._path` too — pick one semantic and document it. Add a regression test: write →
   reset → write → new process re-opens the path → assert the post-reset fact is (or is not,
   per chosen semantic) recalled.

2. **Dangling doc references — files that do not exist in this repo.** Remove or replace each:
   - `provenance_memory.py:6` → `../REPLY_CARDS.md`, `../receipt_v0.json`
   - `mem0_shim.py:9` → `MEM0_HEADTOHEAD.md`
   - `bench/bench_changed_facts.py:13` and `:88` → `MEM0_HEADTOHEAD.md` (the real harness is now
     `bench/bench_vs_mem0.py` — point there)
   - `bench/bench_vs_mem0.py` module docstring → references `TAXONOMY.md`, an old filename
     `samb_mem0.py`, a Windows venv path, and says "Ollama Qwen" while the header comment says
     `llama3.1` — reconcile the docstring with the header (header is the recorded truth).

## P2 — hardening + CI

3. **CI.** Add `.github/workflows/ci.yml`: Python 3.9–3.13 matrix, `pytest -q`,
   `python bench/bench_changed_facts.py`, `python bench/fuzz_oracle.py` (all three are offline and
   fast; the mem0 harness stays excluded — it needs Ollama, as its header already says).

4. **`provenance_memory.py:203-210` — in-memory state mutates before `_append_disk()`.** An I/O
   error (disk full) leaves memory and disk diverged, while the docs imply atomicity. Either append
   to disk first and mutate memory only on success, or narrow the claim in README/docstrings to
   "validation is atomic; a mid-write I/O error can leave the last record memory-only".

5. **`mem0_shim.py:80` — `session=source or user_id` conflates two distinct semantics** (who said
   it vs. which conversation). Separate the mapping or document the collapse explicitly in the
   shim's docstring.

## P3 — packaging prep (no upload)

6. Add `py.typed` (the code is fully annotated — export that) and `[project.urls]` with the repo
   link in `pyproject.toml`. Do **not** publish; the owner does the release last.
