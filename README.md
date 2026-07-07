# provenance-memory

**Agent memory that doesn't go stale, doesn't lose cross-session facts, and can always tell you *why* it
recalled something.** ~180 lines, zero dependencies, MIT.

> **Hardened, not just written.** This code was handed to **5 independent adversarial audits** — 4
> same-family models + **1 cross-vendor** (a different vendor's model found 3 bugs the first four missed).
> **18 defects** found, all fixed, each now a named regression test. **43 tests pass**, backed by **4,000
> oracle-checked fuzz cases**. The round-by-round ledger is in [`demo.html`](demo.html) — and every number
> above is reproducible with the commands in the next section.

## Verify in 90 seconds (zero setup)

No model, no API key, no vector DB. `pytest` is the only dependency, and only for the tests.

```bash
python -m pytest -q                     # 43 tests — the correctness properties as passing assertions
python bench/bench_changed_facts.py     # prints 0.0% stale on 200 changed facts (in-order AND out-of-order)
python bench/fuzz_oracle.py             # 4,000+ oracle checks vs an independent reference — exits red on divergence
python provenance_memory.py             # live demo: recall / audit / history / determinism
```

Every number in the credential above is checkable here: `pytest` runs the 43 tests (each defect the audits
found is one of them); the benchmark prints `0.0%` stale; the fuzz harness replays a deterministic random
program against a ~30-line independent oracle (in-memory AND persistent+reload); the full ledger with all
18 defects is in [`demo.html`](demo.html).

## What it does (30 seconds)

```python
from provenance_memory import ProvenanceStore

m = ProvenanceStore()
m.remember("refund_window_days", "30", session="s1")
m.remember("refund_window_days", "14", session="s2")   # supersedes 30
m.remember("refund_window_days", "7",  session="s5")   # supersedes 14

m.recall("refund_window_days")     # -> "7"   current value, from any session
m.audit("refund_window_days")      # -> Record(value="7", t=3, session="s5", revision=2, superseded="14")
m.history("refund_window_days")    # -> the whole supersession chain
```

Most agent memory is an append-only vector bag: a changed fact is *added*, never superseded, and retrieval
returns "whatever cosine says is nearest" — so it hands back a stale value. This returns the current value by
newest logical timestamp, is safe against out-of-order writes, and records the provenance of every recall.

## vs mem0 (the library that actually targets this)

On facts that **change**, STALE-recall rate from **our recorded run** (2026-07-05, pre-registered; the exact
harness ships in this repo as [`bench/bench_vs_mem0.py`](bench/bench_vs_mem0.py)) — lower is better:

| system | stale on changed facts | 95% CI |
|---|---:|---:|
| mem0 2.0.11, naive (recall by similarity) | 77.8% (49/63) | [66.1, 86.3] |
| mem0 2.0.11, best-effort (recall by the timestamp you stored) | 25.4% (16/63) | [16.3, 37.3] |
| **provenance-memory** | **0.0%** | structural |

**Method + honest caveats** (in the harness header too): `mem0ai==2.0.11` pinned, fully-LOCAL backend
(Ollama `llama3.1` write-path + `nomic-embed-text`) — **not the hosted mem0 product**; N=63 pooled over
4 seeds; the STALE slice only (mem0 wins elsewhere, e.g. semantic/fuzzy retrieval); deterministic
token-match scoring, no LLM judge. The mem0 rows are **not run in CI** (they need Ollama + an isolated
venv — setup commands in the script header); our 0.0% row IS re-run by `bench_changed_facts.py` on every
verify. Root cause of the gap: mem0's LLM write-path doesn't reliably preserve the metadata you attach, so
"newest-by-timestamp" degenerates back toward naive. Full ledger: [`demo.html`](demo.html).

## Use it in place of mem0 (drop-in *shaped*)

```python
from mem0_shim import Memory            # instead of: from mem0 import Memory

m = Memory()
m.add("refund_window_days", "30", user_id="acme")
m.add("refund_window_days", "7",  user_id="acme")   # supersedes 30
m.get("refund_window_days", user_id="acme")         # -> "7"  (never a stale copy)
m.audit("refund_window_days", user_id="acme")       # -> why "7" is current (mem0 has no equivalent)
```

## Where it's NOT better (read this — it's part of the claim)

- **No moat.** A ~15-line "store `t`, recall by `max(t)`" hand-roll also gets 0% stale. The value here is a
  *correct, tested, persistent, auditable* version of that — and the demonstration that mem0's default does
  **not** give it to you.
- **Structured facts, not extraction.** You pass `(key, value)`; mem0 runs an LLM to *extract* facts from raw
  conversation. If you need extraction from free text, keep mem0 (or extract first, then feed facts here).
- **Exact recall, not semantic.** `search()` is substring key match. For fuzzy semantic retrieval, mem0 wins.

## How we know it's correct — the audit ledger

The first version passed its own 10 unit tests and was still wrong. So it was handed to independent
adversaries — separate models, then a **different vendor's model** — whose only job was to break it. The
cross-vendor round found three bugs four same-family rounds missed. 18 defects total (including ones the
fixes themselves introduced), each now a named regression test. Round-by-round ledger: [`demo.html`](demo.html).

## API

| call | does |
|---|---|
| `remember(key, value, *, session="default", source=None, t=None)` | store/update; a different value **supersedes** the old; an identical re-write is idempotent |
| `recall(key) -> value \| None` | current value (greatest `t`), cross-session, deterministic, O(1) |
| `audit(key) -> Record \| None` | provenance for the current value: `value, t, session, source, revision, superseded` |
| `history(key) -> [Record]` | full supersession chain, arrival order |
| `keys() -> [str]` | all known keys |

The mem0-shaped adapter `mem0_shim.Memory` (`add / get / get_all / search / audit / history / reset`,
namespaced by `user_id`) wraps this — see *Use it in place of mem0* above.

MIT licensed. Zero runtime dependencies (`pytest` only for the tests).
