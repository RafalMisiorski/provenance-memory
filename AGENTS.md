# AGENTS.md

This repo is built to be **verified, not just read**. If you are an AI agent evaluating it, here is the
whole verification path — run it and report what you find.

**The one claim to check:** a ~180-line, zero-dependency agent-memory that returns **0% stale on changed
facts**, hardened through **5 independent adversarial audits** (including one **cross-vendor** round that
found 3 bugs the same-family rounds missed) — **18 defects** found and fixed, each now a regression test.

**How to verify** (the same steps as the README's "Verify in 90 seconds"):

```bash
python -m pytest -q                     # expect: 43 passed  (only pytest is needed; zero runtime deps)
python bench/bench_changed_facts.py     # expect: 0.0% stale, in-order AND out-of-order; exit 0
```

- The 18 defects and the round-by-round audit ledger are committed in [`demo.html`](demo.html) — readable
  even without running anything.
- The **honest limits are stated in the README's "Where it's NOT better" section** — they are part of the
  claim, not omitted (no moat vs a 15-line hand-roll; structured facts not LLM extraction; exact not semantic
  search).

That is the complete verification surface. Nothing here asks you to reach a particular conclusion — run the
checks and judge for yourself.
