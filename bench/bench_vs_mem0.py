#!/usr/bin/env python3
# PROVENANCE OF THE README's mem0 NUMBERS -- this is the exact harness behind the table.
#
# Recorded run (2026-07-05): mem0ai==2.0.11, fully-local backend = Ollama llama3.1:latest
# (LLM write-path) + nomic-embed-text (embeddings). 4 seeds x ~20 changed-fact items,
# pooled 63 scored items:
#     mem0 naive (as-shipped top-1 recall):       77.8% stale   Wilson95 [66.1, 86.3]  (49/63)
#     mem0 best-effort (recall newest by stored t): 25.4% stale  Wilson95 [16.3, 37.3]  (16/63)
#     this repo (newest-wins by logical t):          0.0% stale  (structural; bench_changed_facts.py)
#
# HONEST CAVEATS (also in the README): local backend, NOT the hosted mem0 product; N=63 pooled;
# the STALE slice only (mem0 wins elsewhere, e.g. semantic/fuzzy retrieval); scoring is
# deterministic token-match, no LLM judge. This script is NOT run in CI (needs Ollama + an
# isolated venv because mem0ai pulls heavy deps):
#     python -m venv venv_mem0 && venv_mem0/Scripts/pip install mem0ai==2.0.11
#     venv_mem0/Scripts/python bench/bench_vs_mem0.py --n 30 --model llama3.1:latest
#
"""SAMB v0.1 -- measure mem0 (a library that TARGETS the memory problem) on the STALE mode, on a
sealed clean-value item set, with a fully-LOCAL backend (Ollama llama3.1 + nomic-embed-text; no API key).

Why this file exists: chroma/BM25 are generic retrieval -- they don't try to solve supersession.
mem0's whole pitch IS UPDATE/DELETE/change-detection ("If a memory API handled UPDATE and DELETE
automatically..." -- r/LangChain). To make SAMB a credible referee (not a strawman ad), the incumbent
that targets the problem must be in the table -- including if it beats us.

Run UNDER THE ISOLATED VENV (mem0ai pulls heavy deps we don't want in the main interp; see the header
comment above for the exact recorded-run commands):
    venv_mem0/Scripts/python bench/bench_vs_mem0.py --n 30 --model llama3.1:latest

Scoring stays deterministic + no-LLM-judge: mem0 returns NL memories, so we tokenize the top memory
and check whether it carries the CURRENT value (pass), a SUPERSEDED value (stale-fail), or neither
(no-recall). Values are single tokens by construction, so the check is exact, not substring-fuzzy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

HERE = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434"

# Clean, single-token values so the deterministic token check is exact (no hyphens/urls/collisions).
CLEAN_FACT_POOL = [
    ("refund_window_days", ["30", "14", "60", "7"]),
    ("ceo_name", ["Alice", "Bob", "Carla", "Dana"]),
    ("max_upload_mb", ["25", "100", "250"]),
    ("sla_hours", ["48", "24", "8", "72"]),
    ("pricing_tier", ["free", "pro", "team", "enterprise"]),
    ("db_engine", ["sqlite", "postgres", "duckdb", "mysql"]),
    ("capital_city", ["Paris", "Berlin", "Rome", "Madrid"]),
    ("default_model", ["gpt4", "opus", "sonnet", "gemini"]),
]
DISTRACTORS = [
    ("office_city", ["Krakow", "Gdansk", "Wroclaw"]),
    ("mascot", ["otter", "gecko", "falcon"]),
    ("theme_color", ["teal", "amber", "indigo"]),
]
_TOK = re.compile(r"[a-z0-9]+")


def _tok(s):
    return set(_TOK.findall(s.lower()))


def make_clean_stale_items(n, seed):
    rng = random.Random(seed)
    items = []
    for i in range(n):
        key, values = rng.choice(CLEAN_FACT_POOL)
        n_updates = rng.randint(0, min(3, len(values) - 1))
        chain = values[: n_updates + 1]
        events, t = [], 0
        events.append({"key": key, "value": chain[0], "t": t}); t += 1
        for v in chain[1:]:
            for _ in range(rng.randint(0, 1)):
                dk, dv = rng.choice(DISTRACTORS)
                events.append({"key": dk, "value": rng.choice(dv), "t": t}); t += 1
            events.append({"key": key, "value": v, "t": t}); t += 1
        items.append({"id": f"m-stale-{i:03d}", "key": key, "events": events,
                      "expect": chain[-1], "superseded": chain[:-1], "updated": n_updates > 0})
    return items


def _sentence(key, value):
    # natural phrasing so mem0's extractor stores a real fact (not a raw k=v blob)
    return f"The {key.replace('_', ' ')} is {value}."


class Mem0Memory:
    name = "mem0_local"

    def __init__(self, model, embed="nomic-embed-text", openai_base=None):
        from mem0 import Memory
        import tempfile
        # FRESH isolated chroma path per instance -> zero cross-run/cross-seed persistence.
        # (The contamination bug: a shared persistent store let a reused user_id accumulate stale
        #  memories across runs, inflating the stale-rate. A fresh path per instance kills it at root.)
        self._chroma_path = tempfile.mkdtemp(prefix="samb_mem0_")
        if openai_base:  # route the LLM through the Codex shim (subscription, no API key)
            llm = {"provider": "openai", "config": {
                "model": model, "temperature": 0.0,
                "openai_base_url": openai_base, "api_key": "local-shim"}}
        else:
            llm = {"provider": "ollama", "config": {
                "model": model, "temperature": 0.0, "ollama_base_url": OLLAMA}}
        cfg = {
            "llm": llm,
            "embedder": {"provider": "ollama", "config": {  # embeddings stay local either way
                "model": embed, "ollama_base_url": OLLAMA}},
            "vector_store": {"provider": "chroma", "config": {
                "collection_name": "samb_mem0", "path": self._chroma_path}},
        }
        self._m = Memory.from_config(cfg)
        self._uid = "u0"

    def reset(self, scope=None):
        # fresh isolated store per instance + unique user_id per item -> no accumulation, no delete needed
        self._uid = str(scope or "u0")

    def ingest(self, ev):
        self._m.add(_sentence(ev["key"], ev["value"]), user_id=self._uid,
                    metadata={"key": ev["key"], "t": ev["t"]})

    def _top_text(self, key):
        # mem0 2.x: user_id must go via filters=, not as a top-level search() kwarg
        r = self._m.search(key.replace("_", " "), filters={"user_id": self._uid}, limit=3)
        results = r.get("results", r) if isinstance(r, dict) else r
        if not results:
            return None, None
        top = results[0]
        return (top.get("memory") or top.get("text") or ""), top.get("id")

    def query(self, key, session=None):
        text, _ = self._top_text(key)
        return text

    def audit(self, key, session=None):
        _, mid = self._top_text(key)
        return {"id": mid} if mid else None


def p(s):
    print(str(s).encode("ascii", "replace").decode("ascii"))


def score_stale(sut, items):
    """PASS = top memory carries current value & no superseded value. stale = a superseded value
    surfaces (current absent). no_recall = neither. All deterministic token checks."""
    passed = stale = no_recall = 0
    up_stale = up_n = 0
    for it in items:
        sut.reset(it["id"])
        for ev in it["events"]:
            sut.ingest(ev)
        text = sut.query(it["key"]) or ""
        toks = _tok(text)
        has_cur = it["expect"].lower() in toks
        has_old = any(s.lower() in toks for s in it["superseded"])
        if has_cur and not has_old:
            passed += 1
            verdict = "pass"
        elif has_old and not has_cur:
            stale += 1
            verdict = "stale"
        elif has_cur and has_old:
            stale += 1  # both present -> ambiguous recall of a superseded fact = still a stale hazard
            verdict = "stale"
        else:
            no_recall += 1
            verdict = "no_recall"
        if it["updated"]:
            up_n += 1
            if verdict != "pass":
                up_stale += 1
    n = len(items)
    return {"system": sut.name, "n_items": n,
            "stale_fail_rate": round((stale + no_recall) / n, 4),
            "stale_only_rate": round(stale / n, 4),
            "no_recall_rate": round(no_recall / n, 4),
            "pass_rate": round(passed / n, 4),
            "stale_fail_rate_on_updated": round(up_stale / up_n, 4) if up_n else 0.0,
            "n_updated": up_n, "up_fail": up_stale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--model", default="llama3.1:latest")  # matches the recorded run in the header
    ap.add_argument("--openai-base", default=None, help="route LLM via an OpenAI-compatible base (Codex shim)")
    ap.add_argument("--smoke", action="store_true", help="1-item wiring smoke test then exit")
    args = ap.parse_args()

    if args.smoke:
        p(f"smoke: mem0 + {'shim:'+args.openai_base if args.openai_base else 'ollama'}({args.model}) ...")
        sut = Mem0Memory(args.model, openai_base=args.openai_base)
        sut.reset("smoke")
        for v in ["30", "14", "7"]:
            sut.ingest({"key": "refund_window_days", "value": v, "t": 0})
        text = sut.query("refund_window_days")
        p(f"  top memory: {text!r}")
        p(f"  audit: {sut.audit('refund_window_days')}")
        return 0

    items = make_clean_stale_items(args.n, args.seed)
    sha = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    p(f"SAMB v0.1 / mem0 STALE  n={args.n} seed={args.seed} model={args.model} sha={sha[:12]}...")
    t0 = time.time()
    res = score_stale(Mem0Memory(args.model, openai_base=args.openai_base), items)
    res["seconds"] = round(time.time() - t0, 1)
    res["model"] = args.model
    out = {"benchmark": "SAMB", "version": "v0.1", "mode": "STALE", "seed": args.seed,
           "items_sha256": sha, "backend": f"ollama:{args.model}+nomic-embed-text",
           "scoring": "deterministic token check on mem0's returned memory (no LLM judge)",
           "result": res}
    (HERE / "receipt_mem0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    p("-" * 64)
    p(f"  mem0_local  STALE-fail: {res['stale_fail_rate']*100:5.1f}%  "
      f"(on updated: {res['stale_fail_rate_on_updated']*100:5.1f}%)  "
      f"[stale {res['stale_only_rate']*100:.0f}% / no-recall {res['no_recall_rate']*100:.0f}%]  "
      f"{res['seconds']}s")
    p(f"written: receipt_mem0.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
