#!/usr/bin/env python3
"""Oracle-checked fuzz harness -- the code behind the README's "4,000 oracle-checked fuzz cases".

Runs a deterministic random program of remember/recall/audit ops against ProvenanceStore and
mirrors every step in an INDEPENDENT ~30-line oracle implementing the documented contract:

    the value with the greatest t is current; on a t tie the FIRST arrival wins;
    older/equal late writes land in history but never change the current value.

After EVERY operation the store's recall() and audit().t are checked against the oracle
(2 checks/op). In persistent mode the store is periodically re-opened from disk and every key
re-verified -- so crash-safe replay is fuzzed too, not just the in-memory path.

Run:    python bench/fuzz_oracle.py             # 2,000 ops -> 4,000+ oracle checks, ~2s
        python bench/fuzz_oracle.py --ops 10000 --seed 7
Exit code is non-zero on the first divergence (regression guard, CI-friendly).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from provenance_memory import ProvenanceStore  # noqa: E402

KEYS = ["k%d" % i for i in range(9)]
SESSIONS = ["s1", "s2", "s3"]


def rand_value(rng: random.Random):
    pick = rng.randrange(6)
    if pick == 0:
        return rng.randrange(100)
    if pick == 1:
        return "v%d" % rng.randrange(50)
    if pick == 2:
        return None
    if pick == 3:
        return bool(rng.randrange(2))
    if pick == 4:
        return [rng.randrange(10) for _ in range(rng.randrange(3))]
    return {"a": rng.randrange(10)}


def type_aware_eq(a, b) -> bool:
    """The contract's equality: bool is not int, 1 is not 1.0, deep over containers."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(type_aware_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(type_aware_eq(a[k], b[k]) for k in a)
    return a == b


class Oracle:
    """Independent reference implementing the documented contract:
    greatest t wins; a t-tie keeps the first arrival; a strictly-newer write with an EQUAL
    (type-aware) value is corroboration -- current value AND its t stay unchanged."""

    def __init__(self) -> None:
        self.cur: dict = {}

    def write(self, key: str, value, t: int) -> None:
        if key not in self.cur:
            self.cur[key] = (t, value)
            return
        cur_t, cur_v = self.cur[key]
        if t > cur_t and not type_aware_eq(value, cur_v):
            self.cur[key] = (t, value)

    def recall(self, key: str):
        return self.cur[key][1] if key in self.cur else None

    def t(self, key: str):
        return self.cur[key][0] if key in self.cur else None


def run(ops: int, seed: int, persistent: bool) -> int:
    rng = random.Random(seed)
    path = None
    if persistent:
        path = os.path.join(tempfile.mkdtemp(prefix="provmem_fuzz_"), "mem.jsonl")
    store = ProvenanceStore(path=path)
    oracle = Oracle()
    clock = 0
    checks = 0
    for i in range(ops):
        key = rng.choice(KEYS)
        value = rand_value(rng)
        # mix auto-clock, explicit-future, duplicate and stale timestamps
        mode = rng.randrange(10)
        if mode < 6:                      # auto (monotone) -- the common path
            clock += 1
            t = clock
            store.remember(key, value, session=rng.choice(SESSIONS), t=t)
        else:                             # explicit: late/duplicate/future
            t = rng.randrange(1, clock + 5) if clock else 1
            clock = max(clock, t)
            store.remember(key, value, session=rng.choice(SESSIONS), t=t)
        oracle.write(key, value, t)

        probe = rng.choice(KEYS)
        got, want = store.recall(probe), oracle.recall(probe)
        if got != want:
            print(f"DIVERGENCE at op {i}: recall({probe}) = {got!r}, oracle says {want!r}")
            return 1
        checks += 1
        rec = store.audit(probe)
        got_t = rec.t if rec else None
        if got_t != oracle.t(probe):
            print(f"DIVERGENCE at op {i}: audit({probe}).t = {got_t}, oracle says {oracle.t(probe)}")
            return 1
        checks += 1

        if persistent and i % 250 == 249:  # crash-safe replay: reopen from disk, verify everything
            store = ProvenanceStore(path=path)
            for k in KEYS:
                if store.recall(k) != oracle.recall(k):
                    print(f"DIVERGENCE after reload at op {i}: key {k}")
                    return 1
                checks += 1
    print(f"OK: {ops} ops, {checks} oracle checks, 0 divergences "
          f"({'persistent+replay' if persistent else 'in-memory'}, seed={seed})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=1000, help="ops PER MODE (2 modes run)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rc = run(args.ops, args.seed, persistent=False)
    if rc:
        return rc
    return run(args.ops, args.seed + 1, persistent=True)


if __name__ == "__main__":
    raise SystemExit(main())
