"""Reproducible 'changed-facts' benchmark for the provenance-memory side.

The job under test (JTBD): a fact CHANGES over time; a recall must return the CURRENT
value, never a superseded one. This is the failure mode teams report with default vector
memory ("STALE"), and the one mem0's LLM write-path does not reliably fix.

This harness measures OUR side only -- it is deterministic and dependency-free, so anyone
can rerun it in seconds with no model, key, or vector DB:

    python bench/bench_changed_facts.py

For the mem0 comparison numbers (which need a local LLM + chromadb + ollama and are
therefore not run here), see MEM0_HEADTOHEAD.md and the pre-registered receipts
receipt_mem0_besteffort*.json. Summary of that pre-registered result, so the claim
travels with its caveats:

    on facts that change, STALE-recall rate (lower is better):
      mem0, naive (recall by similarity)        ~77.8%   [66, 86]
      mem0, best-effort (recall by stored max t) ~25.4%   [16, 37]   <- even used competently
      provenance-memory (this repo)                0.0%              <- deterministic
    caveat: a ~15-line "store t + recall by max(t)" hand-roll also scores 0%. There is no
    moat here; the point is a correct, tested, persistent, auditable version of that, and
    that mem0's DEFAULT does not give it to you.

Exit code is non-zero if our side is not exactly 0% stale (a regression guard).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mem0_shim import Memory

SEED = 20260706
N_FACTS = 200
CHANGES_PER_FACT = 5


def build_corpus(seed):
    """Each fact changes CHANGES_PER_FACT times. Returns (writes, truth).
    writes = list of (key, value, t); truth = {key: current_value_by_max_t}."""
    rng = random.Random(seed)
    writes = []
    truth = {}
    for i in range(N_FACTS):
        key = f"fact_{i}"
        # strictly increasing logical times, distinct per write
        for j in range(CHANGES_PER_FACT):
            t = j + 1
            value = f"v{i}_{j}_{rng.randint(1000, 9999)}"
            writes.append((key, value, t))
            truth[key] = value  # last written (max t) is ground truth
    return writes, truth


def measure(writes, truth, shuffle):
    m = Memory()
    order = list(writes)
    if shuffle:
        random.Random(SEED + 1).shuffle(order)  # arrival order scrambled; t is intact
    for key, value, t in order:
        m.add(key, value, user_id="bench", t=t)
    stale = 0
    for key, current in truth.items():
        if m.get(key, user_id="bench") != current:
            stale += 1
    return stale


def main():
    writes, truth = build_corpus(SEED)
    n_facts = len(truth)
    n_changes = len(writes)

    stale_inorder = measure(writes, truth, shuffle=False)
    stale_ooo = measure(writes, truth, shuffle=True)

    def rate(s):
        return 100.0 * s / n_facts

    print("provenance-memory :: changed-facts benchmark")
    print(f"  facts={n_facts}  writes={n_changes}  changes/fact={CHANGES_PER_FACT}  seed={SEED}")
    print(f"  in-order      stale-recall: {stale_inorder}/{n_facts} = {rate(stale_inorder):.1f}%")
    print(f"  out-of-order  stale-recall: {stale_ooo}/{n_facts} = {rate(stale_ooo):.1f}%")
    print("  (out-of-order = writes arrive shuffled, logical time t intact)")
    print()
    print("  reference (pre-registered, see MEM0_HEADTOHEAD.md -- NOT run here):")
    print("    mem0 naive       ~77.8% stale   |   mem0 best-effort ~25.4% stale")
    print("    caveat: a ~15-line max(t) hand-roll also = 0.0% (no moat; correctness + audit is the point)")

    ok = (stale_inorder == 0 and stale_ooo == 0)
    print()
    print("RESULT:", "PASS (0.0% stale, in-order AND out-of-order)" if ok else "FAIL -- regression")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
