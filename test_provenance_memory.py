"""Regression tests: the three SAMB failure modes + every defect two independent adversarial audits
broke earlier versions on, encoded as properties this store MUST hold.

"check my repo -- I don't have this problem", executable. Run:
    python -m pytest test_provenance_memory.py -q
    python test_provenance_memory.py        # no pytest? runs the same asserts standalone
"""
import math
import os
import tempfile

import pytest

from provenance_memory import ProvenanceStore


# --- the three SAMB failure modes -----------------------------------------------------------------

def test_stale_returns_current_not_superseded():
    m = ProvenanceStore()
    for v in ["30", "14", "60", "7"]:
        m.remember("refund_window_days", v)
    assert m.recall("refund_window_days") == "7"
    assert m.audit("refund_window_days").superseded == "60"
    assert [r.value for r in m.history("refund_window_days")] == ["30", "14", "60", "7"]


def test_xsession_recall_is_global():
    m = ProvenanceStore()
    m.remember("language", "pl", session="s0")
    for s in range(1, 6):
        m.remember(f"noise_{s}", "x", session=f"s{s}")
    assert m.recall("language") == "pl"


def test_nondet_deterministic_and_auditable():
    m = ProvenanceStore()
    m.remember("api_base_url", "v1.api.co")
    m.remember("api_base_url", "v2.api.co")
    assert {m.recall("api_base_url") for _ in range(200)} == {"v2.api.co"}
    assert m.audit("api_base_url") is not None


def test_idempotent_rewrite_no_spurious_revision():
    m = ProvenanceStore()
    m.remember("plan", "pro")
    m.remember("plan", "pro")
    assert m.audit("plan").revision == 0
    m.remember("plan", "enterprise")
    assert m.audit("plan").revision == 1


def test_unknown_key_is_none():
    assert ProvenanceStore().recall("never_stored") is None


# --- out-of-order correctness (audit #1: v0 superseded by ARRIVAL order) ---------------------------

def test_out_of_order_late_old_write_does_not_win():
    m = ProvenanceStore()
    m.remember("k", "NEW", t=100)
    m.remember("k", "OLD", t=5)
    assert m.recall("k") == "NEW"


def test_out_of_order_interleaved():
    m = ProvenanceStore()
    m.remember("refund", "30", t=1)
    m.remember("refund", "7", t=3)
    m.remember("refund", "14", t=2)
    assert m.recall("refund") == "7"


def test_out_of_order_audit_does_not_lie():
    """audit #2 BUG5: a late OLD write that was never current must NOT be reported as 'superseded',
    and must NOT inflate revision."""
    m = ProvenanceStore()
    m.remember("k", "NEW", t=100)
    m.remember("k", "OLD", t=5)
    a = m.audit("k")
    assert a.superseded is None
    assert a.revision == 0


def test_equal_t_does_not_dethrone():
    """audit #2 BUG8: only a STRICTLY newer t supersedes; an equal-t write does not."""
    m = ProvenanceStore()
    m.remember("k", "FIRST", t=7)
    m.remember("k", "SECOND", t=7)
    assert m.recall("k") == "FIRST"


def test_value_reverting_is_a_real_change():
    m = ProvenanceStore()
    for v in ["30", "14", "30"]:
        m.remember("k", v)
    assert m.recall("k") == "30"
    assert m.audit("k").revision == 2          # 30->14->30 is two real changes, not idempotent


# --- equality (audit #2 BUG4/BUG10: shallow _eq dropped genuinely-different container facts) --------

def test_type_aware_equality_scalars():
    m = ProvenanceStore()
    m.remember("a", 1, t=1); m.remember("a", True, t=2)
    assert m.recall("a") is True and m.audit("a").revision == 1
    m.remember("b", 1, t=1); m.remember("b", 1.0, t=2)
    assert m.recall("b") == 1.0 and type(m.recall("b")) is float


def test_type_aware_equality_containers():
    m = ProvenanceStore()
    m.remember("n", [1], t=1)
    m.remember("n", [True], t=2)                # different fact per the type-aware claim
    assert m.recall("n") == [True]
    assert m.audit("n").revision == 1


def test_nested_nan_is_idempotent():
    m = ProvenanceStore()
    m.remember("ln", [float("nan")], t=1)
    r = m.remember("ln", [float("nan")], t=2)
    assert r.revision == 0                       # symmetric with top-level NaN handling


# --- mutation immutability (audit #1: v0 stored by reference) --------------------------------------

def test_deep_copy_history_is_immutable():
    m = ProvenanceStore()
    payload = ["a"]
    m.remember("k", payload)
    payload.append("X")
    assert m.recall("k") == ["a"]
    assert m.history("k")[0].value == ["a"]


# --- idempotency holes (audit #2 BUG6) ------------------------------------------------------------

def test_idempotent_same_value_older_t():
    m = ProvenanceStore()
    m.remember("k", "A", t=5)
    r = m.remember("k", "A", t=2)                # same current value, older t -> still idempotent
    assert r.revision == 0
    assert len(m.history("k")) == 1


# --- persistence contract (audit #2 BUG1/BUG2/BUG3) -----------------------------------------------

def test_persistence_across_process_boundary():
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    w = ProvenanceStore(path=path)
    w.remember("language", "pl", session="s0")
    w.remember("language", "en", session="s3")
    del w
    r = ProvenanceStore(path=path)               # a brand-new process attaching to the log
    assert r.recall("language") == "en"
    assert r.audit("language").superseded == "pl"
    assert r.audit("language").revision == 1


def test_persistence_reload_is_idempotent_no_revision_churn():
    """BUG1: re-asserting the same fact each session must NOT grow revision forever."""
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    for _ in range(4):
        s = ProvenanceStore(path=path)
        s.remember("x", "v")
        del s
    assert ProvenanceStore(path=path).audit("x").revision == 0


def test_persistence_rejects_non_json_native_atomically():
    """BUG2: a non-round-trip-stable value is rejected BEFORE any state change -- nothing in memory,
    nothing on disk (no divergence)."""
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    s = ProvenanceStore(path=path)
    for bad in [(1, 2), {1, 2}, {1: "a"}]:       # tuple, set, int-keyed dict
        with pytest.raises(ValueError):
            s.remember("k", bad)
        assert s.recall("k") is None             # not stored in memory
    assert ProvenanceStore(path=path).recall("k") is None   # not on disk either


def test_persistence_tolerates_corrupt_line():
    """BUG3: a truncated/partial line (crash mid-write) is skipped, not fatal; good records survive."""
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    s = ProvenanceStore(path=path)
    s.remember("good", "value", t=1)
    del s
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"key": "oops", "value": "trunc')   # simulated crash mid-write
    r = ProvenanceStore(path=path)               # must NOT raise
    assert r.recall("good") == "value"
    assert r.load_skipped == 1


def test_in_memory_mode_allows_arbitrary_objects():
    """The persistence contract restricts values; the in-memory mode (no path) does not."""
    m = ProvenanceStore()
    m.remember("k", (1, 2))
    assert m.recall("k") == (1, 2)


# --- second independent audit (opus): regressions the rewrite introduced -------------------------

def test_persistence_rejects_bad_source_atomically():
    """NEW-1: the gate must validate the WHOLE record. A non-JSON source/session must be rejected
    BEFORE any mutation -- no disk/memory divergence via an unchecked field."""
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    s = ProvenanceStore(path=path)
    s.remember("k", "A")
    with pytest.raises(ValueError):
        s.remember("k", "B", source={1, 2})       # bad source (a set)
    with pytest.raises(ValueError):
        s.remember("k", "B", session={1, 2})       # bad session
    assert s.recall("k") == "A"                     # memory NOT advanced
    assert ProvenanceStore(path=path).recall("k") == "A"   # disk agrees -> no divergence


def test_corroboration_from_new_source_is_recorded():
    """NEW-2 / prior bug 9: a different session/source re-attesting the current value is NOT dropped --
    it lands in history as corroboration (current value unchanged, no spurious revision)."""
    m = ProvenanceStore()
    m.remember("fact", "blue", source="observer_A", t=1)
    m.remember("fact", "blue", source="observer_B", t=10)   # corroboration, different source
    assert m.recall("fact") == "blue"
    assert m.audit("fact").revision == 0
    sources = [r.source for r in m.history("fact")]
    assert "observer_A" in sources and "observer_B" in sources


def test_identical_reattest_same_attestor_is_pure_noop():
    m = ProvenanceStore()
    m.remember("k", "v", source="a")
    m.remember("k", "v", source="a")               # identical fact + attestor
    assert len(m.history("k")) == 1                 # not recorded twice


def test_cyclic_value_does_not_crash():
    """NEW-5: a cyclic in-memory value must not RecursionError the store on a later update."""
    m = ProvenanceStore()
    cyc = []
    cyc.append(cyc)
    m.remember("k", cyc)
    m.remember("k", [1, 2, 3])                      # must not raise
    assert m.recall("k") == [1, 2, 3]


# --- third independent audit (opus): regressions the round-2 fixes introduced --------------------

def test_deep_but_equal_value_is_idempotent():
    """R1: a legitimately-deep value (nested >100) that is structurally equal must be recognized as the
    same fact -- the old depth cap wrongly declared it different and churned history/revision."""
    import json
    blob = "[" * 200 + "1" + "]" * 200
    v1 = json.loads(blob)
    v2 = json.loads(blob)                           # equal structure, distinct object
    m = ProvenanceStore()
    m.remember("k", v1, t=1)
    r = m.remember("k", v2, t=2)
    assert r.revision == 0
    assert len(m.history("k")) == 1


def test_repeated_corroboration_is_bounded():
    """R2: repeatedly re-attesting the same value from the same non-current attestor must not bloat the
    log -- it is dedup'd against the immediately-preceding observation."""
    m = ProvenanceStore()
    m.remember("k", "A", source="s1")
    for _ in range(5):
        m.remember("k", "A", source="s2")           # same non-current attestor, repeated
    src2 = [r for r in m.history("k") if r.source == "s2"]
    assert len(src2) == 1                            # recorded once, not five times


def test_stale_last_entry_does_not_block_promotion():
    """audit-3b (HIGH): an out-of-order older write leaves a stale value as the LAST history entry; a
    later NEWER write of that same value must still be recorded and become current (not dedup-dropped)."""
    m = ProvenanceStore()
    m.remember("k", "A", t=5)
    m.remember("k", "B", t=2)                        # out-of-order older; current stays A, last entry = B
    m.remember("k", "B", t=10)                       # newer -> must promote to current
    assert m.recall("k") == "B"
    assert m.audit("k").revision == 1


def test_stale_last_entry_promotion_persists():
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    w = ProvenanceStore(path=path)
    w.remember("k", "A", t=5)
    w.remember("k", "B", t=2)
    w.remember("k", "B", t=10)
    del w
    assert ProvenanceStore(path=path).recall("k") == "B"   # newer write not lost on disk


# --- cross-vendor audit (Codex): load-path + deep-value corners the Claude audits missed ----------

def test_corrupt_utf8_bytes_line_is_skipped():
    """Codex #1 + re-verify: invalid UTF-8 bytes -- whether a whole garbage line OR bytes INSIDE a JSON
    string -- must be skipped on load (not crash, and not silently loaded as replacement-char data)."""
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    with open(path, "wb") as f:
        f.write(b'{"key":"good","t":1,"value":"ok","session":"s","source":null}\n')
        f.write(b"\xff\xfe\xfa\n")                                        # whole-line garbage
        f.write(b'{"key":"instr","t":2,"value":"A\xffB","session":"s","source":null}\n')  # bad bytes in a string
    r = ProvenanceStore(path=path)                    # must NOT raise UnicodeDecodeError
    assert r.recall("good") == "ok"
    assert r.recall("instr") is None                  # corrupt-in-string line skipped, not loaded mangled
    assert r.load_skipped == 2


def test_nan_line_is_rejected_on_load():
    """Codex #2: json.loads accepts NaN by default; load must enforce the write contract and skip it."""
    d = tempfile.mkdtemp(prefix="provmem_")
    path = os.path.join(d, "mem.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"key":"good","t":1,"value":"ok","session":"s","source":null}\n')
        f.write('{"key":"bad","t":2,"value": NaN,"session":"s","source":null}\n')
    r = ProvenanceStore(path=path)
    assert r.recall("good") == "ok"
    assert r.recall("bad") is None                    # NaN line not smuggled in
    assert r.load_skipped == 1


def test_too_deep_value_fails_cleanly_not_recursionerror():
    """Codex #3: a value nested past the recursion limit must raise a clear ValueError, not crash.
    Depth 5000 also overflows repr() at the default limit on 3.12+, so it regresses the persist-path
    bug where the ValueError message did `{fv!r}` and RecursionError leaked out before the raise."""
    v = []
    for _ in range(5000):
        v = [v]
    m = ProvenanceStore()
    with pytest.raises(ValueError):                   # in-memory: deepcopy guarded
        m.remember("deep", v)
    assert m.recall("deep") is None                   # nothing stored (state intact)
    d = tempfile.mkdtemp(prefix="provmem_")
    p = ProvenanceStore(path=os.path.join(d, "m.jsonl"))
    with pytest.raises(ValueError):                   # persist: _json_native guard + bounded repr in msg
        p.remember("deep", v)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            print(f"ok  {name}")
        except Exception as e:  # pytest.raises works standalone too
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            raise
    print(f"all {len(fns)} passed")
