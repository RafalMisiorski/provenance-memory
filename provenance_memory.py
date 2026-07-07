"""provenance_memory -- a tiny, dependency-free agent memory that doesn't go stale, doesn't lose
cross-session facts, and can always tell you WHY it recalled something.

Most agent memory is an append-only vector bag: every fact you write is added, nothing is ever
superseded, and retrieval is "whatever cosine says is nearest". Three failure modes fall out of that
(the mem0 head-to-head that motivates them ships here as bench/bench_vs_mem0.py):

  STALE          a fact changes; the old value is still in the index; similarity can't tell which is
                 current, so you get the superseded one.
  XSESSION_MISS  memory is partitioned per session/user-thread; a fact from an earlier session is
                 invisible later.
  NON_DET        an LLM (or a tie-break) decides what to recall, so the same query returns different
                 things and you can't audit why.

Design (deliberately boring + transparent -- no model, no API key, no vector db):

  * change-detection      the value with the greatest logical timestamp t is current. A STRICTLY newer
                          t supersedes; a late-arriving OLDER-or-equal write is recorded in history but
                          never dethrones the current value (out-of-order safe).
  * type-aware equality   idempotency + change-detection compare with type-strict, DEEP equality, so
                          1 vs True, 1 vs 1.0, [1] vs [True] are all different facts (never silently
                          dropped as "the same"). Legitimately-deep values compare fully; only a true
                          cycle is caught (RecursionError) and falls back to identity, never crashing.
  * corroboration kept    re-observing a value from a DIFFERENT session/source is recorded in history()
                          as corroboration (provenance is not dropped). A write identical to the
                          immediately-preceding observation from the SAME attestor is a pure no-op, so a
                          re-write loop can't bloat the log.
  * one global namespace  facts are not siloed per session. store in session 1, recall in session 9.
  * deterministic + audit recall(k) is an O(1) lookup: same input, same output. audit(k) gives value,
                          when, which session, source, revision, and what it superseded -- correct under
                          out-of-order writes (a late stale write does NOT inflate them).

Persistence (opt-in, with a contract stated plainly rather than silently violated):
  ProvenanceStore(path=...) appends every accepted observation to a JSONL log and replays it on init,
  so state survives a PROCESS boundary.
    - the WHOLE record (value, session, source) must be JSON-serializable and round-trip stable:
      str / int / float / bool / None / list / dict-with-string-keys. A tuple, set, bytes, datetime,
      int-keyed dict, or NaN/Inf -- in ANY field -- is REJECTED with a ValueError BEFORE anything is
      stored (no silent corruption, no half-write). Use in-memory mode (no path) for arbitrary objects.
    - a bad line -- truncated/partial (crash mid-write), invalid UTF-8 bytes, or a value that violates
      the contract (e.g. NaN/Inf, which json.loads accepts by default) -- is SKIPPED on load, not fatal;
      the count is exposed as `.load_skipped`. Load enforces the SAME JSON-native contract as writes, so
      one bad line never bricks the store and never smuggles in a value the writer would have rejected.
    - SINGLE-WRITER per path. Two live stores appending to one file is unsupported (private clocks + no
      file lock -> auto-t collisions and divergent in-memory views).
    - NOT crash-atomic per write: validation runs before any mutation, but the in-memory state is
      updated and THEN the record is appended to disk. A mid-write I/O error (e.g. disk full) can
      leave the last record in memory but not on disk; a subsequent reopen would not see it. State is
      consistent within a process; durability of the final write is best-effort, not two-phase.
  A value nested beyond the interpreter's recursion limit (~hundreds deep) is rejected with a clear
  ValueError on store, not a RecursionError crash -- state is left intact.

Honest scope: on in-order, single-value-per-key facts this is equivalent to a ~15-line vector-store +
newest-wins query -- there is no capability moat. What it buys you is a correct, tested, persistent,
audit-carrying drop-in so you don't hand-roll (and mis-handle out-of-order / mutation / persistence)
yourself. The deterministic-freshness idea is also known in the literature (arXiv 2606.01435).
"""
from __future__ import annotations

import copy
import json
import math
import os
import reprlib
from dataclasses import dataclass
from typing import Any, Optional

def _strict_eq(a: Any, b: Any) -> bool:
    """Type-strict, DEEP equality. type(a) is type(b) at every level; NaN equals NaN; containers are
    compared element-wise so [1] != [True]. Legitimately deep values compare fully; only a CYCLE (which
    would recurse forever) is caught via RecursionError and falls back to identity -- so this never
    crashes, and a merely-deep-but-equal value is still recognized as equal (not falsely churned)."""
    try:
        return _seq(a, b)
    except RecursionError:
        return a is b


def _seq(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_seq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_seq(a[k], b[k]) for k in a)
    return a == b


def _json_native(v: Any) -> bool:
    """True iff v survives a JSON round-trip unchanged (so persistence can't silently corrupt its
    type). Rejects tuple/set/bytes/datetime, int/float-keyed dicts, NaN/Inf."""
    try:
        back = json.loads(json.dumps(v, allow_nan=False))
        return back == v and _same_shape(back, v)
    except (TypeError, ValueError, RecursionError):
        return False


def _same_shape(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_shape(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_shape(a[k], b[k]) for k in a)
    return True


@dataclass
class Record:
    """A stored observation / the current value, plus the provenance to audit it."""
    key: str
    value: Any
    t: int                      # logical timestamp of the write
    session: str                # which session/thread wrote it
    source: Optional[str]       # optional origin tag (doc id, tool, user, ...)
    revision: int               # audit(): times the current value actually changed. history(): arrival index
    superseded: Optional[Any]   # audit(): the value the current one actually replaced. history(): None


class ProvenanceStore:
    """Deterministic, cross-session, change-detecting key/value memory with a correct audit trail.

    Public API: remember / recall / audit / history / keys. No third-party deps, no randomness.
    Optional `path` gives durable, cross-process persistence (SINGLE-WRITER; see the module docstring).
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._history: dict[str, list[dict]] = {}   # key -> observations in ARRIVAL order (audit trail)
        self._current: dict[str, dict] = {}         # key -> the winning observation (O(1) recall)
        self._rev: dict[str, int] = {}              # key -> count of real current-value changes
        self._prev: dict[str, Any] = {}             # key -> value the current one replaced
        self._clock = 0
        self._path = path
        self.load_skipped = 0                       # corrupt/partial lines skipped on load (transparency)
        if path and os.path.exists(path):
            self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        # Read as bytes and STRICT-decode each line: a line with invalid UTF-8 ANYWHERE (even inside a
        # JSON string) is skipped whole -- never silently repaired with replacement chars (which would
        # smuggle in mangled data) and never allowed to raise UnicodeDecodeError and brick the load.
        with open(self._path, "rb") as f:
            for bline in f:
                try:
                    line = bline.decode("utf-8").strip()
                except UnicodeDecodeError:
                    self.load_skipped += 1
                    continue
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    key = obs["key"]
                    t = int(obs["t"])
                    value = obs["value"]
                    session = obs.get("session", "default")
                    source = obs.get("source")
                except (ValueError, KeyError, TypeError):
                    self.load_skipped += 1          # corrupt/truncated/partial line -> skip, never fatal
                    continue
                # enforce the SAME JSON-native contract on load as on write: json.loads() accepts NaN/Inf
                # by default, so a hand-edited/corrupt log could inject a value the writer would reject.
                if not (_json_native(value) and _json_native(session) and _json_native(source)):
                    self.load_skipped += 1
                    continue
                self._ingest(key, value, t, session, source, persist=False)

    def _append_disk(self, key: str, obs: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **obs}, sort_keys=True) + "\n")

    # -- core ingest (shared by remember + replay) --------------------------
    def _ingest(self, key: str, value: Any, t: int, session: str, source: Optional[str],
                persist: bool) -> None:
        cur = self._current.get(key)
        same_value = cur is not None and _strict_eq(cur["value"], value)
        # A write equal to the CURRENT value can never promote a different value, so skipping it is
        # always safe. If it also duplicates the immediately-preceding observation from the same
        # attestor, it carries no new info -> pure no-op (bounds trivial re-write loops). A write of a
        # DIFFERENT value is ALWAYS recorded: it is either a promotion or genuine out-of-order history --
        # never dedup'd (that was audit-3b: a stale last-entry must not block a newer promotion).
        if same_value:
            last = self._history[key][-1]
            if last["session"] == session and last["source"] == source and _strict_eq(last["value"], value):
                self._clock = max(self._clock, t)
                return
        # persistence validation covers the WHOLE record (value, session, source) BEFORE any mutation
        if persist and self._path is not None:
            for field, fv in (("value", value), ("session", session), ("source", source)):
                if not _json_native(fv):
                    # reprlib.repr (bounded maxlevel/maxlist) — a plain {fv!r} recurses on the very
                    # kind of value this guard rejects (a too-deep structure), raising RecursionError
                    # INSIDE the error message before the ValueError is built (version-dependent).
                    raise ValueError(
                        f"persistence mode requires JSON-native, round-trip-stable {field}; "
                        f"got {type(fv).__name__} {reprlib.repr(fv)} (use in-memory mode for arbitrary objects)")
        try:
            stored = copy.deepcopy(value)
        except RecursionError:
            raise ValueError("value is nested too deeply to store (exceeds the interpreter recursion "
                             "limit); flatten it or raise sys.setrecursionlimit before storing")
        obs = {"value": stored, "t": t, "session": session, "source": source}
        self._history.setdefault(key, []).append(obs)    # corroboration or a real change is recorded
        self._clock = max(self._clock, t)
        if not same_value and (cur is None or t > cur["t"]):   # STRICT newer + different value supersedes
            self._prev[key] = cur["value"] if cur is not None else None
            self._rev[key] = self._rev.get(key, 0) + (1 if cur is not None else 0)
            self._current[key] = obs
        if persist and self._path is not None:
            self._append_disk(key, obs)

    # -- writes --------------------------------------------------------------
    def remember(self, key: str, value: Any, *, session: str = "default",
                 source: Optional[str] = None, t: Optional[int] = None) -> Optional[Record]:
        """Store (or update) a fact. The value with the greatest t is current; a strictly newer,
        different value supersedes. Re-writing the CURRENT value from the same attestor is idempotent;
        from a different session/source it is recorded as corroboration. A late-arriving older/equal
        write is kept in history but does not change the current value. Returns the current authoritative
        record (== audit(key)) -- NOT a write-receipt: on an out-of-order/corroborating write the return
        reflects current state, which may differ from what you just passed."""
        if t is None:
            self._clock += 1
            t = self._clock
        self._ingest(key, value, t, session, source, persist=True)
        return self.audit(key)

    update = remember

    # -- reads (deterministic) ----------------------------------------------
    def recall(self, key: str) -> Optional[Any]:
        """Return the CURRENT value (greatest t), cross-session, or None. O(1), pure -> deterministic."""
        cur = self._current.get(key)
        return copy.deepcopy(cur["value"]) if cur is not None else None

    def audit(self, key: str) -> Optional[Record]:
        """Authoritative provenance for the current value: value, when, session, source, how many times
        it actually changed, and what it replaced. Correct under out-of-order writes."""
        cur = self._current.get(key)
        if cur is None:
            return None
        return Record(key=key, value=copy.deepcopy(cur["value"]), t=cur["t"], session=cur["session"],
                      source=cur["source"], revision=self._rev.get(key, 0),
                      superseded=copy.deepcopy(self._prev.get(key)))

    def history(self, key: str) -> list[Record]:
        """Every observation for key as Records, in ARRIVAL order (includes late/stale writes and
        corroborations). Here `revision` is the arrival index and `superseded` is None; use audit() for
        the authoritative current-value provenance."""
        out = []
        for i, o in enumerate(self._history.get(key, [])):
            out.append(Record(key=key, value=copy.deepcopy(o["value"]), t=o["t"],
                              session=o["session"], source=o["source"], revision=i, superseded=None))
        return out

    def keys(self) -> list[str]:
        return sorted(self._current)


__all__ = ["ProvenanceStore", "Record"]


def _demo() -> None:
    def p(s: str) -> None:
        print(str(s).encode("ascii", "replace").decode("ascii"))

    m = ProvenanceStore()
    m.remember("refund_window_days", "30", session="s1", source="policy_v1")
    m.remember("refund_window_days", "14", session="s2", source="policy_v2")  # supersedes 30
    m.remember("refund_window_days", "7",  session="s5", source="policy_v3")  # supersedes 14
    p(f"recall (current, cross-session) -> {m.recall('refund_window_days')}")   # 7
    a = m.audit("refund_window_days")
    p(f"audit -> value={a.value} t={a.t} session={a.session} rev={a.revision} replaced={a.superseded}")
    p(f"history values -> {[r.value for r in m.history('refund_window_days')]}")   # ['30','14','7']
    m.remember("region", "eu", t=10)
    m.remember("region", "us", t=3)   # older fact, arrives second
    a = m.audit("region")
    p(f"out-of-order recall -> {m.recall('region')}  rev={a.revision} replaced={a.superseded}"
      f"  (must be 'eu', rev 0, replaced None)")
    p(f"deterministic over 100 runs -> {len({m.recall('refund_window_days') for _ in range(100)}) == 1}")


if __name__ == "__main__":
    _demo()
