"""mem0-shaped API over provenance-memory -- the DX-parity adapter.

Why this exists
---------------
mem0 is the memory layer most agent devs actually reach for. Its LLM write-path,
however, does not reliably preserve the write-time metadata you attach, so on a fact
that CHANGES ("refund window 30 -> 14 -> 7") it tends to keep every version and return
a superseded one by similarity -- ~25% stale even when you sort its candidates by your
own timestamp (measured, pre-registered: see bench/ and MEM0_HEADTOHEAD.md).

This adapter gives you mem0's shape (`add` / `search` / `get_all` / `get` / `reset`)
backed by the deterministic provenance store, so a changed fact ALWAYS supersedes, at
0 LLM cost, with an audit trail mem0 does not expose.

Honest scope (read this before swapping anything)
-------------------------------------------------
This is mem0-SHAPED, NOT a verbatim drop-in. The differences ARE the trade:
  * You pass a STRUCTURED fact -- `add(key, value)` -- not a raw conversation. mem0 runs
    an LLM to EXTRACT facts from messages; this does not. That extraction is exactly where
    mem0's nondeterminism and timestamp-corruption enter, so replicating it would throw
    away the win. If you already have (key, value) facts -- tool outputs, structured state,
    user settings, extracted entities -- this is for you. If you need fact extraction from
    free text, keep mem0 (or run your own extractor, then feed the facts here).
  * `search()` is EXACT / substring key match, not semantic vector search. For "recall the
    CURRENT value of a known fact" that is what you want; for fuzzy semantic retrieval mem0
    wins.
  * There is no moat here: a ~15-line "store t in metadata + recall by max(t)" hand-roll
    gets the same 0% stale. The point is not a secret algorithm -- it is a correct,
    tested, zero-dependency, persistent, auditable version of the thing you would otherwise
    hand-roll, and a demonstration that mem0's default does NOT give you.

Everything here is deterministic and dependency-free.
"""
from __future__ import annotations

from typing import Any, Optional

from provenance_memory import ProvenanceStore, Record

__all__ = ["Memory"]

_NS = "\x1f"  # unit separator: namespaces user_id from key without colliding with real keys


class Memory:
    """A mem0-shaped, deterministic, provenance-tracked fact store for STRUCTURED facts.

    >>> m = Memory()
    >>> _ = m.add("refund_window_days", "30", user_id="acme")
    >>> _ = m.add("refund_window_days", "14", user_id="acme")   # supersedes 30
    >>> _ = m.add("refund_window_days", "7",  user_id="acme")    # supersedes 14
    >>> m.get("refund_window_days", user_id="acme")
    '7'
    >>> [r["value"] for r in m.search("refund", user_id="acme")]
    ['7']
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._store = ProvenanceStore(path=path)

    # -- namespacing ---------------------------------------------------------
    @staticmethod
    def _k(user_id: str, key: str) -> str:
        return f"{user_id}{_NS}{key}"

    @staticmethod
    def _split(nskey: str) -> tuple[str, str]:
        user_id, _, key = nskey.partition(_NS)
        return user_id, key

    # -- writes --------------------------------------------------------------
    def add(self, key: str, value: Any, *, user_id: str = "default",
            source: Optional[str] = None, t: Optional[int] = None) -> dict:
        """Store or update a structured fact for `user_id`. The value with the greatest
        logical time is current; a strictly newer, different value supersedes. Returns the
        current authoritative record as a dict (see `audit`). NOTE: takes a (key, value)
        fact, not a messages list -- see the module docstring on scope."""
        rec = self._store.remember(self._k(user_id, key), value,
                                   session=source or user_id, source=source, t=t)
        return self._as_dict(key, rec)

    # -- reads (deterministic) ----------------------------------------------
    def get(self, key: str, *, user_id: str = "default") -> Optional[Any]:
        """Current value of a known fact for `user_id`, or None. O(1), deterministic."""
        return self._store.recall(self._k(user_id, key))

    def get_all(self, *, user_id: str = "default") -> list[dict]:
        """Every CURRENT fact for `user_id` (superseded values excluded), key-sorted."""
        out = []
        prefix = f"{user_id}{_NS}"
        for nskey in self._store.keys():
            if nskey.startswith(prefix):
                _, key = self._split(nskey)
                out.append(self._as_dict(key, self._store.audit(nskey)))
        return out

    def search(self, query: str, *, user_id: str = "default") -> list[dict]:
        """Return current facts whose KEY contains `query` (case-insensitive substring).
        Exact/substring, NOT semantic -- deterministic and dependency-free. For the
        'recall the current value of a known fact' job this is what you want."""
        q = query.lower()
        return [d for d in self.get_all(user_id=user_id) if q in d["key"].lower()]

    # -- provenance (mem0 has no equivalent) --------------------------------
    def audit(self, key: str, *, user_id: str = "default") -> Optional[dict]:
        """Why the current value is current: value, when, source, how many times it
        actually changed (`revision`), and what it replaced (`superseded`). Correct under
        out-of-order writes. This is the pure differentiator -- mem0 does not expose it."""
        rec = self._store.audit(self._k(user_id, key))
        return self._as_dict(key, rec) if rec is not None else None

    def history(self, key: str, *, user_id: str = "default") -> list[dict]:
        """Full supersession chain for a fact, in arrival order."""
        return [self._as_dict(key, r) for r in self._store.history(self._k(user_id, key))]

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Drop all in-memory state. Does NOT truncate a persisted log (open a new path
        for a fresh log); this mirrors keeping the audit trail append-only on disk."""
        self._store = ProvenanceStore(path=None)

    # -- internal ------------------------------------------------------------
    @staticmethod
    def _as_dict(key: str, rec: Optional[Record]) -> dict:
        if rec is None:
            return {"key": key, "value": None}
        return {
            "key": key,
            "value": rec.value,
            "t": rec.t,
            "source": rec.source,
            "revision": rec.revision,
            "superseded": rec.superseded,
        }
