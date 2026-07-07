"""Tests for the mem0-shaped adapter. Deterministic, dependency-free."""
import os
import subprocess
import sys
import tempfile

from mem0_shim import Memory


def test_changed_fact_supersedes():
    m = Memory()
    m.add("refund_window_days", "30", user_id="acme")
    m.add("refund_window_days", "14", user_id="acme")
    m.add("refund_window_days", "7", user_id="acme")
    assert m.get("refund_window_days", user_id="acme") == "7"


def test_user_namespaces_are_isolated():
    m = Memory()
    m.add("plan", "free", user_id="alice")
    m.add("plan", "pro", user_id="bob")
    assert m.get("plan", user_id="alice") == "free"
    assert m.get("plan", user_id="bob") == "pro"


def test_get_all_returns_only_current_values():
    m = Memory()
    m.add("a", "1", user_id="u")
    m.add("a", "2", user_id="u")   # supersede
    m.add("b", "x", user_id="u")
    got = {d["key"]: d["value"] for d in m.get_all(user_id="u")}
    assert got == {"a": "2", "b": "x"}


def test_get_all_excludes_other_users():
    m = Memory()
    m.add("k", "mine", user_id="u1")
    m.add("k", "theirs", user_id="u2")
    keys = [(d["key"], d["value"]) for d in m.get_all(user_id="u1")]
    assert keys == [("k", "mine")]


def test_search_is_substring_key_match():
    m = Memory()
    m.add("refund_window_days", "7", user_id="u")
    m.add("shipping_days", "3", user_id="u")
    hits = {d["key"] for d in m.search("days", user_id="u")}
    assert hits == {"refund_window_days", "shipping_days"}
    hits2 = {d["key"] for d in m.search("refund", user_id="u")}
    assert hits2 == {"refund_window_days"}


def test_audit_reports_supersession():
    m = Memory()
    m.add("x", "old", user_id="u")
    m.add("x", "new", user_id="u")
    a = m.audit("x", user_id="u")
    assert a["value"] == "new"
    assert a["superseded"] == "old"
    assert a["revision"] == 1


def test_audit_absent_key_is_none():
    m = Memory()
    assert m.audit("nope", user_id="u") is None


def test_history_is_full_chain_in_arrival_order():
    m = Memory()
    m.add("x", "a", user_id="u")
    m.add("x", "b", user_id="u")
    m.add("x", "c", user_id="u")
    assert [r["value"] for r in m.history("x", user_id="u")] == ["a", "b", "c"]


def test_out_of_order_write_does_not_dethrone_current():
    # explicit logical times: a late-arriving OLDER write must not win
    m = Memory()
    m.add("x", "new", user_id="u", t=10)
    m.add("x", "stale", user_id="u", t=2)   # arrives later, older t
    assert m.get("x", user_id="u") == "new"


def test_reset_clears_state():
    m = Memory()
    m.add("x", "1", user_id="u")
    m.reset()
    assert m.get("x", user_id="u") is None


def test_persistence_across_process():
    # a fresh Memory on the same path recalls prior facts -- the cross-session claim
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        m1 = Memory(path=path)
        m1.add("refund_window_days", "30", user_id="acme")
        m1.add("refund_window_days", "7", user_id="acme")
        # brand-new process reading the same log
        code = (
            "from mem0_shim import Memory;"
            f"m=Memory(path={path!r});"
            "print(m.get('refund_window_days', user_id='acme'))"
        )
        env = dict(os.environ)
        here = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=here, env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "7", out.stdout
    finally:
        os.remove(path)


def test_reset_truncates_and_keeps_persisting():
    # reset() on a persisted store must wipe the log AND keep persisting to the same path:
    # a new process re-opening the path sees ONLY post-reset facts (not the silent in-memory
    # switch the old reset() caused, which dropped persistence while keeping the stale log).
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        m = Memory(path=path)
        m.add("k", "before", user_id="u")
        m.reset()
        m.add("k", "after", user_id="u")
        code = (
            "from mem0_shim import Memory;"
            f"m=Memory(path={path!r});"
            "print(repr(m.get('k', user_id='u')))"
        )
        env = dict(os.environ)
        here = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=here, env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "'after'", out.stdout   # post-reset write persisted; pre-reset gone
    finally:
        os.remove(path)


def test_add_returns_current_record_dict():
    m = Memory()
    r = m.add("k", "v", user_id="u")
    assert r["key"] == "k" and r["value"] == "v"
