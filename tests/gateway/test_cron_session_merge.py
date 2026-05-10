"""Tests for gateway.cron_session_merge — the cron→chat-session merge hook.

See module docstring of gateway/cron_session_merge.py for context: we restore
the cron breadcrumb in the chat session by UPDATEing the last assistant
message instead of INSERTing a new one (which would break alternation).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_hermes_home(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        (home / "sessions").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        # Force any cached get_hermes_home() consumers to use the new dir.
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "_HERMES_HOME_OVERRIDE", None, raising=False)

        # gateway.mirror caches the path at import time — patch it.
        import gateway.mirror as mirror_mod
        monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", home / "sessions")
        monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", home / "sessions" / "sessions.json")

        yield home


@pytest.fixture
def session_with_messages(tmp_hermes_home, monkeypatch):
    """Create a SessionDB with a session and pre-populated index entry.

    Returns (session_id, db_path, append_msg_callable).
    """
    from hermes_state import SessionDB

    db_path = tmp_hermes_home / "state.db"
    import hermes_state as hs
    monkeypatch.setattr(hs, "DEFAULT_DB_PATH", db_path)
    db = SessionDB(db_path=db_path)
    session_id = "test-sess-001"
    db.create_session(session_id, source="gateway")

    # Build a sessions.json that gateway.mirror._find_session_id can match.
    index = {
        session_id: {
            "session_id": session_id,
            "platform": "telegram",
            "origin": {"platform": "telegram", "chat_id": "123", "user_id": "u1"},
            "updated_at": "2026-01-01T00:00:00",
        }
    }
    (tmp_hermes_home / "sessions" / "sessions.json").write_text(json.dumps(index))

    def _append(role, content):
        return db.append_message(session_id=session_id, role=role, content=content)

    yield session_id, db_path, _append
    db.close()


def _read_messages(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, role, content FROM messages ORDER BY id"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
    finally:
        conn.close()


def _enable_merge(monkeypatch, **overrides):
    cfg = {"enabled": True, "log_merges": True, "skip_if_target_is_local_or_all": True}
    cfg.update(overrides)
    monkeypatch.setattr(
        "gateway.cron_session_merge._load_merge_config",
        lambda: cfg,
    )


def test_updates_last_assistant_in_place(session_with_messages, monkeypatch):
    _enable_merge(monkeypatch)
    session_id, db_path, append = session_with_messages
    append("user", "hi")
    append("assistant", "hello there")
    rows_before = _read_messages(db_path)
    assert len(rows_before) == 2

    from gateway.cron_session_merge import merge_cron_into_chat_session
    ok = merge_cron_into_chat_session(
        platform="telegram",
        chat_id="123",
        message_text="cron says hi",
        cron_job_name="撩阳",
        cron_job_id="job-abc",
        user_id="u1",
        fire_ts=1700000000.0,
    )
    assert ok is True

    rows_after = _read_messages(db_path)
    assert len(rows_after) == 2, "row count must NOT change (UPDATE not INSERT)"
    last = rows_after[-1]
    assert last[1] == "assistant"
    assert "hello there" in last[2]
    assert "cron says hi" in last[2]
    assert "撩阳" in last[2]
    assert "cron-merge:job-abc:1700000000" in last[2]


def test_inserts_when_last_is_user(session_with_messages, monkeypatch):
    _enable_merge(monkeypatch)
    session_id, db_path, append = session_with_messages
    append("user", "still talking")

    from gateway.cron_session_merge import merge_cron_into_chat_session
    ok = merge_cron_into_chat_session(
        platform="telegram", chat_id="123",
        message_text="cron text", cron_job_name="j", cron_job_id="jid",
        user_id="u1",
    )
    assert ok is True
    rows = _read_messages(db_path)
    assert len(rows) == 2
    assert rows[-1][1] == "assistant"
    assert "cron text" in rows[-1][2]


def test_inserts_when_session_empty(session_with_messages, monkeypatch):
    _enable_merge(monkeypatch)
    session_id, db_path, _append = session_with_messages

    from gateway.cron_session_merge import merge_cron_into_chat_session
    ok = merge_cron_into_chat_session(
        platform="telegram", chat_id="123",
        message_text="lone cron", cron_job_name="j", cron_job_id="jid",
        user_id="u1",
    )
    assert ok is True
    rows = _read_messages(db_path)
    assert len(rows) == 1
    assert rows[0][1] == "assistant"
    assert "lone cron" in rows[0][2]


def test_disabled_is_noop(session_with_messages, monkeypatch):
    monkeypatch.setattr(
        "gateway.cron_session_merge._load_merge_config",
        lambda: {"enabled": False, "log_merges": True, "skip_if_target_is_local_or_all": True},
    )
    session_id, db_path, append = session_with_messages
    append("assistant", "prior")

    from gateway.cron_session_merge import merge_cron_into_chat_session
    ok = merge_cron_into_chat_session(
        platform="telegram", chat_id="123",
        message_text="x", cron_job_name="j", cron_job_id="jid",
    )
    assert ok is False
    rows = _read_messages(db_path)
    assert len(rows) == 1
    assert rows[0][2] == "prior"


@pytest.mark.parametrize("plat", ["local", "all", "LOCAL", "ALL", ""])
def test_skips_local_or_all_target(session_with_messages, monkeypatch, plat):
    _enable_merge(monkeypatch)
    from gateway.cron_session_merge import merge_cron_into_chat_session
    ok = merge_cron_into_chat_session(
        platform=plat, chat_id="x",
        message_text="y", cron_job_name="j", cron_job_id="jid",
    )
    assert ok is False


def test_sessiondb_exception_does_not_propagate(session_with_messages, monkeypatch):
    _enable_merge(monkeypatch)

    class Boom:
        def __init__(self): raise RuntimeError("db unavailable")

    monkeypatch.setattr("hermes_state.SessionDB", Boom)

    from gateway.cron_session_merge import merge_cron_into_chat_session
    # Must not raise.
    ok = merge_cron_into_chat_session(
        platform="telegram", chat_id="123",
        message_text="x", cron_job_name="j", cron_job_id="jid",
        user_id="u1",
    )
    assert ok is False


def test_idempotent_double_merge(session_with_messages, monkeypatch):
    """Running the same (job_id, fire_ts) twice must not double-append."""
    _enable_merge(monkeypatch)
    session_id, db_path, append = session_with_messages
    append("assistant", "base")

    from gateway.cron_session_merge import merge_cron_into_chat_session
    kwargs = dict(
        platform="telegram", chat_id="123",
        message_text="once", cron_job_name="j", cron_job_id="jid",
        user_id="u1", fire_ts=1700000000.0,
    )
    assert merge_cron_into_chat_session(**kwargs) is True
    # second call same fire_ts -> skipped, no further mutation
    assert merge_cron_into_chat_session(**kwargs) is False

    rows = _read_messages(db_path)
    assert len(rows) == 1
    # exactly one occurrence of "once"
    assert rows[0][2].count("once") == 1
    assert rows[0][2].count("cron-merge:jid:1700000000") == 1
