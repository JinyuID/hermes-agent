"""
Merge cron job outputs into the originating chat session's transcript.

Background
----------
Upstream commit 37a9979459 (PR #2313, issue #2221) removed the cron→session
mirror entirely because mirroring inserted a *new* assistant message after a
cron run, which often produced two consecutive assistant messages and broke
provider message-alternation requirements.

This module restores the user-visible benefit of that mirror — the chat-side
agent has context about what its cron-self said — without re-introducing the
alternation hazard.  Strategy:

    1. Look up the chat session (same lookup as gateway/mirror.py).
    2. Read the LAST message in that session.
    3. If role == "assistant"  → APPEND the cron content to that message's
                                  ``content`` field (UPDATE, not INSERT).
                                  No new row, alternation untouched.
    4. Otherwise               → INSERT a new assistant message (safe: the
                                  prior turn was user/tool/empty).

Idempotency: each merge stamps a marker line containing the cron job id and
fire timestamp.  Before merging we check whether the same marker is already
present and skip if so — protects against double-fires and replays.

Never raises: every failure is swallowed and logged at warning level.  Cron
delivery succeeded; the in-session breadcrumb is best-effort.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# Marker template — the cron_job_id makes it unique per job, and the
# fire_ts (epoch seconds, integer) makes it unique per fire.  Together they
# form the idempotency key.
_MARKER_FMT = "<!-- cron-merge:{job_id}:{fire_ts} -->"

_SEPARATOR = "\n\n---\n*[cron · {name}] {hhmm}*\n{marker}\n\n{text}"


def _is_disabled_target(platform: str) -> bool:
    """Targets like 'local' or 'all' have no real chat session to merge into."""
    if not platform:
        return True
    p = platform.lower().strip()
    return p in {"local", "all", "origin", ""}


def _load_merge_config() -> dict:
    """Pull ``gateway.cron_session_merge`` from user config with safe defaults."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        merge_cfg = ((cfg.get("gateway") or {}).get("cron_session_merge") or {})
    except Exception:
        merge_cfg = {}
    return {
        "enabled": bool(merge_cfg.get("enabled", False)),
        "log_merges": bool(merge_cfg.get("log_merges", True)),
        "skip_if_target_is_local_or_all": bool(
            merge_cfg.get("skip_if_target_is_local_or_all", True)
        ),
    }


def merge_cron_into_chat_session(
    platform: str,
    chat_id: str,
    message_text: str,
    cron_job_name: str,
    cron_job_id: str,
    *,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    fire_ts: Optional[float] = None,
) -> bool:
    """
    Best-effort merge of a cron message into the chat session belonging to the
    same (platform, chat_id, thread_id, user_id) tuple.

    Returns
    -------
    bool
        True  : merged (UPDATE) or inserted a new assistant message.
        False : disabled, no matching session, already merged, or an error
                occurred.  Never raises.
    """
    cfg = _load_merge_config()
    if not cfg["enabled"]:
        return False

    if cfg["skip_if_target_is_local_or_all"] and _is_disabled_target(platform):
        return False

    if not message_text or not str(message_text).strip():
        return False

    try:
        # Reuse the session-lookup logic from gateway/mirror.py — we want the
        # exact same matching semantics so the mirror entry lands in the
        # session a real send_message would have hit.
        from gateway.mirror import _find_session_id

        session_id = _find_session_id(
            platform,
            str(chat_id),
            thread_id=thread_id,
            user_id=user_id,
        )
        if not session_id:
            logger.debug(
                "cron-merge: no session for %s:%s thread=%s user=%s",
                platform, chat_id, thread_id, user_id,
            )
            return False

        ts = fire_ts if fire_ts is not None else time.time()
        marker = _MARKER_FMT.format(job_id=cron_job_id or "?", fire_ts=int(ts))
        hhmm = datetime.fromtimestamp(ts).strftime("%H:%M")
        snippet = _SEPARATOR.format(
            name=cron_job_name or cron_job_id or "cron",
            hhmm=hhmm,
            marker=marker,
            text=str(message_text).strip(),
        )

        result = _merge_or_insert(session_id, snippet, marker)

        if result == "skipped-duplicate":
            logger.debug("cron-merge: duplicate marker for job=%s, skipped", cron_job_id)
            return False
        if result and cfg["log_merges"]:
            logger.info(
                "cron-merge: %s for job %s into session %s (%s:%s)",
                result, cron_job_id, session_id, platform, chat_id,
            )
        return bool(result)

    except Exception as exc:
        # Swallow — cron delivery already succeeded; this is breadcrumb-only.
        logger.warning("cron-merge: failed for job=%s: %s", cron_job_id, exc)
        return False


def _merge_or_insert(session_id: str, snippet: str, marker: str) -> Optional[str]:
    """
    Atomically: read last message; if assistant, UPDATE its content with the
    snippet appended (skipping if marker already present); else INSERT a new
    assistant message.

    Returns one of {"updated", "inserted", "skipped-duplicate"} on success,
    or None on lookup failure.
    """
    try:
        from hermes_state import SessionDB
    except Exception as exc:
        logger.debug("cron-merge: SessionDB import failed: %s", exc)
        return None

    db = None
    try:
        db = SessionDB()

        def _do(conn):
            row = conn.execute(
                "SELECT id, role, content FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()

            if row is not None:
                msg_id, role, content = row[0], row[1], row[2]
                # Idempotency: bail out if the same fire was already merged.
                if content and marker in str(content):
                    return "skipped-duplicate"

                if role == "assistant":
                    decoded = _decode_text_content(content)
                    new_content = (decoded or "") + snippet
                    # Re-encode if original was a list-of-parts; otherwise
                    # plain string.  Keep it simple: append as plain text and
                    # re-serialize iff the original was JSON-encoded list.
                    stored = _reencode_like(content, new_content)
                    conn.execute(
                        "UPDATE messages SET content = ? WHERE id = ?",
                        (stored, msg_id),
                    )
                    return "updated"

            # No prior message OR last message is not assistant — safe to
            # INSERT a brand-new assistant turn.  Strip the leading
            # separator so the new turn starts cleanly.
            standalone = snippet.lstrip("\n").lstrip("-").lstrip("\n")
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'assistant', ?, ?)",
                (session_id, standalone, time.time()),
            )
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (session_id,),
            )
            _ = cursor.lastrowid
            return "inserted"

        return db._execute_write(_do)

    except sqlite3.Error as exc:
        logger.warning("cron-merge: sqlite error on session %s: %s", session_id, exc)
        return None
    except Exception as exc:
        logger.warning("cron-merge: unexpected error on session %s: %s", session_id, exc)
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _decode_text_content(content) -> str:
    """Multimodal content can be JSON-encoded list of parts.  Pull text out.

    Mirrors hermes_state.SessionDB._decode_content's intent without importing
    a private classmethod (keep this module standalone).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        # Could be raw text OR a JSON-encoded list.
        s = content.lstrip()
        if s.startswith("["):
            try:
                parts = json.loads(content)
                if isinstance(parts, list):
                    out = []
                    for p in parts:
                        if isinstance(p, dict):
                            if p.get("type") == "text" and "text" in p:
                                out.append(str(p["text"]))
                        elif isinstance(p, str):
                            out.append(p)
                    return "".join(out)
            except Exception:
                pass
        return content
    if isinstance(content, list):
        return "".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    return str(content)


def _reencode_like(original, new_text: str):
    """If original was a JSON-encoded multimodal list, return a re-encoded
    list with the appended text in a trailing text-part.  Otherwise return
    plain string."""
    if isinstance(original, str):
        s = original.lstrip()
        if s.startswith("["):
            try:
                parts = json.loads(original)
                if isinstance(parts, list):
                    parts.append({"type": "text", "text": new_text[len(_decode_text_content(original)):]})
                    return json.dumps(parts, ensure_ascii=False)
            except Exception:
                pass
    return new_text
