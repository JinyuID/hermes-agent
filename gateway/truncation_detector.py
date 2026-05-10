"""Heuristic truncation detector for assistant responses.

Some upstream proxies (notably Microsoft Copilot's ``claude-opus-4.7-1m-internal``
SSE bridge) occasionally return ``finish_reason='stop'`` while the actual text
ends mid-sentence / mid-list-item. This module provides:

* :func:`is_likely_truncated` — pure heuristic, easy to unit-test.
* :func:`maybe_continue` — given a callback that invokes the agent, optionally
  request a continuation and return the stitched-together response.

Both pieces are gated behind ``gateway.truncation_detection.enabled`` in
``~/.hermes/config.yaml``. Default is **off** so the heuristic can never fire
without an explicit opt-in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Characters that look like a "natural" sentence/clause terminator. Mix of
# CJK, English, common closing brackets/quotes, and a few markdown closers.
_NATURAL_END_CHARS = set(
    "。！？!?…」』）】)]}>》〕〉"   # CJK + closing brackets
    ".;"                                       # English terminators (no comma)
    "\"'`"                                     # quote marks
    "_~"                                       # markdown emphasis closers (sans '*')
)

# Emoji-ish: anything in the supplementary planes commonly used for emoji,
# plus a few BMP symbol blocks. Cheap regex so we don't need the `emoji` dep.
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

# Trailing patterns that strongly suggest the model was about to keep going:
#   "- "                  (empty bullet)
#   "* "
#   "1. "                 (empty numbered item)
#   "**foo**"             (bold heading then nothing)
#   ":" / "：" / "—" / "-" / "、" / ","   (clause connector)
_DANGLING_TAIL_RE = re.compile(
    r"(?:"
    r"(?:^|\n|[。！？.!?]\s*)[-*+]\s*$"           # empty bullet (line-start or after sentence)
    r"|(?:^|\n|[。！？.!?]\s*)(?:[-*+]\s+)?\*\*[^*\n]+\*\*\s*$"  # bold heading (opt bullet) w/ no body
    r"|[:：,，、—\-]\s*$"                          # dangling connector
    r")",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TruncationConfig:
    enabled: bool = False
    max_continuations: int = 1
    log_triggers: bool = True
    min_length: int = 100

    @classmethod
    def from_yaml(cls, hermes_home: Optional[Path] = None) -> "TruncationConfig":
        """Read ``gateway.truncation_detection`` from ``~/.hermes/config.yaml``.

        Missing / malformed config returns the all-default (disabled) instance.
        """
        try:
            import yaml  # local import; yaml is already a hermes dep
            home = hermes_home or Path.home() / ".hermes"
            cfg_path = home / "config.yaml"
            if not cfg_path.exists():
                return cls()
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            block = ((data.get("gateway") or {}).get("truncation_detection")) or {}
            return cls(
                enabled=bool(block.get("enabled", False)),
                max_continuations=int(block.get("max_continuations", 1)),
                log_triggers=bool(block.get("log_triggers", True)),
                min_length=int(block.get("min_length", 100)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("TruncationConfig.from_yaml fallback: %s", exc)
            return cls()


def _ends_naturally(text: str) -> bool:
    """Return True when the final non-whitespace char looks like a real ending."""
    stripped = text.rstrip()
    if not stripped:
        return True
    last = stripped[-1]
    if last in _NATURAL_END_CHARS:
        return True
    # Treat `**` (bold close) as a natural ending — common closer for the
    # last word in a sentence, e.g. "...wrapped in **bold**".
    if last == "*" and len(stripped) >= 2 and stripped[-2] == "*":
        return True
    if _EMOJI_RE.search(last):
        return True
    return False


def is_likely_truncated(
    content: str,
    finish_reason: Optional[str],
    *,
    min_length: int = 100,
) -> bool:
    """Heuristically decide whether ``content`` looks mid-sentence.

    Triggers only when:
      * ``finish_reason == 'stop'``  (i.e. provider claims completion)
      * content is longer than ``min_length`` chars
      * the tail is *not* a natural terminator AND/OR matches a dangling pattern
    """
    if not content or finish_reason != "stop":
        return False
    if len(content) <= min_length:
        return False

    tail = content[-200:]  # cheap windowed scan
    if _DANGLING_TAIL_RE.search(tail):
        return True
    if not _ends_naturally(content):
        return True
    return False


CONTINUATION_PROMPT = (
    "你上一条回复看起来在中途被截断了。请直接从你上次停下的地方自然续写下去，"
    "不要重复任何已经说过的内容，也不要重新打招呼或重述前文。"
    "如果你认为上一条其实已经讲完了，就只回复一个英文句号 `.` 即可。"
)


def maybe_continue(
    response_text: str,
    finish_reason: Optional[str],
    *,
    config: TruncationConfig,
    continue_fn: Callable[[str], str],
) -> str:
    """Stitch a continuation onto ``response_text`` if it looks truncated.

    ``continue_fn`` is a thin closure the caller supplies — it receives the
    continuation prompt and must return the agent's follow-up text (or ``""``
    if the agent declined / errored). This keeps the detector decoupled from
    the gateway's specific ``run_conversation`` plumbing.
    """
    if not config.enabled:
        return response_text

    text = response_text or ""
    triggered = 0
    while triggered < max(1, config.max_continuations):
        if not is_likely_truncated(text, finish_reason, min_length=config.min_length):
            break
        if config.log_triggers:
            tail_snippet = text[-80:].replace("\n", "\\n")
            logger.warning(
                "truncation detected, attempting continuation "
                "(orig_len=%d, finish_reason=%s, tail=%r)",
                len(text), finish_reason, tail_snippet,
            )
        try:
            extra = continue_fn(CONTINUATION_PROMPT) or ""
        except Exception as exc:
            logger.warning("Continuation call failed: %s", exc)
            break
        extra = extra.strip()
        # Model says "nothing more to add" — bail out cleanly.
        if extra in {".", "。", ""}:
            break
        # Pick a join string: blank if the previous tail already ended a line.
        joiner = "" if text.endswith(("\n", " ")) else ""
        text = text + joiner + extra
        triggered += 1
        # After a successful continuation, treat finish_reason as 'stop' for
        # the loop condition (continue_fn may not give us a new one).
        finish_reason = "stop"
    return text
