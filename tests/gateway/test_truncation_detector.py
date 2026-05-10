"""Unit tests for gateway.truncation_detector heuristic + stitching."""

from __future__ import annotations

import pytest

from gateway.truncation_detector import (
    CONTINUATION_PROMPT,
    TruncationConfig,
    is_likely_truncated,
    maybe_continue,
)


# ---- is_likely_truncated --------------------------------------------------

LONG = "这是一段足够长的占位文字。" * 20  # ~ 240 chars, ends with 。

@pytest.mark.parametrize(
    "tail",
    [
        "4. 他们自己想留还是想走？——但他",     # observed id=4421
        "70多的人，每",                         # observed id=4431 (comma tail)
        "- **工具路由**",                       # observed id=4688 (bold heading, dangling)
        "首先我们来看：",                       # CJK colon connector
        "Here's the thing -",                  # English dash connector
        "important note —",                    # em-dash connector
        "* ",                                  # empty bullet
        "**Highlights**",                      # bold heading w/ nothing after
    ],
)
def test_truncated_tails_trigger(tail):
    text = LONG + tail
    assert is_likely_truncated(text, "stop") is True


@pytest.mark.parametrize(
    "tail",
    [
        "好的，就这样。",      # CJK period
        "All done!",            # English bang
        "Ship it. 🚀",          # emoji
        "结束」",               # CJK closing quote
        "wrapped in **bold**",  # natural bold close
    ],
)
def test_natural_endings_do_not_trigger(tail):
    text = LONG + tail
    assert is_likely_truncated(text, "stop") is False


def test_non_stop_finish_never_triggers():
    text = LONG + "- "
    assert is_likely_truncated(text, "length") is False
    assert is_likely_truncated(text, None) is False


def test_short_responses_skipped():
    # Below min_length, even a clearly-dangling tail is ignored —
    # could be an intentionally short reply ("好的，") and false positives are bad.
    assert is_likely_truncated("好的，", "stop") is False


def test_empty_content():
    assert is_likely_truncated("", "stop") is False


# ---- maybe_continue stitching --------------------------------------------

def test_disabled_config_is_noop():
    cfg = TruncationConfig(enabled=False)
    text = LONG + "- "
    out = maybe_continue(text, "stop", config=cfg, continue_fn=lambda p: "FAIL")
    assert out == text


def test_continuation_appends_when_triggered():
    cfg = TruncationConfig(enabled=True, max_continuations=1)
    seen = {}
    def fake(prompt):
        seen["prompt"] = prompt
        return "续写部分。"
    text = LONG + "- "
    out = maybe_continue(text, "stop", config=cfg, continue_fn=fake)
    assert out.endswith("续写部分。")
    assert seen["prompt"] == CONTINUATION_PROMPT


def test_continuation_skipped_when_natural_ending():
    cfg = TruncationConfig(enabled=True)
    text = LONG + "完成。"
    out = maybe_continue(text, "stop", config=cfg, continue_fn=lambda p: "BAD")
    assert out == text


def test_model_signals_done_with_period():
    cfg = TruncationConfig(enabled=True, max_continuations=3)
    text = LONG + "- "
    calls = {"n": 0}
    def fake(_):
        calls["n"] += 1
        return "."  # "nothing more to add"
    out = maybe_continue(text, "stop", config=cfg, continue_fn=fake)
    assert calls["n"] == 1   # bailed out after the period
    assert out == text


def test_max_continuations_respected():
    cfg = TruncationConfig(enabled=True, max_continuations=2)
    calls = {"n": 0}
    def fake(_):
        calls["n"] += 1
        return "更多内容，"   # itself triggers another round
    text = LONG + "- "
    maybe_continue(text, "stop", config=cfg, continue_fn=fake)
    assert calls["n"] == 2


def test_continue_fn_exception_swallowed():
    cfg = TruncationConfig(enabled=True)
    text = LONG + "- "
    def boom(_):
        raise RuntimeError("network down")
    out = maybe_continue(text, "stop", config=cfg, continue_fn=boom)
    assert out == text  # falls back to original
