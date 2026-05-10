"""Tests for tools/vision_smart.py routing logic."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from tools import vision_smart


# --------------------------------------------------------------------------- #
# Pure-function routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/tmp/screenshot_2024.png", "ocr"),
        ("/home/u/Pictures/Screen Shot 2024-01-01.png", "ocr"),
        ("/tmp/img_abc123def.png", "ocr"),
        ("/tmp/截屏-001.png", "ocr"),
        ("/tmp/random.png", "ocr"),  # any .png defaults to OCR
        ("/tmp/photo.jpg", "gemini"),
        ("/tmp/photo.JPEG", "gemini"),
        ("/tmp/photo.webp", "gemini"),
        ("https://example.com/x.png", "gemini"),  # URL → Gemini
        ("http://foo/y.jpg", "gemini"),
        ("", "gemini"),
        ("/tmp/no_extension", "gemini"),
    ],
)
def test_decide_route(url, expected):
    assert vision_smart.decide_route(url) == expected


# --------------------------------------------------------------------------- #
# OCR result acceptance
# --------------------------------------------------------------------------- #


def test_ocr_acceptable_high_confidence():
    ok, _ = vision_smart._ocr_result_is_acceptable(
        {"text": "Hello world this is a sentence", "avg_confidence": 0.9, "error": None},
        min_confidence=0.5,
        short_text_threshold=40,
    )
    assert ok


def test_ocr_rejected_low_conf_short_text():
    ok, reason = vision_smart._ocr_result_is_acceptable(
        {"text": "ab", "avg_confidence": 0.3, "error": None},
        min_confidence=0.5,
        short_text_threshold=40,
    )
    assert not ok
    assert "low_confidence" in reason


def test_ocr_rejected_empty_text():
    ok, reason = vision_smart._ocr_result_is_acceptable(
        {"text": "   ", "avg_confidence": 0.99, "error": None},
        min_confidence=0.5,
        short_text_threshold=40,
    )
    assert not ok
    assert reason == "ocr_empty_text"


def test_ocr_rejected_on_error():
    ok, reason = vision_smart._ocr_result_is_acceptable(
        {"text": "x", "avg_confidence": 0.9, "error": "boom"},
        min_confidence=0.5,
        short_text_threshold=40,
    )
    assert not ok
    assert "ocr_error" in reason


def test_ocr_low_conf_but_long_text_accepted():
    """Long text with low confidence is still useful — don't waste a Gemini call."""
    ok, _ = vision_smart._ocr_result_is_acceptable(
        {
            "text": "x" * 200,
            "avg_confidence": 0.3,
            "error": None,
        },
        min_confidence=0.5,
        short_text_threshold=40,
    )
    assert ok


# --------------------------------------------------------------------------- #
# End-to-end routing with mocked OCR + Gemini
# --------------------------------------------------------------------------- #


CFG_DEFAULTS = {
    "ocr_endpoint": "http://127.0.0.1:8765/ocr",
    "ocr_timeout": 5.0,
    "min_confidence": 0.5,
    "short_text_threshold": 40,
}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@patch.object(vision_smart, "_load_vision_cfg", return_value=CFG_DEFAULTS)
@patch.object(vision_smart, "_call_ocr")
def test_screenshot_routes_to_ocr(mock_ocr, _cfg):
    mock_ocr.return_value = {
        "text": "Hello world from the screenshot",
        "avg_confidence": 0.92,
        "line_count": 1,
        "error": None,
        "boxes": None,
    }
    result = asyncio.run(
        vision_smart.vision_smart_handler_impl("/tmp/screenshot.png", "")
    )
    assert result["source"] == "ocr"
    assert "Hello" in result["text"]
    assert result["confidence"] == pytest.approx(0.92)
    mock_ocr.assert_called_once()


@patch.object(vision_smart, "_load_vision_cfg", return_value=CFG_DEFAULTS)
@patch.object(vision_smart, "_gemini_fallback")
def test_jpg_routes_directly_to_gemini(mock_gemini, _cfg):
    async def fake(*_a, **_kw):
        return "A photo of a cat"
    mock_gemini.side_effect = fake

    result = asyncio.run(
        vision_smart.vision_smart_handler_impl("/tmp/photo.jpg", "what is this?")
    )
    assert result["source"] == "gemini"
    assert result["text"] == "A photo of a cat"
    assert result["details"]["route"] == "gemini_direct"


@patch.object(vision_smart, "_load_vision_cfg", return_value=CFG_DEFAULTS)
@patch.object(vision_smart, "_call_ocr")
@patch.object(vision_smart, "_gemini_fallback")
def test_low_confidence_ocr_falls_back_to_gemini(mock_gemini, mock_ocr, _cfg):
    mock_ocr.return_value = {
        "text": "??",
        "avg_confidence": 0.1,
        "line_count": 1,
        "error": None,
    }

    async def fake(*_a, **_kw):
        return "gemini description"
    mock_gemini.side_effect = fake

    result = asyncio.run(
        vision_smart.vision_smart_handler_impl("/tmp/screenshot.png", "")
    )
    assert result["source"] == "both"
    assert result["text"] == "gemini description"
    assert "ocr_text" in result["details"]
    assert "rejected_reason" in result["details"]["ocr_rejected_reason"] or \
        "low_confidence" in result["details"]["ocr_rejected_reason"]


@patch.object(vision_smart, "_load_vision_cfg", return_value=CFG_DEFAULTS)
@patch.object(vision_smart, "_call_ocr", side_effect=RuntimeError("connection refused"))
@patch.object(vision_smart, "_gemini_fallback")
def test_ocr_http_error_falls_back_to_gemini(mock_gemini, _ocr, _cfg):
    async def fake(*_a, **_kw):
        return "gemini desc"
    mock_gemini.side_effect = fake

    result = asyncio.run(
        vision_smart.vision_smart_handler_impl("/tmp/screenshot.png", "")
    )
    assert result["source"] == "gemini"
    assert "ocr_error" in result["details"]


@patch.object(
    vision_smart,
    "_load_vision_cfg",
    return_value={**CFG_DEFAULTS, "ocr_endpoint": ""},
)
@patch.object(vision_smart, "_gemini_fallback")
def test_no_ocr_endpoint_uses_gemini(mock_gemini, _cfg):
    async def fake(*_a, **_kw):
        return "g"
    mock_gemini.side_effect = fake

    result = asyncio.run(
        vision_smart.vision_smart_handler_impl("/tmp/screenshot.png", "")
    )
    assert result["source"] == "gemini"


@patch.object(vision_smart, "_load_vision_cfg", return_value=CFG_DEFAULTS)
@patch.object(vision_smart, "_gemini_fallback")
def test_url_always_gemini(mock_gemini, _cfg):
    async def fake(*_a, **_kw):
        return "g"
    mock_gemini.side_effect = fake

    result = asyncio.run(
        vision_smart.vision_smart_handler_impl("https://example.com/foo.png", "")
    )
    assert result["source"] == "gemini"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_vision_tool_is_registered():
    from tools.registry import registry

    entry = registry.get_entry("vision")
    assert entry is not None
    assert entry.toolset == "vision"
    assert entry.is_async is True
    schema = entry.schema
    assert schema["name"] == "vision"
    assert "image_url" in schema["parameters"]["properties"]


def test_handler_returns_json_string():
    from tools.registry import registry

    entry = registry.get_entry("vision")
    with patch.object(vision_smart, "_load_vision_cfg", return_value=CFG_DEFAULTS), \
         patch.object(vision_smart, "_gemini_fallback") as mock_g:
        async def fake(*_a, **_kw):
            return "hi"
        mock_g.side_effect = fake

        coro = entry.handler({"image_url": "/tmp/x.jpg"})
        out = asyncio.run(coro)
    parsed = json.loads(out)
    assert parsed["source"] == "gemini"
