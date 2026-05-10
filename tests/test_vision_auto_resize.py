"""Tests for vision_smart auto-resize preprocessing."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import patch, MagicMock

import pytest

from tools import vision_smart


# --------------------------------------------------------------------------- #
# _maybe_resize unit tests
# --------------------------------------------------------------------------- #


def test_maybe_resize_small_file_noop(tmp_path):
    """Small file under both limits → return original path unchanged."""
    p = tmp_path / "tiny.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    with patch.object(vision_smart, "_get_image_dimensions", return_value=(800, 600)):
        out = vision_smart._maybe_resize(str(p), max_bytes=5_000_000, max_pixels=2048)

    assert out == str(p)


def test_maybe_resize_oversized_invokes_ffmpeg(tmp_path):
    """File > max_bytes triggers ffmpeg call and returns cached path."""
    p = tmp_path / "big.jpg"
    p.write_bytes(b"x" * (6 * 1024 * 1024))  # 6MB

    fake_run = MagicMock()
    def _fake_run(cmd, **kw):
        # simulate ffmpeg writing the output file
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"resized-jpg-bytes" * 100)
        return MagicMock(returncode=0)
    fake_run.side_effect = _fake_run

    with patch.object(vision_smart, "_get_image_dimensions", return_value=(4000, 3000)), \
         patch.object(vision_smart.shutil, "which", return_value="/usr/bin/ffmpeg"), \
         patch.object(vision_smart.subprocess, "run", fake_run):
        out = vision_smart._maybe_resize(str(p), max_bytes=5_000_000, max_pixels=2048)

    assert out != str(p)
    assert out.startswith(vision_smart.RESIZE_CACHE_DIR)
    assert out.endswith(".jpg")
    assert os.path.isfile(out)
    fake_run.assert_called_once()
    cmd = fake_run.call_args.args[0]
    assert cmd[0] == "ffmpeg"
    assert "-q:v" in cmd


def test_maybe_resize_oversized_pixels_only(tmp_path):
    """File small bytes but huge pixels still triggers resize."""
    p = tmp_path / "wide.png"
    p.write_bytes(b"x" * 1024)  # only 1KB

    def _fake_run(cmd, **kw):
        with open(cmd[-1], "wb") as f:
            f.write(b"y" * 50)
        return MagicMock(returncode=0)

    with patch.object(vision_smart, "_get_image_dimensions", return_value=(5000, 1000)), \
         patch.object(vision_smart.shutil, "which", return_value="/usr/bin/ffmpeg"), \
         patch.object(vision_smart.subprocess, "run", side_effect=_fake_run) as run_mock:
        out = vision_smart._maybe_resize(str(p), max_bytes=5_000_000, max_pixels=2048)

    assert out != str(p)
    run_mock.assert_called_once()


def test_maybe_resize_no_ffmpeg_fallback(tmp_path):
    """ffmpeg missing → log warning + return original path."""
    p = tmp_path / "big.jpg"
    p.write_bytes(b"x" * (6 * 1024 * 1024))

    with patch.object(vision_smart, "_get_image_dimensions", return_value=(4000, 3000)), \
         patch.object(vision_smart.shutil, "which", return_value=None):
        out = vision_smart._maybe_resize(str(p), max_bytes=5_000_000, max_pixels=2048)
    assert out == str(p)


def test_maybe_resize_missing_file():
    assert vision_smart._maybe_resize("/nonexistent/foo.png") == "/nonexistent/foo.png"


def test_maybe_resize_ffmpeg_failure_returns_original(tmp_path):
    p = tmp_path / "big.jpg"
    p.write_bytes(b"x" * (6 * 1024 * 1024))

    import subprocess as real_sp
    with patch.object(vision_smart, "_get_image_dimensions", return_value=(4000, 3000)), \
         patch.object(vision_smart.shutil, "which", return_value="/usr/bin/ffmpeg"), \
         patch.object(vision_smart.subprocess, "run",
                      side_effect=real_sp.CalledProcessError(1, "ffmpeg")):
        out = vision_smart._maybe_resize(str(p), max_bytes=5_000_000, max_pixels=2048)
    assert out == str(p)


# --------------------------------------------------------------------------- #
# End-to-end: handler swaps path before Gemini fallback
# --------------------------------------------------------------------------- #


def test_handler_calls_resize_and_passes_new_path(tmp_path):
    """vision_smart_handler_impl must replace local path with resized one before Gemini."""
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"x" * (6 * 1024 * 1024))
    resized = "/tmp/hermes_vision_resized/fake.jpg"

    seen = {}

    async def fake_gemini(url, question):
        seen["url"] = url
        return "describe-result"

    fake_cfg = {
        "ocr_endpoint": "",  # disabled → Gemini direct
        "ocr_timeout": 5.0,
        "min_confidence": 0.5,
        "short_text_threshold": 40,
        "auto_resize": True,
        "max_bytes": 5_000_000,
        "max_pixels": 2048,
    }

    with patch.object(vision_smart, "_load_vision_cfg", return_value=fake_cfg), \
         patch.object(vision_smart, "_maybe_resize", return_value=resized) as resize_mock, \
         patch.object(vision_smart, "_gemini_fallback", side_effect=fake_gemini):
        result = asyncio.run(
            vision_smart.vision_smart_handler_impl(str(p), "what is this?")
        )

    resize_mock.assert_called_once()
    assert seen["url"] == resized
    assert result["details"].get("resized_to") == resized
    assert result["details"].get("resized_from") == str(p)


def test_handler_skips_resize_for_url():
    """URLs must NOT be touched by the resizer."""
    seen = {}

    async def fake_gemini(url, question):
        seen["url"] = url
        return "ok"

    fake_cfg = {
        "ocr_endpoint": "",
        "ocr_timeout": 5.0,
        "min_confidence": 0.5,
        "short_text_threshold": 40,
        "auto_resize": True,
        "max_bytes": 5_000_000,
        "max_pixels": 2048,
    }

    with patch.object(vision_smart, "_load_vision_cfg", return_value=fake_cfg), \
         patch.object(vision_smart, "_maybe_resize") as resize_mock, \
         patch.object(vision_smart, "_gemini_fallback", side_effect=fake_gemini):
        result = asyncio.run(
            vision_smart.vision_smart_handler_impl("https://x.com/a.jpg", "")
        )

    resize_mock.assert_not_called()
    assert seen["url"] == "https://x.com/a.jpg"
    assert "resized_to" not in result["details"]


def test_handler_disabled_via_config(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"x" * 100)

    async def fake_gemini(url, question):
        return "ok"

    fake_cfg = {
        "ocr_endpoint": "",
        "ocr_timeout": 5.0,
        "min_confidence": 0.5,
        "short_text_threshold": 40,
        "auto_resize": False,
        "max_bytes": 5_000_000,
        "max_pixels": 2048,
    }

    with patch.object(vision_smart, "_load_vision_cfg", return_value=fake_cfg), \
         patch.object(vision_smart, "_maybe_resize") as resize_mock, \
         patch.object(vision_smart, "_gemini_fallback", side_effect=fake_gemini):
        asyncio.run(vision_smart.vision_smart_handler_impl(str(p), ""))

    resize_mock.assert_not_called()
