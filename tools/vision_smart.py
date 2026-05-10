"""Smart vision tool: routes screenshots → local OCR, photos → Gemini.

Wraps the existing ``vision_analyze`` primitive so the main model only has
to call a single tool (``vision``) without deciding between OCR and a
multimodal LLM.

Routing rules (mirrors the ``creative/vision-smart-routing`` skill):

1. http(s) URL → Gemini (local OCR only accepts file paths).
2. Local ``.png`` OR path containing ``screenshot|截屏|Screen Shot|img_<hash>.png``
   → try OCR first.
3. Local ``.jpg``/``.jpeg`` (or anything else) → Gemini directly.
4. OCR fallback to Gemini if: HTTP failure / timeout / empty text /
   ``avg_confidence < min_confidence`` AND text is short.

OCR endpoint is configurable via ``config.yaml`` → ``tools.vision`` block:

    tools:
      vision:
        ocr_endpoint: "http://127.0.0.1:8765/ocr"
        ocr_timeout: 5.0
        min_confidence: 0.5
        short_text_threshold: 40

If ``ocr_endpoint`` is empty/missing, OCR is skipped and everything goes
to Gemini (graceful degradation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Dict, Optional, Tuple
from urllib.parse import urlparse

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_OCR_ENDPOINT = "http://127.0.0.1:8765/ocr"
DEFAULT_OCR_TIMEOUT = 5.0
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_SHORT_TEXT_THRESHOLD = 40
DEFAULT_AUTO_RESIZE = True
DEFAULT_RESIZE_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_RESIZE_MAX_PIXELS = 2048
RESIZE_CACHE_DIR = "/tmp/hermes_vision_resized"

# Path patterns that strongly suggest "this is a screenshot" rather than a
# user-supplied photo.  Case-insensitive.
_SCREENSHOT_HINT_RE = re.compile(
    r"(screenshot|screen[\s_-]?shot|截屏|截图|img_[0-9a-f]{6,})",
    re.IGNORECASE,
)


def _load_vision_cfg() -> Dict[str, Any]:
    """Load ``tools.vision`` block from user config, with sane defaults.

    Never raises — falls back to defaults on any error.
    """
    cfg: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        full = load_config() or {}
        cfg = ((full.get("tools") or {}).get("vision") or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("vision_smart: load_config failed: %s", exc)
        cfg = {}

    return {
        "ocr_endpoint": cfg.get("ocr_endpoint", DEFAULT_OCR_ENDPOINT) or "",
        "ocr_timeout": float(cfg.get("ocr_timeout", DEFAULT_OCR_TIMEOUT)),
        "min_confidence": float(cfg.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        "short_text_threshold": int(
            cfg.get("short_text_threshold", DEFAULT_SHORT_TEXT_THRESHOLD)
        ),
        "auto_resize": bool(cfg.get("auto_resize", DEFAULT_AUTO_RESIZE)),
        "max_bytes": int(cfg.get("max_bytes", DEFAULT_RESIZE_MAX_BYTES)),
        "max_pixels": int(cfg.get("max_pixels", DEFAULT_RESIZE_MAX_PIXELS)),
    }


# --------------------------------------------------------------------------- #
# Auto-resize preprocessing
# --------------------------------------------------------------------------- #


def _get_image_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) using Pillow if available, else ffprobe.

    Returns None on failure.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x", path,
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode().strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception as exc:
        logger.debug("vision_smart: ffprobe failed for %s: %s", path, exc)
        return None


def _maybe_resize(
    image_path: str,
    max_bytes: int = DEFAULT_RESIZE_MAX_BYTES,
    max_pixels: int = DEFAULT_RESIZE_MAX_PIXELS,
) -> str:
    """If image exceeds size/pixel limits, ffmpeg-compress to a cached jpg.

    Returns the path to use (resized cache path or original on no-op /
    failure). Never raises.
    """
    try:
        if not image_path or not os.path.isfile(image_path):
            return image_path
        size = os.path.getsize(image_path)
        dims = _get_image_dimensions(image_path)
        max_dim = max(dims) if dims else 0
        if size <= max_bytes and (max_dim == 0 or max_dim <= max_pixels):
            return image_path

        if not shutil.which("ffmpeg"):
            logger.warning(
                "vision_smart: ffmpeg not found; skipping auto-resize for %s",
                image_path,
            )
            return image_path

        os.makedirs(RESIZE_CACHE_DIR, exist_ok=True)
        # Cache key includes path + mtime + size so updates bust the cache.
        try:
            mtime = os.path.getmtime(image_path)
        except OSError:
            mtime = 0
        key = hashlib.sha1(
            f"{image_path}|{mtime}|{size}|{max_pixels}".encode("utf-8")
        ).hexdigest()
        out_path = os.path.join(RESIZE_CACHE_DIR, f"{key}.jpg")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            logger.debug("vision_smart: using cached resize %s", out_path)
            return out_path

        # Scale longer side to max_pixels, preserve aspect, q:v 5 (~mid-quality).
        vf = f"scale='if(gt(iw,ih),{max_pixels},-2)':'if(gt(iw,ih),-2,{max_pixels})'"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", image_path,
            "-vf", vf,
            "-q:v", "5",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "vision_smart: ffmpeg resize failed for %s: %s; using original",
                image_path, exc,
            )
            return image_path

        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            return image_path
        logger.info(
            "vision_smart: resized %s (%d bytes, %s) -> %s (%d bytes)",
            image_path, size, dims, out_path, os.path.getsize(out_path),
        )
        return out_path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("vision_smart: _maybe_resize crashed: %s", exc)
        return image_path


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False


def decide_route(image_url: str) -> str:
    """Return ``"ocr"`` if we should try OCR first, else ``"gemini"``.

    Pure function — no I/O, easy to unit-test.
    """
    if not image_url:
        return "gemini"
    if _is_url(image_url):
        # OCR service only accepts local paths.
        return "gemini"

    lower = image_url.lower()
    ext = Path(lower).suffix
    if ext == ".png":
        return "ocr"
    if _SCREENSHOT_HINT_RE.search(image_url):
        return "ocr"
    if ext in (".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"):
        return "gemini"
    # Unknown / no extension → Gemini is the safer general-purpose fallback.
    return "gemini"


# --------------------------------------------------------------------------- #
# OCR call
# --------------------------------------------------------------------------- #


def _call_ocr(image_path: str, endpoint: str, timeout: float) -> Dict[str, Any]:
    """POST to local OCR service.  Returns parsed dict, raises on failure."""
    import httpx  # lazy import — httpx is already a hermes-agent dep

    resp = httpx.post(
        endpoint,
        json={"image_path": image_path},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _ocr_result_is_acceptable(
    ocr: Dict[str, Any], min_confidence: float, short_text_threshold: int
) -> Tuple[bool, str]:
    """Decide if OCR output is good enough to skip Gemini fallback.

    Returns ``(ok, reason)``.  ``reason`` describes the rejection when
    ``ok`` is False (used for source/details so the main model can see
    why we fell back).
    """
    if ocr.get("error"):
        return False, f"ocr_error: {ocr['error']}"
    text = (ocr.get("text") or "").strip()
    conf = float(ocr.get("avg_confidence") or 0.0)
    if not text:
        return False, "ocr_empty_text"
    if conf < min_confidence and len(text) < short_text_threshold:
        return False, f"ocr_low_confidence({conf:.2f}<{min_confidence})"
    return True, ""


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #


async def _gemini_fallback(image_url: str, question: str) -> str:
    """Delegate to the existing ``vision_analyze`` handler.

    Returns its raw string result (already JSON or plain text envelope).
    """
    # Reuse the registered handler so we automatically pick up the
    # native-vision fast path when the main model supports images.
    entry = registry.get_entry("vision_analyze")
    if entry is None:
        return tool_error(
            "vision_analyze tool unavailable; cannot route image to Gemini",
            tool="vision",
        )
    args = {"image_url": image_url, "question": question or "Describe this image."}
    result = entry.handler(args)
    if hasattr(result, "__await__"):
        return await result
    return result  # type: ignore[return-value]


async def vision_smart_handler_impl(
    image_url: str, question: str = ""
) -> Dict[str, Any]:
    """Core routing logic, returning a structured dict.

    Public-ish for unit tests — the registered handler wraps this and
    JSON-encodes the result.
    """
    cfg = _load_vision_cfg()
    route = decide_route(image_url)
    endpoint = cfg["ocr_endpoint"]

    # Auto-resize local oversized images before any downstream call.
    # URLs are passed through untouched (existing vision_analyze handles them).
    effective_url = image_url
    resized_from: Optional[str] = None
    if cfg.get("auto_resize", DEFAULT_AUTO_RESIZE) and image_url and not _is_url(image_url):
        try:
            abs_in = str(Path(image_url).expanduser().resolve(strict=False))
        except Exception:
            abs_in = image_url
        new_path = _maybe_resize(
            abs_in,
            max_bytes=cfg.get("max_bytes", DEFAULT_RESIZE_MAX_BYTES),
            max_pixels=cfg.get("max_pixels", DEFAULT_RESIZE_MAX_PIXELS),
        )
        if new_path != abs_in:
            effective_url = new_path
            resized_from = image_url

    # If routing says OCR but OCR isn't configured, skip straight to Gemini.
    if route == "ocr" and not endpoint:
        logger.debug("vision_smart: OCR endpoint not configured; using Gemini")
        route = "gemini"

    if route == "ocr":
        # Resolve to absolute path — OCR service expects an absolute local path.
        try:
            abs_path = str(Path(effective_url).expanduser().resolve(strict=False))
        except Exception:
            abs_path = effective_url

        try:
            ocr = _call_ocr(abs_path, endpoint, cfg["ocr_timeout"])
        except Exception as exc:
            logger.info("vision_smart: OCR failed (%s); falling back to Gemini", exc)
            gemini_text = await _gemini_fallback(effective_url, question)
            return {
                "text": gemini_text,
                "source": "gemini",
                "confidence": None,
                "details": {
                    "ocr_error": str(exc),
                    "route": "ocr->gemini_fallback",
                    **({"resized_from": resized_from, "resized_to": effective_url} if resized_from else {}),
                },
            }

        ok, reason = _ocr_result_is_acceptable(
            ocr, cfg["min_confidence"], cfg["short_text_threshold"]
        )
        if ok:
            return {
                "text": ocr.get("text", ""),
                "source": "ocr",
                "confidence": float(ocr.get("avg_confidence") or 0.0),
                "details": {
                    "line_count": ocr.get("line_count"),
                    "endpoint": endpoint,
                    **({"resized_from": resized_from, "resized_to": effective_url} if resized_from else {}),
                },
            }

        # Low-confidence / empty → fallback to Gemini, return both for visibility.
        logger.info("vision_smart: OCR rejected (%s); fallback Gemini", reason)
        gemini_text = await _gemini_fallback(effective_url, question)
        return {
            "text": gemini_text,
            "source": "both",
            "confidence": float(ocr.get("avg_confidence") or 0.0),
            "details": {
                "ocr_text": ocr.get("text", ""),
                "ocr_rejected_reason": reason,
                "route": "ocr->gemini_fallback",
                **({"resized_from": resized_from, "resized_to": effective_url} if resized_from else {}),
            },
        }

    # Direct Gemini route.
    gemini_text = await _gemini_fallback(effective_url, question)
    return {
        "text": gemini_text,
        "source": "gemini",
        "confidence": None,
        "details": {
            "route": "gemini_direct",
            **({"resized_from": resized_from, "resized_to": effective_url} if resized_from else {}),
        },
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

VISION_SMART_SCHEMA = {
    "name": "vision",
    "description": (
        "Smart image analysis — your DEFAULT tool for any image. Automatically "
        "routes screenshots / .png files to a fast local OCR service and "
        "photos / .jpg files to Gemini multimodal vision, with automatic "
        "fallback to Gemini if OCR quality is poor. Accepts a local file path "
        "or http(s) URL. Returns {text, source, confidence, details}. "
        "Use the lower-level `vision_analyze` only when you explicitly need "
        "multimodal reasoning over the pixels (e.g. visual Q&A, diagram "
        "understanding); for plain text extraction or general descriptions, "
        "use this `vision` tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": (
                    "Local absolute path or http(s) URL of the image. "
                    "Local paths are required for OCR routing; URLs always "
                    "go to Gemini."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "Optional question or context. Only used when the image "
                    "is routed to Gemini (ignored for pure OCR)."
                ),
            },
        },
        "required": ["image_url"],
    },
}


def _handle_vision_smart(args: Dict[str, Any], **_kw: Any) -> Awaitable[str]:
    image_url = args.get("image_url", "")
    question = args.get("question", "") or ""

    async def _run() -> str:
        if not image_url:
            return tool_error("vision: 'image_url' is required", tool="vision")
        try:
            result = await vision_smart_handler_impl(image_url, question)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.exception("vision_smart handler crashed")
            return tool_error(f"vision routing failed: {exc}", tool="vision")

    return _run()


def _check_vision_smart() -> bool:
    """Always available — it gracefully falls back when OCR is down."""
    return True


registry.register(
    name="vision",
    toolset="vision",
    schema=VISION_SMART_SCHEMA,
    handler=_handle_vision_smart,
    check_fn=_check_vision_smart,
    is_async=True,
    emoji="🔭",
)
