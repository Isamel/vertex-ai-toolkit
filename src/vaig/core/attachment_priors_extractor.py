"""Attachment-priors extractor for SPEC-ATT-10 §6.5.1.

Extracts an ``AttachmentPriors`` object from attached documents in a single
bounded LLM pass.  Results are cached in-process by a composite key that
covers the attachment text, the system prompt, and the model ID so that prompt
or model changes always produce a fresh extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import AttachmentPriors

logger = logging.getLogger(__name__)

# Bounded in-process LRU cache: composite_key → AttachmentPriors.
# Evicts oldest entry once the limit is reached to prevent unbounded growth.
_CACHE_MAX_SIZE = 64
_PRIOR_CACHE: OrderedDict[str, AttachmentPriors] = OrderedDict()

# Fingerprint length (hex chars, 64-bit width)
_FP_LENGTH = 16


def fingerprint(text: str) -> str:
    """Return a 16-hex-char SHA-256 fingerprint of *text*.

    Used as a component of the composite cache key for ``AttachmentPriors``.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FP_LENGTH]


def _cache_key(attachment_text: str, system_prompt: str, model_id: str) -> str:
    """Return a composite cache key covering text, prompt, and model.

    Combining all three ensures that changes to the prompt or model always
    produce a fresh LLM extraction even for identical attachment text.
    """
    combined = f"{system_prompt}||{model_id}||{attachment_text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[: _FP_LENGTH * 2]


def get_cached(key: str) -> AttachmentPriors | None:
    """Return cached ``AttachmentPriors`` for *key*, or ``None`` if absent.

    Moves the entry to the end (most-recently-used position) on a hit.
    """
    if key not in _PRIOR_CACHE:
        return None
    _PRIOR_CACHE.move_to_end(key)
    return _PRIOR_CACHE[key]


def set_cached(key: str, priors: AttachmentPriors) -> None:
    """Store *priors* under *key*, evicting the oldest entry when at capacity."""
    if key in _PRIOR_CACHE:
        _PRIOR_CACHE.move_to_end(key)
    else:
        if len(_PRIOR_CACHE) >= _CACHE_MAX_SIZE:
            _PRIOR_CACHE.popitem(last=False)
    _PRIOR_CACHE[key] = priors


def clear_cache() -> None:
    """Clear the in-process cache (test helper)."""
    _PRIOR_CACHE.clear()


def parse_priors_json(raw: str) -> AttachmentPriors:
    """Parse *raw* JSON string into an ``AttachmentPriors`` model.

    Strips optional markdown code fences that some models emit despite
    the prompt instructing plain JSON output.

    Raises
    ------
    ValueError
        When *raw* cannot be parsed as valid JSON or does not satisfy the
        ``AttachmentPriors`` schema.
    """
    from vaig.skills.service_health.schema import AttachmentPriors

    cleaned = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove opening fence line (```json or ```)
        lines = lines[1:]
        # Remove trailing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    data: Any = json.loads(cleaned)
    return AttachmentPriors.model_validate(data)


def extract_priors(
    attachment_text: str,
    client: Any,
    *,
    model_id: str | None = None,
) -> AttachmentPriors:
    """Extract ``AttachmentPriors`` from *attachment_text* using *client*.

    Results are cached by a composite key covering the attachment text, the
    system prompt, and the model ID.  Changing any of these three inputs
    bypasses the cache and triggers a fresh LLM call.

    This is a **post-hoc enrichment** pass: it runs after the main
    ``execute_with_tools`` fan-out and populates
    ``OrchestratorResult.attachment_priors`` / ``HealthReport.attachment_priors``
    for downstream consumers (reporters, CLI output).  Sub-gatherers in the
    current sprint do *not* receive these priors during their execution; that
    wiring is deferred to §6.5.2.

    Parameters
    ----------
    attachment_text:
        Concatenated rendered attachment context (already chunked/truncated).
    client:
        A ``GeminiClient`` instance (or any object with a compatible
        ``.generate(prompt, *, system_instruction)`` method).
    model_id:
        Optional model override for the extraction call.  When ``None`` the
        client's current model is used.
    """
    from vaig.skills.service_health.prompts._attachment_gatherer import (
        SYSTEM_PROMPT,
        build_user_prompt,
    )
    from vaig.skills.service_health.schema import AttachmentPriors

    _model_id = model_id or ""
    key = _cache_key(attachment_text, SYSTEM_PROMPT, _model_id)

    cached = get_cached(key)
    if cached is not None:
        logger.debug("attachment_priors: cache hit for key %s — skipping LLM call", key[:8])
        return cached

    logger.debug("attachment_priors: cache miss for key %s — calling LLM", key[:8])

    user_prompt = build_user_prompt(attachment_text)

    kwargs: dict[str, Any] = {"system_instruction": SYSTEM_PROMPT}
    if model_id is not None:
        kwargs["model_id"] = model_id

    try:
        result = client.generate(user_prompt, **kwargs)
        raw_text: str = result.text if hasattr(result, "text") else str(result)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        logger.warning("attachment_priors: LLM call failed — returning empty priors: %s", exc)
        empty = AttachmentPriors()
        set_cached(key, empty)
        return empty

    try:
        priors = parse_priors_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("attachment_priors: JSON parse failed — returning empty priors: %s", exc)
        priors = AttachmentPriors()

    set_cached(key, priors)
    return priors
