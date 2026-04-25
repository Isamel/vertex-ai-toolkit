"""Attachment-priors extractor for SPEC-ATT-10 §6.5.1.

Extracts an ``AttachmentPriors`` object from attached documents in a single
bounded LLM pass.  Results are cached in-process by attachment fingerprint so
subsequent runs with the same attachments skip the LLM call entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import AttachmentPriors

logger = logging.getLogger(__name__)

# In-process cache: fingerprint → AttachmentPriors
_PRIOR_CACHE: dict[str, AttachmentPriors] = {}

# Fingerprint length (hex chars, 64-bit width)
_FP_LENGTH = 16


def fingerprint(text: str) -> str:
    """Return a 16-hex-char SHA-256 fingerprint of *text*.

    Used to cache ``AttachmentPriors`` across runs with identical attachments
    (SPEC-ATT-10 §6.5.1 — "fully cached by attachment fingerprint").
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FP_LENGTH]


def get_cached(fp: str) -> AttachmentPriors | None:
    """Return cached ``AttachmentPriors`` for *fp*, or ``None`` if absent."""
    return _PRIOR_CACHE.get(fp)


def set_cached(fp: str, priors: AttachmentPriors) -> None:
    """Store *priors* in the in-process cache under *fp*."""
    _PRIOR_CACHE[fp] = priors


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

    The result is cached by the SHA-256 fingerprint of *attachment_text*.
    If the same attachment text was seen in a previous call within this
    process, the LLM call is skipped and the cached object is returned.

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

    fp = fingerprint(attachment_text)

    cached = get_cached(fp)
    if cached is not None:
        logger.debug("attachment_priors: cache hit for fingerprint %s — skipping LLM call", fp)
        return cached

    logger.debug("attachment_priors: cache miss for fingerprint %s — calling LLM", fp)

    user_prompt = build_user_prompt(attachment_text)

    kwargs: dict[str, Any] = {"system_instruction": SYSTEM_PROMPT}
    if model_id is not None:
        kwargs["model_id"] = model_id

    try:
        result = client.generate(user_prompt, **kwargs)
        raw_text: str = result.text if hasattr(result, "text") else str(result)
    except Exception as exc:
        logger.warning("attachment_priors: LLM call failed — returning empty priors: %s", exc)
        empty = AttachmentPriors()
        set_cached(fp, empty)
        return empty

    try:
        priors = parse_priors_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("attachment_priors: JSON parse failed — returning empty priors: %s", exc)
        priors = AttachmentPriors()

    set_cached(fp, priors)
    return priors
