"""JSON pre-cleaning utilities for LLM-generated output.

LLMs occasionally wrap their JSON responses in markdown code fences or add
conversational preamble/postamble around the actual JSON object.  This module
provides a best-effort cleaner that strips those artefacts before parsing.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Matches ```json ... ``` or ``` ... ``` fences (non-greedy, DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def clean_llm_json(raw: str) -> str:
    """Strip common LLM artefacts from *raw* to expose a clean JSON string.

    Cleaning steps (applied in order):

    1. Strip markdown code fences (```json...``` or plain ``...``).
    2. Discard any text before the first ``{`` and after the last ``}``.
    3. Validate the trimmed slice; if it is not valid JSON, use a
       stack-based boundary scan to find the actual closing brace instead
       of the naive ``rfind("}")`` which can match ``}`` inside strings or
       trailing garbage text.
    4. Attempt best-effort repair of truncated JSON (unclosed brackets/braces).

    The function always returns a string — on failure it returns the original
    *raw* value unchanged so the caller can still attempt ``json.loads`` and
    get a meaningful error.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        A cleaned string, hopefully containing only valid JSON.
    """
    if not raw or not raw.strip():
        return raw

    cleaned = raw

    # Step 1 — strip markdown code fences
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        logger.debug("clean_llm_json: stripped markdown code fence")
        cleaned = fence_match.group(1)

    # Step 2 — discard text before first '{' and after last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1:
        # No JSON object found at all — return as-is
        logger.debug("clean_llm_json: no '{' found, returning original")
        return raw

    if last_brace < first_brace:
        # Truncated JSON (no closing brace) — attempt repair below
        logger.debug("clean_llm_json: no '}' found after '{', attempting repair")
        cleaned = cleaned[first_brace:]
    else:
        if first_brace > 0 or last_brace < len(cleaned) - 1:
            logger.debug(
                "clean_llm_json: trimmed %d leading and %d trailing characters",
                first_brace,
                len(cleaned) - last_brace - 1,
            )
        candidate = cleaned[first_brace : last_brace + 1]

        # Step 3 — validate the rfind-trimmed candidate.
        # rfind('}') can match a '}' inside a string value or in trailing
        # garbage text.  If the candidate is already valid JSON we keep it;
        # otherwise fall back to the stack-based scanner to find the real
        # JSON boundary.
        try:
            json.loads(candidate)
            cleaned = candidate
        except (json.JSONDecodeError, ValueError):
            logger.debug(
                "clean_llm_json: rfind candidate invalid JSON — using stack scan"
            )
            cleaned = _extract_json_by_stack(cleaned[first_brace:])

    # Step 4 — best-effort repair of truncated JSON
    cleaned = _repair_truncated_json(cleaned)

    return cleaned


def _extract_json_by_stack(text: str) -> str:
    """Find the outermost complete JSON object boundary using a brace stack.

    Scans *text* (which must start at the first ``{``) character-by-character
    to track nested braces, skipping over string literals.  Returns the
    substring from the start up to (and including) the brace that closes the
    outermost object.

    If no complete object is found (truncated input), the full *text* is
    returned so :func:`_repair_truncated_json` can close it.

    Args:
        text: Input string starting at the first ``{``.

    Returns:
        The shortest prefix of *text* that forms a complete JSON object, or
        *text* unchanged if the object is incomplete.
    """
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]

    # Outermost object never closed — return full text for repair
    return text


def _repair_truncated_json(text: str) -> str:
    """Attempt to close any unclosed brackets and braces in *text*.

    This is a best-effort heuristic — it will not fix all truncation cases,
    but it handles the common case where the LLM stopped mid-array or
    mid-object.  Strings that are still open (odd number of unescaped quotes)
    are NOT repaired here — that would require a full parser.

    Args:
        text: A JSON string that may be truncated.

    Returns:
        The input string with any unclosed ``[`` / ``{`` closed in LIFO order.
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in ("{", "["):
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if not stack:
        return text

    # Close unclosed containers in reverse order
    closing_map = {"{": "}", "[": "]"}
    suffix = "".join(closing_map[c] for c in reversed(stack))
    logger.debug("clean_llm_json: appended repair suffix %r", suffix)
    return text + suffix
