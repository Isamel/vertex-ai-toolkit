"""Shared utilities for agent modules."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def deduplicate_response(text: str, *, threshold: int = 3) -> str:
    """Remove repeated sentences/lines from model output.

    Gemini can sometimes produce pathological repetition — the same
    sentence hundreds of times in a single response, especially at low
    temperature with high ``max_output_tokens``.  This function acts as
    a safety net: it scans the text line by line and truncates once a
    line has been seen more than *threshold* consecutive times.

    The algorithm is intentionally conservative:
    - It only counts **consecutive** repetitions (not scattered ones).
    - Short lines (≤ 10 chars) are ignored to avoid false positives on
      blank lines, bullets, braces, etc.
    - A ``[truncated — repeated text removed]`` marker is appended when
      truncation occurs so the user knows something was cut.

    Args:
        text: The raw model response text.
        threshold: How many consecutive identical lines to allow before
                   truncating.  Default is 3 (keeps first 3 occurrences).

    Returns:
        The cleaned text, possibly truncated.
    """
    if not text:
        return text

    lines = text.split("\n")
    result: list[str] = []
    prev_line: str | None = None
    repeat_count = 0
    truncated = False

    for line in lines:
        stripped = line.strip()

        # Skip short-line tracking — too many false positives
        if len(stripped) <= 10:
            result.append(line)
            prev_line = None
            repeat_count = 0
            continue

        if stripped == prev_line:
            repeat_count += 1
            if repeat_count > threshold:
                truncated = True
                continue  # Drop this repeated line
            result.append(line)
        else:
            prev_line = stripped
            repeat_count = 1
            result.append(line)

    cleaned = "\n".join(result)
    if truncated:
        cleaned = cleaned.rstrip() + "\n\n[truncated — repeated text removed]"
        logger.warning(
            "Deduplicated model response — removed repeated lines "
            "(threshold=%d)",
            threshold,
        )
    return cleaned
