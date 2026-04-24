from __future__ import annotations

"""Unit tests for the shared attachment-context prefix helper."""

from vaig.skills.service_health.prompts._shared import (
    ATTACHMENT_HEADER,
    _prefix_attachment_context,
)


def test_identity_when_none() -> None:
    """context=None → returns system_instruction unchanged."""
    instruction = "You are a helpful SRE agent."
    result = _prefix_attachment_context(instruction, None)
    assert result == instruction


def test_prefixes_when_provided() -> None:
    """context='X' → result starts with header + context and contains instruction."""
    instruction = "You are a helpful SRE agent."
    result = _prefix_attachment_context(instruction, "X")
    assert result.startswith(f"{ATTACHMENT_HEADER}X\n\n")
    assert instruction in result


def test_empty_string_context() -> None:
    """context='' → treated as non-None (falsy), NOT identity.

    Design decision: empty string is falsy in Python, so _prefix_attachment_context
    treats it identically to None (short-circuits) and returns the original
    system_instruction unchanged. This matches the design spec's note that
    attachment_context=None or '' → no prefix. We document this explicitly.
    """
    instruction = "You are a helpful SRE agent."
    result = _prefix_attachment_context(instruction, "")
    # Empty string is falsy → identity (same as None)
    assert result == instruction
    assert ATTACHMENT_HEADER not in result
