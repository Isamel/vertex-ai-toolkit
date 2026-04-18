"""Redact sensitive values from tool call outputs before storage."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that match common secret formats.
# Each tuple is (human label, compiled regex).  The first capture group
# in every pattern is the *prefix* that should be preserved; the rest of
# the match is replaced with ``_REDACTED``.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Bearer/API token",
        re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.]{20,}", re.IGNORECASE),
    ),
    (
        "Generic API key",
        re.compile(
            r'(["\']?(?:api[_-]?key|apikey|api[_-]?secret)["\']?\s*[:=]\s*["\']?)'
            r"[A-Za-z0-9\-_\.]{16,}",
            re.IGNORECASE,
        ),
    ),
    (
        "Generic token",
        re.compile(
            r'(["\']?(?:token|secret|password|passwd|pwd|credentials?)["\']?\s*[:=]\s*["\']?)'
            r"[A-Za-z0-9\-_\.\/\+]{8,}",
            re.IGNORECASE,
        ),
    ),
    (
        "Base64 secret (long)",
        re.compile(r"(data:\s*)[A-Za-z0-9+/]{50,}={0,2}"),
    ),
    (
        "K8s service account token",
        re.compile(r"(eyJhbGciOi)[A-Za-z0-9\-_\.]+", re.IGNORECASE),
    ),
    (
        "Private key block",
        re.compile(
            r"(-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE|ENCRYPTED)\s+KEY-----)"
            r".+?"
            r"(-----END\s+(?:RSA\s+)?(?:PRIVATE|ENCRYPTED)\s+KEY-----)",
            re.DOTALL,
        ),
    ),
    (
        "GCP SA key",
        re.compile(r'("private_key"\s*:\s*")[^"]{20,}', re.IGNORECASE),
    ),
    (
        "Hex secret (32+)",
        re.compile(
            r'(["\']?(?:secret|key|token|hash)["\']?\s*[:=]\s*["\']?)[0-9a-fA-F]{32,}',
        ),
    ),
    (
        "AWS access key",
        re.compile(r"(AKIA)[0-9A-Z]{16}"),
    ),
    (
        "AWS secret key",
        re.compile(
            r'(["\']?(?:aws[_-]?secret[_-]?access[_-]?key|aws[_-]?secret)["\']?\s*[:=]\s*["\']?)'
            r"[A-Za-z0-9/+]{40}",
            re.IGNORECASE,
        ),
    ),
    (
        "GitHub ghp_ token",
        re.compile(r"(ghp_)[A-Za-z0-9]{36}"),
    ),
    (
        "GitHub gho_ token",
        re.compile(r"(gho_)[A-Za-z0-9]{36}"),
    ),
    (
        "Slack webhook URL",
        re.compile(r"(https://hooks\.slack\.com/services/T)[^\"'\s]{30,}"),
    ),
    (
        "DB connection string",
        re.compile(r"((postgres|mysql|mongodb)://)[^\"'\s]{10,}"),
    ),
]

_REDACTED = "***REDACTED***"


def redact_sensitive_output(output: str) -> tuple[str, int]:
    """Redact sensitive patterns from tool output.

    The full output is preserved (no truncation); only matched secret
    values are replaced with ``***REDACTED***``.

    Returns:
        Tuple of ``(redacted_output, redaction_count)``.
    """
    if not output:
        return output, 0

    redaction_count = 0
    result = output

    for _pattern_name, pattern in _SECRET_PATTERNS:

        def _replacer(match: re.Match[str]) -> str:
            nonlocal redaction_count
            redaction_count += 1
            # Keep all captured groups (prefix, suffix, etc.), redact between them.
            groups = [g for g in match.groups() if g is not None]
            if len(groups) > 1:
                return _REDACTED.join(groups)
            return groups[0] + _REDACTED if groups else _REDACTED

        result = pattern.sub(_replacer, result)

    if redaction_count > 0:
        logger.info("Redacted %d sensitive value(s) from tool output", redaction_count)

    return result, redaction_count
