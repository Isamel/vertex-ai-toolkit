"""Secret redaction pipeline for repo content before chunking/embedding.

Runs **before** chunking so no original secrets reach the vector index or LLM
context window.  Preserves character offsets (length-preserving replacement) so
line-range traceability is maintained.

Usage::

    redactor = SecretRedactor()
    result = redactor.redact(file_content, file_path="values.yaml")
    # result.redacted_content is safe to embed
    # result.redactions is the audit trail (no original secrets)
"""

from __future__ import annotations

import collections
import math
import re
from pathlib import Path

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RedactionEntry(BaseModel):
    """Record of a single redaction (audit log entry).

    NEVER stores the original secret value.
    """

    file: str
    line: int
    kind: str  # "aws_key", "gcp_sa", "jwt", "high_entropy", "yaml_key_value", …
    reason: str


class RedactionResult(BaseModel):
    """Result of redacting a single file."""

    redacted_content: str
    redactions: list[RedactionEntry]
    skipped: bool = False  # True if secret_density_too_high
    skip_reason: str | None = None


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # (kind, pattern, reason)
    (
        "aws_access_key",
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "AWS access key ID",
    ),
    (
        "aws_secret_key",
        re.compile(r"(?<![A-Za-z0-9/+])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
        "Possible AWS secret key",
    ),
    (
        "gcp_sa_key",
        re.compile(r'"private_key"\s*:\s*"-----BEGIN'),
        "GCP service account private key",
    ),
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        "JWT token",
    ),
    (
        "bearer_token",
        re.compile(r"[Bb]earer\s+[A-Za-z0-9_-]{20,}"),
        "Bearer token",
    ),
    (
        "ssh_private_key",
        re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "SSH private key",
    ),
    (
        "azure_connection",
        re.compile(r"DefaultEndpointsProtocol=https;AccountName="),
        "Azure connection string",
    ),
    (
        "generic_api_key",
        re.compile(
            r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?[A-Za-z0-9_-]{16,}"
        ),
        "Generic API key",
    ),
]

# YAML key patterns that indicate sensitive values
_YAML_SENSITIVE_KEYS = re.compile(
    r"(?i)^(\s*)(['\"]?)(password|passwd|token|secret|key|credential|auth|apikey)s?(['\"]?)(\s*:\s*)(.+)$",
    re.MULTILINE,
)

# Known-safe patterns that should NOT be flagged by high-entropy detector
_KNOWN_SAFE_PATTERNS = [
    re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE),  # UUID
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = collections.Counter(s)
    length = len(s)
    prob = [c / length for c in counts.values()]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def _is_known_safe(s: str) -> bool:
    """Return True if *s* is a recognised non-secret value."""
    if any(p.fullmatch(s) for p in _KNOWN_SAFE_PATTERNS):
        return True
    # Very short tokens are unlikely secrets
    if len(s) < 24:
        return True
    return False


def is_high_entropy_secret(
    s: str, threshold: float = 4.0, min_len: int = 24
) -> bool:
    """Check if *s* is a likely secret based on Shannon entropy."""
    if len(s) < min_len:
        return False
    if _is_known_safe(s):
        return False
    return shannon_entropy(s) > threshold


def is_pre_encrypted(content_peek: str) -> bool:
    """Return True if the file is already encrypted (SealedSecret or SOPS)."""
    peek = content_peek[:500]
    return (
        "apiVersion: bitnami.com/v1alpha1" in peek
        or "sops:" in peek
        or "ENC[AES256" in peek
    )


def _offset_to_line(content: str, offset: int) -> int:
    """Return 1-based line number for a byte *offset* inside *content*."""
    return content[:offset].count("\n") + 1


def _redact_preserving_length(
    content: str, start: int, end: int, replacement: str
) -> str:
    """Replace ``content[start:end]`` with *replacement*, padding to preserve length."""
    original_len = end - start
    if len(replacement) < original_len:
        replacement = replacement + "X" * (original_len - len(replacement))
    elif len(replacement) > original_len:
        replacement = replacement[:original_len]
    return content[:start] + replacement + content[end:]


# ---------------------------------------------------------------------------
# Main redactor
# ---------------------------------------------------------------------------


class SecretRedactor:
    """Redacts secrets from file content before chunking/embedding.

    Parameters
    ----------
    enabled:
        Set to ``False`` to disable all redaction (opt-out mode for demos).
    density_threshold:
        If the ratio ``len(redactions) / line_count`` exceeds this value the
        entire file is skipped with ``skip_reason="secret_density_too_high"``.
        Default 0.1 (10 %).
    """

    def __init__(
        self,
        enabled: bool = True,
        density_threshold: float = 0.1,
    ) -> None:
        self.enabled = enabled
        self.density_threshold = density_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, content: str, file_path: str) -> RedactionResult:
        """Redact secrets from *content*.

        Preserves byte offsets for line-range traceability by padding
        replacements to match the original length.

        Parameters
        ----------
        content:
            Raw file content.
        file_path:
            Used only for audit log entries — never read from disk here.

        Returns
        -------
        RedactionResult
            ``redacted_content`` is safe to embed.  ``redactions`` is the
            audit trail (no original secret values).
        """
        if not self.enabled:
            return RedactionResult(redacted_content=content, redactions=[])

        # Pre-encrypted files pass through unchanged
        if is_pre_encrypted(content):
            return RedactionResult(redacted_content=content, redactions=[])

        redactions: list[RedactionEntry] = []
        redacted = content

        # ── 1. Pattern-based detection ──────────────────────────────
        for kind, pattern, reason in _SECRET_PATTERNS:
            for m in list(pattern.finditer(redacted)):
                line_no = _offset_to_line(redacted, m.start())
                replacement_label = f"<redacted:{kind}>"
                redacted = _redact_preserving_length(
                    redacted, m.start(), m.end(), replacement_label
                )
                redactions.append(
                    RedactionEntry(
                        file=file_path,
                        line=line_no,
                        kind=kind,
                        reason=reason,
                    )
                )

        # ── 2. YAML key heuristic ───────────────────────────────────
        def _yaml_replacer(m: re.Match[str]) -> str:
            indent = m.group(1)
            q1 = m.group(2)
            key_name = m.group(3)
            q2 = m.group(4)
            sep = m.group(5)
            value = m.group(6)

            # Only redact values that are long enough to be real secrets;
            # short values (< 20 chars) are left unchanged to avoid
            # corrupting the <redacted:...> placeholder in LLM context.
            if len(value) < 20:
                return m.group(0)

            # Record the redaction (line computed from current state)
            line_no = _offset_to_line(redacted, m.start())
            redactions.append(
                RedactionEntry(
                    file=file_path,
                    line=line_no,
                    kind="yaml_key_value",
                    reason=f"Sensitive YAML key: {key_name}",
                )
            )
            replacement_value = f"<redacted:{key_name.lower()}>"
            # Preserve length of the value portion
            if len(replacement_value) < len(value):
                replacement_value = replacement_value + "X" * (
                    len(value) - len(replacement_value)
                )
            elif len(replacement_value) > len(value):
                replacement_value = replacement_value[: len(value)]

            return f"{indent}{q1}{key_name}{q2}{sep}{replacement_value}"

        # We need to handle YAML replacements carefully because the regex
        # captures the full line; we do a two-pass approach here to
        # correctly compute line numbers before mutating `redacted`.
        yaml_redacted = _YAML_SENSITIVE_KEYS.sub(_yaml_replacer, redacted)
        redacted = yaml_redacted

        # ── 3. High-entropy strings (most expensive — last) ─────────
        # Split on whitespace/quotes/colons to get candidate tokens
        _TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{24,}")
        offset_delta = 0  # accumulates as we mutate `redacted`
        base = redacted  # snapshot before mutations in this pass
        for m in _TOKEN_RE.finditer(base):
            token = m.group()
            if is_high_entropy_secret(token):
                actual_start = m.start() + offset_delta
                actual_end = m.end() + offset_delta
                line_no = _offset_to_line(redacted, actual_start)
                label = "<redacted:high_entropy>"
                old_len = actual_end - actual_start
                new_segment = label + "X" * max(0, old_len - len(label))
                if len(new_segment) > old_len:
                    new_segment = new_segment[:old_len]
                redacted = redacted[:actual_start] + new_segment + redacted[actual_end:]
                offset_delta += len(new_segment) - (m.end() - m.start())
                redactions.append(
                    RedactionEntry(
                        file=file_path,
                        line=line_no,
                        kind="high_entropy",
                        reason="High-entropy string (possible secret)",
                    )
                )

        # ── 4. Density check ────────────────────────────────────────
        # Only apply density check when the file is large enough that
        # a meaningful ratio can be computed (at least 5 lines).
        line_count = max(len(content.splitlines()), 1)
        if line_count >= 5 and len(redactions) / line_count > self.density_threshold:
            return RedactionResult(
                redacted_content="",
                redactions=redactions,
                skipped=True,
                skip_reason="secret_density_too_high",
            )

        return RedactionResult(redacted_content=redacted, redactions=redactions)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def write_redaction_log(
    run_id: str,
    entries: list[RedactionEntry],
    log_dir: Path | None = None,
) -> Path:
    """Write redaction audit log.

    Entries are written (overwriting any previous file) as JSONL to
    ``.vaig/repo-redactions/{run_id}.jsonl``.

    Parameters
    ----------
    run_id:
        Unique identifier for this indexing run.
    entries:
        Redaction entries collected across all files.
    log_dir:
        Override base directory.  Defaults to
        ``Path.cwd() / ".vaig" / "repo-redactions"``.

    Returns
    -------
    Path
        Absolute path to the written log file.
    """
    base = log_dir or (Path.cwd() / ".vaig" / "repo-redactions")
    base.mkdir(parents=True, exist_ok=True)
    log_file = base / f"{run_id}.jsonl"
    with log_file.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.model_dump_json() + "\n")
    return log_file.resolve()
