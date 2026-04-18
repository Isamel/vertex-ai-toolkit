"""Observation fingerprinting for finding deduplication across runs.

A fingerprint is a 16-character SHA-256 prefix derived from a
finding's stable identity fields, after stripping PII and transient
tokens (pod hashes, UUIDs, IPs, timestamps, counters).
"""

from __future__ import annotations

import hashlib
import re

# Reuse canonical regex patterns from schema.py to normalise transient tokens.
# These patterns are intentionally re-declared here to keep the memory
# sub-package free of circular imports with service_health.schema.
_POD_HASH_RE = re.compile(r"-[a-z0-9]{5,10}-[a-z0-9]{5}\b")
_COUNTER_RE = re.compile(r"\b\d+\b")
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?"
)
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _strip_pii(text: str) -> str:
    """Remove PII / sensitive tokens using the output redactor when available.

    Falls back to a no-op when ``vaig.core.output_redactor`` is unavailable
    (e.g. in isolated unit tests).
    """
    try:
        from vaig.core.output_redactor import redact_sensitive_output  # noqa: PLC0415

        redacted, _ = redact_sensitive_output(text)
        return redacted
    except Exception:  # noqa: BLE001
        return text


def _normalise(text: str) -> str:
    """Normalise *text* by stripping PII and all transient tokens.

    The goal is to produce a stable canonical form that is identical
    across re-runs where only transient details change.
    """
    text = _strip_pii(text)
    text = _TIMESTAMP_RE.sub("<ts>", text)
    text = _UUID_RE.sub("<uuid>", text)
    text = _IP_RE.sub("<ip>", text)
    text = _POD_HASH_RE.sub("-<hash>", text)
    text = _COUNTER_RE.sub("<n>", text)
    return text.lower().strip()


class ObservationFingerprint:
    """Computes a stable 16-hex-char fingerprint for a finding.

    The fingerprint is derived from:
    ``sha256(kind|resource_kind|wildcard_name|canonical_symptom)[:16]``

    where each component is normalised to strip transient tokens.
    """

    @staticmethod
    def compute(
        kind: str,
        resource_kind: str,
        name: str,
        symptom: str,
    ) -> str:
        """Return a 16-character hex fingerprint.

        Args:
            kind: Finding category/kind (e.g. ``"pod-health"``).
            resource_kind: Kubernetes resource kind (e.g. ``"Deployment"``).
            name: Resource name (pod hashes will be stripped).
            symptom: Canonical symptom text (counters / IPs will be stripped).
        """
        canonical = "|".join(
            [
                _normalise(kind),
                _normalise(resource_kind),
                _normalise(name),
                _normalise(symptom),
            ]
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def from_finding(
        category: str,
        service: str,
        title: str,
        description: str,
    ) -> str:
        """Convenience wrapper that maps Finding fields to fingerprint inputs.

        Args:
            category: ``Finding.category`` (maps to *kind*).
            service: ``Finding.service`` (maps to *resource_kind*).
            title: ``Finding.title`` (maps to *name* — pod hashes stripped).
            description: ``Finding.description`` (maps to *symptom*).
        """
        return ObservationFingerprint.compute(
            kind=category,
            resource_kind=service,
            name=title,
            symptom=description,
        )
