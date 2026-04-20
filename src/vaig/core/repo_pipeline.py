"""Tiered file-processing pipeline (SPEC-V2-REPO-02, SPEC-V2-REPO-06).

Implements the T-0 → T-1 classifier layer:

- ``FileMeta``            – metadata from T-0 discovery
- ``TierOutcome``         – processing tier for a file
- ``Relevance``           – relevance level for classification
- ``ClassificationResult``– output of the classifier
- ``EvidenceGap``         – records what was NOT processed and why (REPO-06)
- ``classify_file()``     – apply classifier rules to a FileMeta
- ``classify_file_with_evidence()`` – classify and emit evidence gaps
- ``apply_file_cap()``    – enforce max_files cap with evidence gaps
- ``generate_repo_guidance()`` – full-scan safety-net finding (REPO-06)
- ``detect_file_kind()``  – sniff first 4 KB to determine YAML subkind
- ``is_binary_file()``    – detect binary content by null-byte check
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vaig.core.config import RepoInvestigationConfig

# ── Glob matching ─────────────────────────────────────────────────────────────


def _glob_match(path: str, pattern: str) -> bool:
    """Match *path* against *pattern* with full ``**`` glob support.

    Handles:
    - ``*``  — any characters within a single path component
    - ``**`` — zero or more path components (including zero)
    - ``?``  — any single character within a component

    This is a lightweight alternative to ``pathlib.Path.match`` which has
    incomplete ``**`` support in Python ≤ 3.12.

    Args:
        path:    Forward-slash separated relative path (e.g. ``"a/b/c.yaml"``).
        pattern: Glob pattern (e.g. ``"**/Chart.yaml"`` or ``"vendor/**"``).

    Returns:
        ``True`` if *path* matches *pattern*.
    """
    # Normalise separators
    path = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    # Convert glob pattern to regex:
    # 1. Replace ** with a unique placeholder
    # 2. Escape remaining special regex chars
    # 3. Convert * and ? to appropriate patterns
    # 4. Replace placeholder with multi-segment pattern

    _DOUBLE_STAR = "\x00DS\x00"
    _SINGLE_STAR = "\x00SS\x00"
    _QUESTION = "\x00QM\x00"

    # Replace ** before * to avoid conflict
    pattern = pattern.replace("**", _DOUBLE_STAR)
    pattern = pattern.replace("*", _SINGLE_STAR)
    pattern = pattern.replace("?", _QUESTION)

    # Escape remaining regex metacharacters
    pattern = re.escape(pattern)

    # Restore placeholders as regex patterns
    # ** matches zero or more path components (may include slashes)
    # When ** appears as **/ (prefix), it means "any directory prefix or empty"
    pattern = pattern.replace(
        re.escape(_DOUBLE_STAR) + re.escape("/"),
        "(?:.*/)?"
    )
    # Remaining ** (e.g. at end: "dir/**")
    pattern = pattern.replace(re.escape(_DOUBLE_STAR), ".*")
    # * matches anything within a path component (no slash)
    pattern = pattern.replace(re.escape(_SINGLE_STAR), "[^/]*")
    # ? matches any single char within a component
    pattern = pattern.replace(re.escape(_QUESTION), "[^/]")

    regex = "^" + pattern + "$"
    return bool(re.match(regex, path))


class FileMeta(BaseModel):
    """File metadata from T-0 discovery."""

    path: str
    size: int
    sha: str | None = None
    kind: str = "unknown"
    is_binary: bool = False


class TierOutcome(StrEnum):
    """Processing tier outcome for a file."""

    SKIP = "skip"
    HEADER = "header"
    CHUNKED = "chunked"
    CHUNKED_STREAMING = "chunked_streaming"


class Relevance(StrEnum):
    """Relevance level for classification."""

    HIGH = "high"
    MEDIUM = "med"
    LOW = "low"
    AUTO = "auto"


class ClassificationResult(BaseModel):
    """Result of classifying a file."""

    file: FileMeta
    outcome: TierOutcome
    relevance: Relevance = Relevance.LOW
    reason: str


class EvidenceGap(BaseModel):
    """Records what was NOT processed and why (SPEC-V2-REPO-06).

    Emitted when a file is skipped, capped, or handled non-standardly.
    Collected by callers to surface to users without hard-failing the run.
    """

    source: str  # "repo_processing"
    kind: str
    """One of: streaming_used, catastrophic_size, path_file_cap, dropped_over_cap,
    binary_skipped, excluded_glob, chunker_fallback."""
    level: Literal["INFO", "WARN"] = "INFO"
    path: str | None = None
    details: str


# ── Classifier rule infrastructure ────────────────────────────────────────────


@dataclasses.dataclass
class _Rule:
    """A single classifier rule."""

    pred: Callable[[FileMeta, RepoInvestigationConfig], bool]
    outcome: TierOutcome
    relevance: Relevance = Relevance.LOW
    reason: str = ""


def _build_classifier_rules() -> list[_Rule]:
    """Build the ordered list of classifier rules (first match wins)."""
    return [
        # ── Legitimate skips ───────────────────────────────────────────────
        _Rule(
            pred=lambda m, _c: m.is_binary,
            outcome=TierOutcome.SKIP,
            reason="binary",
        ),
        _Rule(
            pred=lambda m, c: any(
                _glob_match(m.path, g) for g in c.exclude_globs
            ),
            outcome=TierOutcome.SKIP,
            reason="exclude_glob",
        ),
        _Rule(
            pred=lambda m, c: m.size > c.max_file_bytes_absolute,
            outcome=TierOutcome.SKIP,
            reason="catastrophic_size",
        ),
        # ── Large text → streaming chunker ────────────────────────────────
        _Rule(
            pred=lambda m, c: m.size > c.streaming_threshold_bytes,
            outcome=TierOutcome.CHUNKED_STREAMING,
            relevance=Relevance.AUTO,
            reason="large_text_stream",
        ),
        # ── High relevance ────────────────────────────────────────────────
        _Rule(
            pred=lambda m, _c: (
                _glob_match(m.path, "**/Chart.yaml")
                or _glob_match(m.path, "**/Chart.lock")
            ),
            outcome=TierOutcome.CHUNKED,
            relevance=Relevance.HIGH,
            reason="helm_chart_root",
        ),
        _Rule(
            pred=lambda m, _c: (
                _glob_match(m.path, "**/values*.yaml")
                or _glob_match(m.path, "**/values*.yml")
            ),
            outcome=TierOutcome.CHUNKED,
            relevance=Relevance.HIGH,
            reason="helm_values",
        ),
        _Rule(
            pred=lambda m, _c: (
                m.kind == "istio_crd"
                or _glob_match(m.path, "**/virtualservice*.yaml")
                or _glob_match(m.path, "**/virtualservice*.yml")
            ),
            outcome=TierOutcome.CHUNKED,
            relevance=Relevance.HIGH,
            reason="istio_routing",
        ),
        _Rule(
            pred=lambda m, _c: (
                (
                    _glob_match(m.path, "**/Application*.yaml")
                    or _glob_match(m.path, "**/Application*.yml")
                )
                and m.kind == "argocd"
            ),
            outcome=TierOutcome.CHUNKED,
            relevance=Relevance.HIGH,
            reason="argocd_app",
        ),
        # ── Medium relevance ──────────────────────────────────────────────
        _Rule(
            pred=lambda m, _c: m.kind in {"k8s_manifest", "kustomization"},
            outcome=TierOutcome.CHUNKED,
            relevance=Relevance.MEDIUM,
            reason="k8s_manifest",
        ),
        _Rule(
            pred=lambda m, _c: (
                _glob_match(m.path, "**/templates/*.yaml")
                or _glob_match(m.path, "**/templates/*.yml")
            ),
            outcome=TierOutcome.CHUNKED,
            relevance=Relevance.MEDIUM,
            reason="helm_template",
        ),
        # ── Low relevance — header-only ───────────────────────────────────
        _Rule(
            pred=lambda m, _c: _glob_match(m.path, "**/README*"),
            outcome=TierOutcome.HEADER,
            relevance=Relevance.LOW,
            reason="readme",
        ),
        _Rule(
            pred=lambda m, _c: _glob_match(m.path, "**/CHANGELOG*"),
            outcome=TierOutcome.HEADER,
            relevance=Relevance.LOW,
            reason="changelog",
        ),
        # ── Default ──────────────────────────────────────────────────────
        _Rule(
            pred=lambda _m, _c: True,
            outcome=TierOutcome.HEADER,
            relevance=Relevance.LOW,
            reason="default",
        ),
    ]


_CLASSIFIER_RULES: list[_Rule] = _build_classifier_rules()


# ── File-kind detection ───────────────────────────────────────────────────────

_YAML_INDICATOR_CHARS = frozenset("{}[]|>:&*#!%@`'\"-")


def _looks_like_yaml(peek: str) -> bool:
    """Heuristic: YAML files typically have colon-separated key: value pairs."""
    lines = peek.splitlines()
    for line in lines[:20]:
        stripped_line = line.strip()
        if stripped_line and ":" in stripped_line and not stripped_line.startswith("#"):
            return True
    return False


def detect_file_kind(content_peek: str) -> str:
    """Detect YAML sub-kind from first 4 KB of content.

    Sniffs well-known ``apiVersion`` / ``kind`` patterns to classify the
    file as a specific Kubernetes/infrastructure type.  Falls back to
    ``"yaml"`` for any valid YAML, and ``"text"`` otherwise.

    Args:
        content_peek: First 4 096 bytes (or fewer) of file content.

    Returns:
        One of: ``argocd``, ``istio_crd``, ``k8s_manifest``,
        ``kustomization``, ``terraform_gke``, ``yaml``, ``text``.
    """
    if "apiVersion: argoproj.io/" in content_peek:
        return "argocd"
    if "apiVersion: networking.istio.io/" in content_peek:
        return "istio_crd"
    if (
        "apiVersion: apps/v1" in content_peek
        or "apiVersion: v1" in content_peek
        or "apiVersion: batch/v1" in content_peek
    ):
        return "k8s_manifest"
    if "kind: Kustomization" in content_peek:
        return "kustomization"
    if 'resource "google_container' in content_peek:
        return "terraform_gke"
    if _looks_like_yaml(content_peek):
        return "yaml"
    return "text"


# ── Binary detection ──────────────────────────────────────────────────────────

_NULL_BYTE = b"\x00"
_MAX_NON_TEXT_RATIO = 0.30


def is_binary_file(path: Path, peek_size: int = 8192) -> bool:
    """Return ``True`` if *path* appears to be a binary file.

    Uses two signals:

    1. Presence of a null byte (``\\x00``) — reliable indicator.
    2. High ratio (>30 %) of non-printable, non-whitespace bytes.

    Args:
        path:      Path to the file.
        peek_size: Number of bytes to read for the check (default 8 192).

    Returns:
        ``True`` if the file is likely binary, ``False`` otherwise.
    """
    try:
        raw = path.read_bytes()[:peek_size]
    except OSError:
        return False

    if _NULL_BYTE in raw:
        return True

    if not raw:
        return False

    non_text = sum(
        1 for b in raw if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F
    )
    return (non_text / len(raw)) > _MAX_NON_TEXT_RATIO


# ── Classifier ────────────────────────────────────────────────────────────────


def classify_file(
    meta: FileMeta,
    config: RepoInvestigationConfig,
) -> ClassificationResult:
    """Apply classifier rules to determine the processing tier for *meta*.

    Rules are evaluated in order; the first matching rule wins.  There is
    always a default rule at the end so this function always returns.

    Args:
        meta:   File metadata (path, size, kind, is_binary).
        config: Repository investigation configuration.

    Returns:
        A :class:`ClassificationResult` with the matched outcome, relevance,
        and reason.
    """
    for rule in _CLASSIFIER_RULES:
        if rule.pred(meta, config):
            return ClassificationResult(
                file=meta,
                outcome=rule.outcome,
                relevance=rule.relevance,
                reason=rule.reason,
            )
    # Unreachable — default rule always matches — but satisfies type-checker.
    return ClassificationResult(
        file=meta,
        outcome=TierOutcome.HEADER,
        relevance=Relevance.LOW,
        reason="default",
    )


def classify_file_with_evidence(
    meta: FileMeta,
    config: RepoInvestigationConfig,
) -> tuple[ClassificationResult, list[EvidenceGap]]:
    """Classify *meta* and emit :class:`EvidenceGap` records for notable outcomes.

    Wraps :func:`classify_file` and adds evidence gap emission for each
    SPEC-V2-REPO-06 situation that warrants it.

    Args:
        meta:   File metadata (path, size, kind, is_binary).
        config: Repository investigation configuration.

    Returns:
        A tuple of (ClassificationResult, list[EvidenceGap]).  The gap list
        is empty for ordinary files processed via the in-memory chunker.
    """
    result = classify_file(meta, config)
    gaps: list[EvidenceGap] = []

    if result.outcome == TierOutcome.SKIP:
        if result.reason == "binary":
            gaps.append(
                EvidenceGap(
                    source="repo_processing",
                    kind="binary_skipped",
                    level="INFO",
                    path=meta.path,
                    details=f"Binary file skipped: {meta.path}",
                )
            )
        elif result.reason == "exclude_glob":
            gaps.append(
                EvidenceGap(
                    source="repo_processing",
                    kind="excluded_glob",
                    level="INFO",
                    path=meta.path,
                    details=f"File excluded by glob pattern: {meta.path}",
                )
            )
        elif result.reason == "catastrophic_size":
            size_mb = meta.size / (1024 * 1024)
            cap_mb = config.max_file_bytes_absolute / (1024 * 1024)
            gaps.append(
                EvidenceGap(
                    source="repo_processing",
                    kind="catastrophic_size",
                    level="WARN",
                    path=meta.path,
                    details=(
                        f"{meta.path} is {size_mb:.0f} MB, exceeds "
                        f"max_file_bytes_absolute ({cap_mb:.0f} MB). "
                        f"File was not indexed. To process: re-run with "
                        f"--repo-max-file-bytes-absolute={meta.size * 2}"
                        f" (note: expect higher disk use and run time)."
                    ),
                )
            )
    elif result.outcome == TierOutcome.CHUNKED_STREAMING:
        gaps.append(
            EvidenceGap(
                source="repo_processing",
                kind="streaming_used",
                level="INFO",
                path=meta.path,
                details=(
                    f"File {meta.path} ({meta.size} bytes) exceeds "
                    f"streaming_threshold_bytes ({config.streaming_threshold_bytes}). "
                    f"Processed via streaming chunker."
                ),
            )
        )

    return result, gaps


def apply_file_cap(
    files: list[FileMeta],
    max_files: int,
) -> tuple[list[FileMeta], list[EvidenceGap]]:
    """Keep first N files by priority (shallow path depth then path lexicographic).

    When the number of files in a path exceeds *max_files*, keeps the
    N shallowest files and emits one summary gap plus one per-file gap for
    each dropped file.

    Args:
        files:     All discovered files in the path.
        max_files: Maximum number of files to retain.

    Returns:
        Tuple of (kept_files, evidence_gaps_for_dropped).  Gaps list is
        empty when ``len(files) <= max_files``.
    """
    if len(files) <= max_files:
        return files, []

    # Sort by depth (shallow first), then by path lexicographically for stability
    sorted_files = sorted(files, key=lambda f: (f.path.count("/"), f.path))
    kept = sorted_files[:max_files]
    dropped = sorted_files[max_files:]

    gaps: list[EvidenceGap] = [
        EvidenceGap(
            source="repo_processing",
            kind="path_file_cap",
            level="WARN",
            details=(
                f"Path has {len(files)} files, kept {max_files}. "
                f"{len(dropped)} files dropped."
            ),
        )
    ]
    gaps.extend(
        EvidenceGap(
            source="repo_processing",
            kind="dropped_over_cap",
            level="WARN",
            path=f.path,
            details=(
                f"File dropped due to cap ({len(files)} total, {max_files} max)"
            ),
        )
        for f in dropped
    )

    return kept, gaps


def generate_repo_guidance(
    tree_outline: list[tuple[str, int]],
) -> dict[str, object]:
    """Generate a repo_guidance INFO finding when no paths/globs specified.

    Builds a lightweight finding that surfaces top directories by YAML file
    count.  Returned as a plain dict (matches the Finding model shape used
    elsewhere in the codebase) so callers don't need to import the heavy
    service_health schema.

    Args:
        tree_outline: List of ``(dir_path, yaml_file_count)`` tuples,
                      typically built from a T-0 directory walk.

    Returns:
        A dict with ``category="repo_guidance"`` and ``severity="INFO"``.
    """
    # Sort descending by file count, take top 5
    top = sorted(tree_outline, key=lambda t: t[1], reverse=True)[:5]

    top_lines = "\n".join(
        f"  - `{dir_path}/` ({count} files)" for dir_path, count in top
    )

    description = (
        "No paths or globs specified. Emitting directory outline only. "
        "To deepen analysis, re-run with `--repo-path <dir>` specifying "
        "the 1–5 directories most likely to contain the relevant config.\n"
        f"Top directories by YAML file count:\n{top_lines}"
    )

    return {
        "id": "repo-guidance-no-paths",
        "title": "No paths or globs specified — full-scan outline only",
        "severity": "INFO",
        "category": "repo_guidance",
        "description": description,
        "recommendations": [dir_path for dir_path, _count in top],
    }
