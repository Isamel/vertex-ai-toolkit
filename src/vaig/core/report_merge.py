"""Structural merger for HealthReports produced by Map-Reduce attachment analysis.

Pure-Python, deterministic, no LLM calls.  Used by ``execute_skill_headless``
to consolidate per-window reports into a single ``HealthReport`` with the
public ``SkillResult`` contract intact.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from vaig.skills.service_health.schema import (
    EvidenceGap,
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    RecommendedAction,
    RootCauseHypothesis,
    Severity,
    TimelineEvent,
)

logger = logging.getLogger(__name__)

# Severity ordering — index in this dict = priority (0 = highest).
# Matches the StrEnum declaration order in schema.py (CRITICAL > HIGH > MEDIUM > LOW > INFO).
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}

_OVERALL_STATUS_FROM_WORST_SEVERITY: dict[Severity, OverallStatus] = {
    Severity.CRITICAL: OverallStatus.CRITICAL,
    Severity.HIGH: OverallStatus.DEGRADED,
    Severity.MEDIUM: OverallStatus.DEGRADED,
    Severity.LOW: OverallStatus.HEALTHY,
    Severity.INFO: OverallStatus.HEALTHY,
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Maximum number of hypotheses to keep in the merged report (schema constraint).
_MAX_HYPOTHESES: int = 4


# ── B2: _slugify ─────────────────────────────────────────────────────────────


def _slugify(value: str) -> str:
    """Normalize a Finding.id or hypothesis title for dedup.

    Lowercase → replace non-alphanumeric runs with ``'-'`` → strip leading/trailing ``'-'``.
    Empty input maps to empty string (caller decides whether to drop).

    Examples::

        >>> _slugify("Crash-Loop: payment-svc!")
        'crash-loop-payment-svc'
        >>> _slugify("")
        ''
    """
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


# ── B3: pure-Python helpers ───────────────────────────────────────────────────


def _longer_non_empty(a: str, b: str) -> str:
    """Return the longer of *a* and *b*; prefer non-empty.

    When both are non-empty, the longer value is returned; ties go to *a*.
    When one is empty the other is returned. When both empty, ``''`` is returned.
    """
    if not a:
        return b
    if not b:
        return a
    return a if len(a) >= len(b) else b


def _ordered_union(*seqs: list[str]) -> list[str]:
    """Return an ordered union of string sequences, preserving first-seen order.

    Deduplicates by value across all input sequences.  Order of first appearance
    is preserved.
    """
    seen: set[str] = set()
    out: list[str] = []
    for seq in seqs:
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _union(buckets: Iterable[list]) -> list:  # type: ignore[type-arg]
    """Concatenate all items from *buckets* without deduplication."""
    out: list = []  # type: ignore[type-arg]
    for b in buckets:
        out.extend(b)
    return out


def _dedup_first(buckets: Iterable[list], *, key: Any) -> list:  # type: ignore[type-arg]
    """Return a list deduped by *key*, keeping first-seen item per key.

    Args:
        buckets: Iterable of lists to draw items from.
        key: Callable that maps an item to a hashable dedup key.
    """
    seen: set[str] = set()
    out: list = []  # type: ignore[type-arg]
    for bucket in buckets:
        for item in bucket:
            k = key(item)
            if k not in seen:
                seen.add(k)
                out.append(item)
    return out


# ── B4: finding merge ─────────────────────────────────────────────────────────


def _collide_findings(a: Finding, b: Finding) -> Finding:
    """Merge two findings with the same slugified id.

    Severity collision: highest severity (lowest ``_SEVERITY_RANK`` index) wins.
    Text fields: longer non-empty string wins.
    List fields: ordered union.
    """
    sev = a.severity if _SEVERITY_RANK[a.severity] <= _SEVERITY_RANK[b.severity] else b.severity
    return a.model_copy(
        update={
            "severity": sev,
            "title": _longer_non_empty(a.title, b.title),
            "description": _longer_non_empty(a.description, b.description),
            "root_cause": _longer_non_empty(a.root_cause, b.root_cause),
            "impact": _longer_non_empty(a.impact, b.impact),
            "evidence": _ordered_union(a.evidence, b.evidence),
            "affected_resources": _ordered_union(a.affected_resources, b.affected_resources),
            "caused_by": _ordered_union(a.caused_by, b.caused_by),
            "causes": _ordered_union(a.causes, b.causes),
        }
    )


def _merge_findings(buckets: Iterable[list[Finding]]) -> list[Finding]:
    """Dedup findings by slugified id; collision keeps highest severity.

    Sort result by ``(_SEVERITY_RANK, slug)`` for a stable, severity-ordered output.
    Findings with an empty slug (empty id) are silently dropped.
    """
    merged: dict[str, Finding] = {}
    for bucket in buckets:
        for f in bucket:
            slug = _slugify(f.id)
            if not slug:
                logger.debug("dropping finding with empty slug id=%r", f.id)
                continue
            existing = merged.get(slug)
            if existing is None:
                merged[slug] = f.model_copy()
                continue
            merged[slug] = _collide_findings(existing, f)
    return sorted(
        merged.values(),
        key=lambda f: (_SEVERITY_RANK[f.severity], _slugify(f.id)),
    )


# ── B5: causal edge pruning ───────────────────────────────────────────────────


def _prune_causal_edges(findings: list[Finding], by_slug: dict[str, Finding]) -> None:
    """Mutate findings in place: remove caused_by/causes entries whose target is missing.

    After dedup across windows, some referenced slugs may no longer exist in the
    merged set.  This pass removes dangling references so the causal graph only
    contains valid edges.
    """
    for f in findings:
        f.caused_by = [s for s in f.caused_by if _slugify(s) in by_slug]
        f.causes = [s for s in f.causes if _slugify(s) in by_slug]


# ── B6: evidence gap merge ────────────────────────────────────────────────────


def _merge_evidence_gaps(buckets: Iterable[list[EvidenceGap]]) -> list[EvidenceGap]:
    """Dedup by ``(source, reason)``; union details into a semicolon-joined string.

    On collision: if ``new_details`` is not already contained in ``existing_details``,
    append with ``'; '`` separator.  Sorted by ``(source, reason)`` for stable output.
    """
    merged: dict[tuple[str, str], EvidenceGap] = {}
    for bucket in buckets:
        for gap in bucket:
            key = (gap.source, gap.reason)
            existing = merged.get(key)
            if existing is None:
                merged[key] = gap.model_copy()
                continue
            existing_details = existing.details or ""
            new_details = gap.details or ""
            if new_details and new_details not in existing_details:
                joined = f"{existing_details}; {new_details}" if existing_details else new_details
                merged[key] = existing.model_copy(update={"details": joined})
    return sorted(merged.values(), key=lambda g: (g.source, g.reason))


# ── B7: hypotheses, recommendations, timeline ────────────────────────────────


def _merge_hypotheses(
    buckets: Iterable[list[RootCauseHypothesis]],
) -> list[RootCauseHypothesis]:
    """Dedup by slugified label; collision keeps highest probability.

    ``HealthReport`` caps ``root_cause_hypotheses`` at 4 (``max_length=4``).
    After merge, truncate to top 4 by probability (descending) to avoid
    Pydantic validation errors when the merged report is constructed.
    """
    merged: dict[str, RootCauseHypothesis] = {}
    for bucket in buckets:
        for h in bucket:
            slug = _slugify(h.label)
            if not slug:
                continue
            existing = merged.get(slug)
            if existing is None or h.probability > existing.probability:
                merged[slug] = h
    ranked = sorted(merged.values(), key=lambda h: h.probability, reverse=True)
    return ranked[:_MAX_HYPOTHESES]


def _merge_recommendations(
    buckets: Iterable[list[RecommendedAction]],
) -> list[RecommendedAction]:
    """Dedup by slugified title; keep highest-priority (lowest int) version."""
    merged: dict[str, RecommendedAction] = {}
    for bucket in buckets:
        for r in bucket:
            slug = _slugify(r.title)
            if not slug:
                continue
            existing = merged.get(slug)
            if existing is None or r.priority < existing.priority:
                merged[slug] = r
    return sorted(merged.values(), key=lambda r: (r.priority, _slugify(r.title)))


def _merge_timeline(buckets: Iterable[list[TimelineEvent]]) -> list[TimelineEvent]:
    """Union timeline events; dedup by ``(time, event, service)``; sort by time."""
    seen: set[tuple] = set()  # type: ignore[type-arg]
    out: list[TimelineEvent] = []
    for bucket in buckets:
        for ev in bucket:
            key = (ev.time, ev.event, getattr(ev, "service", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(ev)
    return sorted(out, key=lambda e: e.time)


# ── B8: executive summary rebuild ────────────────────────────────────────────


def _rebuild_executive_summary(
    findings: list[Finding],
    template: ExecutiveSummary,
) -> ExecutiveSummary:
    """Rebuild ExecutiveSummary from merged findings.

    ``overall_status`` derives from the worst severity present:

    - any CRITICAL → CRITICAL
    - any HIGH or MEDIUM → DEGRADED
    - else (LOW / INFO only) → HEALTHY
    - no findings at all → UNKNOWN

    All other ``ExecutiveSummary`` fields (scope, summary_text, …) are
    inherited from the template (first window's summary) — they describe
    investigation framing, not severity-derived state.
    """
    if not findings:
        return template.model_copy(update={"overall_status": OverallStatus.UNKNOWN})
    worst = min(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity
    return template.model_copy(
        update={
            "overall_status": _OVERALL_STATUS_FROM_WORST_SEVERITY[worst],
        }
    )


# ── B9: causal graph rebuild ──────────────────────────────────────────────────


def _rebuild_causal_graph(findings: list[Finding], by_slug: dict[str, Finding]) -> str | None:
    """Emit a Mermaid ``graph TD`` string from pruned ``caused_by`` edges.

    Returns ``None`` when there are zero valid edges — preserves the original
    'no graph available' semantic in the renderer.
    """
    edges: list[tuple[str, str]] = []
    for f in findings:
        slug_self = _slugify(f.id)
        for upstream in f.caused_by:
            slug_up = _slugify(upstream)
            if slug_up in by_slug:
                edges.append((slug_up, slug_self))
    if not edges:
        return None
    lines = ["graph TD"]
    for a, b in edges:
        lines.append(f"    {a} --> {b}")
    return "\n".join(lines)


# ── B10: public API ───────────────────────────────────────────────────────────


def merge_health_reports(reports: list[HealthReport]) -> HealthReport | None:
    """Merge per-window HealthReports into one consolidated report.

    Args:
        reports: List of ``HealthReport`` objects, one per Map-Reduce window.
            Empty list returns ``None`` (caller decides fallback).

    Returns:
        Merged ``HealthReport``, or ``None`` when *reports* is empty.

    Merge semantics:

    - **Findings**: deduplicated by slugified ``id``; severity collision keeps
      the highest; text fields keep the longer non-empty value; list fields are
      unioned in order.
    - **Causal edges**: dangling references (pointing to findings that were
      deduplicated away) are pruned before the Mermaid graph is rebuilt.
    - **EvidenceGaps**: deduplicated by ``(source, reason)``; details are
      unioned with ``'; '`` separator.
    - **Hypotheses**: deduplicated by slug of ``label``; highest probability
      kept; result capped at 4 (schema ``max_length=4``).
    - **Recommendations**: deduplicated by slug of ``title``; lowest priority
      int (= highest urgency) kept.
    - **Timeline**: union, dedup by ``(timestamp, event, service)``, sorted.
    - **executive_summary.overall_status**: derived from worst severity in
      merged findings.
    - **cluster_overview / service_statuses**: first-seen per primary key
      (these describe global state, not window-local observations).
    - **downgraded_findings / evidence_details / manual_investigations /
      recent_changes**: concatenated without dedup (small, low-risk lists).
    """
    if not reports:
        return None
    if len(reports) == 1:
        return reports[0]

    findings = _merge_findings(r.findings for r in reports)
    findings_by_slug = {_slugify(f.id): f for f in findings}
    _prune_causal_edges(findings, findings_by_slug)

    gaps = _merge_evidence_gaps(r.evidence_gaps for r in reports)
    hypotheses = _merge_hypotheses(r.root_cause_hypotheses for r in reports)
    recommendations = _merge_recommendations(r.recommendations for r in reports)
    timeline = _merge_timeline(r.timeline for r in reports)

    exec_summary = _rebuild_executive_summary(findings, reports[0].executive_summary)
    causal_graph = _rebuild_causal_graph(findings, findings_by_slug)

    base = reports[0]
    return base.model_copy(
        update={
            "executive_summary": exec_summary,
            "findings": findings,
            "evidence_gaps": gaps,
            "root_cause_hypotheses": hypotheses,
            "recommendations": recommendations,
            "timeline": timeline,
            "causal_graph_mermaid": causal_graph,
            "downgraded_findings": _union(r.downgraded_findings for r in reports),
            "evidence_details": _union(r.evidence_details for r in reports),
            "manual_investigations": _union(r.manual_investigations for r in reports),
            "recent_changes": _union(r.recent_changes for r in reports),
            "cluster_overview": _dedup_first((r.cluster_overview for r in reports), key=lambda m: m.metric),
            "service_statuses": _dedup_first((r.service_statuses for r in reports), key=lambda s: s.service),
        }
    )
