"""SPEC-V2-REPO-08 — Repo Evidence section data model and renderer."""
from __future__ import annotations

import html

from pydantic import BaseModel, Field


class PathSummary(BaseModel):
    path: str
    total_files: int
    chunks_retrieved: int


class FileCitation(BaseModel):
    file_path: str
    chunks_retrieved: int
    link_url: str = ""  # populated by adapter.link_for() when available


class FindingEvidenceSummary(BaseModel):
    finding_id: str
    finding_title: str
    severity: str
    snippet_count: int


class RepoEvidenceSection(BaseModel):
    """Data model for the '📁 Repository Evidence' section of the HTML report."""

    repo_label: str = ""  # e.g. "acme/gateway-configs@main"
    path_summaries: list[PathSummary] = Field(default_factory=list)
    total_candidate_files: int = 0
    total_retrieved_chunks: int = 0
    finding_summaries: list[FindingEvidenceSummary] = Field(default_factory=list)
    top_cited_files: list[FileCitation] = Field(default_factory=list)  # top 5 by chunks_retrieved
    unretrieved_file_paths: list[str] = Field(default_factory=list)  # from EvidenceGaps


def build_repo_evidence_section(
    *,
    repo_label: str,
    path_summaries: list[PathSummary],
    findings: list[object],  # Finding objects — duck-typed
    evidence_gaps: list[object],  # EvidenceGap objects — duck-typed
    all_file_citations: list[FileCitation],
) -> RepoEvidenceSection:
    """Assemble a RepoEvidenceSection from pipeline outputs.

    - total_candidate_files = sum(p.total_files for p in path_summaries)
    - total_retrieved_chunks = sum(p.chunks_retrieved for p in path_summaries)
    - finding_summaries = findings that have repo_evidence (getattr fallback to [])
    - top_cited_files = top-5 all_file_citations sorted by chunks_retrieved desc
    - unretrieved_file_paths = [g.details for g in evidence_gaps
        if 'dropped' in g.reason or 'excluded' in g.reason]
    """
    total_candidate_files = sum(p.total_files for p in path_summaries)
    total_retrieved_chunks = sum(p.chunks_retrieved for p in path_summaries)

    finding_summaries: list[FindingEvidenceSummary] = []
    for f in findings:
        repo_evidence = getattr(f, "repo_evidence", []) or []
        if repo_evidence:
            finding_summaries.append(
                FindingEvidenceSummary(
                    finding_id=getattr(f, "id", ""),
                    finding_title=getattr(f, "title", ""),
                    severity=getattr(f, "severity", "INFO"),
                    snippet_count=len(repo_evidence),
                )
            )

    top_cited_files = sorted(
        all_file_citations,
        key=lambda c: c.chunks_retrieved,
        reverse=True,
    )[:5]

    unretrieved_file_paths: list[str] = []
    for g in evidence_gaps:
        reason = getattr(g, "reason", "") or ""
        kind = getattr(g, "kind", "") or ""
        # Check both reason and kind for dropped/excluded markers
        if "dropped" in reason or "excluded" in reason or "dropped" in kind or "excluded" in kind:
            details = getattr(g, "details", "") or ""
            if details:
                unretrieved_file_paths.append(details)

    return RepoEvidenceSection(
        repo_label=repo_label,
        path_summaries=path_summaries,
        total_candidate_files=total_candidate_files,
        total_retrieved_chunks=total_retrieved_chunks,
        finding_summaries=finding_summaries,
        top_cited_files=top_cited_files,
        unretrieved_file_paths=unretrieved_file_paths,
    )


_SEVERITY_ICON = {
    "CRITICAL": "🔴",
    "HIGH": "🟡",
    "MEDIUM": "🟠",
    "LOW": "⚪",
    "INFO": "ℹ️",
}


def _esc(text: str) -> str:
    """HTML-escape a string for safe embedding in attributes and text nodes."""
    return html.escape(str(text), quote=True)


def render_repo_evidence_html(section: RepoEvidenceSection) -> str:
    """Return the HTML fragment for the Repo Evidence section.

    When section is empty (no paths investigated), returns empty string.
    The caller inserts this into the SPA between other sections.

    Uses plain string building (no Jinja2 dependency) to keep this
    module self-contained.
    """
    if not section.path_summaries:
        return ""

    parts: list[str] = []
    parts.append('<section id="section-repo-evidence">')
    parts.append("  <h2>📁 Repository Evidence</h2>")

    # ── Counts strip ────────────────────────────────────────────────────────
    label_html = f" · <code>{_esc(section.repo_label)}</code>" if section.repo_label else ""
    parts.append(
        f'  <p class="repo-evidence-counts">'
        f"{label_html}"
        f" · <strong>{len(section.path_summaries)}</strong> path(s)"
        f" · <strong>{section.total_candidate_files}</strong> candidate file(s)"
        f" · <strong>{section.total_retrieved_chunks}</strong> chunk(s) retrieved"
        f"</p>"
    )

    # ── Paths investigated ───────────────────────────────────────────────────
    if section.path_summaries:
        parts.append("  <h3>Paths investigated</h3>")
        parts.append("  <ul>")
        for ps in section.path_summaries:
            parts.append(
                f"    <li><code>{_esc(ps.path)}</code>"
                f" — {ps.total_files} file(s), {ps.chunks_retrieved} chunk(s)</li>"
            )
        parts.append("  </ul>")

    # ── Findings backed by repo evidence ────────────────────────────────────
    if section.finding_summaries:
        parts.append("  <h3>Findings backed by repo evidence</h3>")
        parts.append("  <ul>")
        for fs in section.finding_summaries:
            icon = _SEVERITY_ICON.get(fs.severity.upper(), "❓")
            parts.append(
                f"    <li>{icon} [{_esc(fs.severity)}]"
                f" <strong>{_esc(fs.finding_title)}</strong>"
                f" — {fs.snippet_count} snippet(s)</li>"
            )
        parts.append("  </ul>")

    # ── Top-cited files ──────────────────────────────────────────────────────
    if section.top_cited_files:
        parts.append("  <h3>Top-cited files</h3>")
        parts.append("  <ol>")
        for fc in section.top_cited_files:
            if fc.link_url:
                file_html = f'<a href="{_esc(fc.link_url)}">{_esc(fc.file_path)}</a>'
            else:
                file_html = f"<code>{_esc(fc.file_path)}</code>"
            parts.append(f"    <li>{file_html} ({fc.chunks_retrieved} chunk(s))</li>")
        parts.append("  </ol>")

    # ── Files NOT retrieved (collapsed) ─────────────────────────────────────
    if section.unretrieved_file_paths:
        parts.append("  <details>")
        parts.append(
            f"    <summary>Files NOT retrieved ({len(section.unretrieved_file_paths)})</summary>"
        )
        parts.append("    <ul>")
        for path in section.unretrieved_file_paths:
            parts.append(f"      <li><code>{_esc(path)}</code></li>")
        parts.append("    </ul>")
        parts.append("  </details>")

    parts.append("</section>")

    return "\n".join(parts)


# TODO(SPEC-V2-REPO-08): Wire render_repo_evidence_html() into the SPA export.
# In src/vaig/ui/html_report.py, after the findings section is rendered,
# call render_repo_evidence_html(report.repo_evidence_section) and inject
# the returned fragment before the </main> closing tag, or add a new sentinel
# placeholder /*{{REPO_EVIDENCE_HTML}}*/"" in spa_template.html.
