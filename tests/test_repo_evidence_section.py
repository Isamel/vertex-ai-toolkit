"""Tests for SPEC-V2-REPO-08 — RepoEvidenceSection model and HTML renderer."""
from __future__ import annotations

from vaig.core.repo_evidence_section import (
    FileCitation,
    FindingEvidenceSummary,
    PathSummary,
    RepoEvidenceSection,
    build_repo_evidence_section,
    render_repo_evidence_html,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


class _FakeFinding:
    """Duck-typed Finding stub."""

    def __init__(
        self,
        finding_id: str = "F-1",
        title: str = "Test Finding",
        severity: str = "HIGH",
        repo_evidence: list | None = None,
    ) -> None:
        self.id = finding_id
        self.title = title
        self.severity = severity
        self.repo_evidence = repo_evidence if repo_evidence is not None else []


class _FakeGap:
    """Duck-typed EvidenceGap stub."""

    def __init__(self, kind: str = "dropped_over_cap", details: str = "some/file.py") -> None:
        self.kind = kind
        self.details = details
        self.reason = ""


def _make_path_summaries(
    *specs: tuple[str, int, int],
) -> list[PathSummary]:
    return [PathSummary(path=p, total_files=tf, chunks_retrieved=cr) for p, tf, cr in specs]


# ── build_repo_evidence_section tests ─────────────────────────────────────────


def test_build_section_totals() -> None:
    summaries = _make_path_summaries(
        ("configs/", 10, 5),
        ("services/", 20, 8),
    )
    section = build_repo_evidence_section(
        repo_label="acme/repo@main",
        path_summaries=summaries,
        findings=[],
        evidence_gaps=[],
        all_file_citations=[],
    )
    assert section.total_candidate_files == 30
    assert section.total_retrieved_chunks == 13


def test_build_section_finding_summaries() -> None:
    findings = [
        _FakeFinding(repo_evidence=[{"file": "a.py"}]),  # has evidence
        _FakeFinding(repo_evidence=[{"file": "b.py"}, {"file": "c.py"}]),  # has evidence
        _FakeFinding(repo_evidence=[]),  # no evidence — excluded
    ]
    section = build_repo_evidence_section(
        repo_label="",
        path_summaries=_make_path_summaries((".", 5, 3)),
        findings=findings,
        evidence_gaps=[],
        all_file_citations=[],
    )
    assert len(section.finding_summaries) == 2


def test_build_section_top_cited_files_limit() -> None:
    citations = [
        FileCitation(file_path=f"file{i}.py", chunks_retrieved=i)
        for i in range(10)
    ]
    section = build_repo_evidence_section(
        repo_label="",
        path_summaries=_make_path_summaries((".", 10, 5)),
        findings=[],
        evidence_gaps=[],
        all_file_citations=citations,
    )
    assert len(section.top_cited_files) <= 5


def test_build_section_top_cited_files_sorted() -> None:
    citations = [
        FileCitation(file_path="low.py", chunks_retrieved=1),
        FileCitation(file_path="high.py", chunks_retrieved=99),
        FileCitation(file_path="mid.py", chunks_retrieved=50),
    ]
    section = build_repo_evidence_section(
        repo_label="",
        path_summaries=_make_path_summaries((".", 3, 3)),
        findings=[],
        evidence_gaps=[],
        all_file_citations=citations,
    )
    assert section.top_cited_files[0].file_path == "high.py"
    assert section.top_cited_files[0].chunks_retrieved == 99


def test_build_section_unretrieved_from_gaps() -> None:
    gaps = [
        _FakeGap(kind="dropped_over_cap", details="too_big.yaml"),
        _FakeGap(kind="excluded_glob", details="secret.env"),
        _FakeGap(kind="binary_skipped", details="image.png"),  # not dropped/excluded
    ]
    section = build_repo_evidence_section(
        repo_label="",
        path_summaries=_make_path_summaries((".", 5, 2)),
        findings=[],
        evidence_gaps=gaps,
        all_file_citations=[],
    )
    assert "too_big.yaml" in section.unretrieved_file_paths
    assert "secret.env" in section.unretrieved_file_paths
    assert "image.png" not in section.unretrieved_file_paths


# ── render_repo_evidence_html tests ───────────────────────────────────────────


def _make_section(**kwargs: object) -> RepoEvidenceSection:
    defaults: dict = {
        "repo_label": "acme/repo@main",
        "path_summaries": [PathSummary(path="configs/", total_files=5, chunks_retrieved=3)],
        "total_candidate_files": 5,
        "total_retrieved_chunks": 3,
        "finding_summaries": [],
        "top_cited_files": [],
        "unretrieved_file_paths": [],
    }
    defaults.update(kwargs)
    return RepoEvidenceSection(**defaults)  # type: ignore[arg-type]


def test_render_html_empty_when_no_paths() -> None:
    section = RepoEvidenceSection()  # no path_summaries
    assert render_repo_evidence_html(section) == ""


def test_render_html_contains_repo_label() -> None:
    section = _make_section(repo_label="my-org/configs@main")
    html = render_repo_evidence_html(section)
    assert "my-org/configs@main" in html


def test_render_html_counts_strip() -> None:
    section = _make_section(
        total_candidate_files=42,
        total_retrieved_chunks=17,
    )
    html = render_repo_evidence_html(section)
    assert "42" in html
    assert "17" in html


def test_render_html_hidden_when_no_repo() -> None:
    """Equivalent to empty section: no path_summaries → empty string."""
    section = RepoEvidenceSection(repo_label="some/repo@main")
    # Still no path_summaries
    assert render_repo_evidence_html(section) == ""


def test_render_html_finding_severity_icon() -> None:
    section = _make_section(
        finding_summaries=[
            FindingEvidenceSummary(
                finding_id="F-1",
                finding_title="Critical Auth Issue",
                severity="CRITICAL",
                snippet_count=3,
            )
        ]
    )
    html = render_repo_evidence_html(section)
    assert "🔴" in html
