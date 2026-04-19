"""Tests for SPEC-RP-01 — Finding fingerprint and dedup logic."""
from __future__ import annotations

from vaig.skills.service_health.schema import Finding, Severity
from vaig.skills.service_health.skill import _dedup_findings, _finding_fingerprint


def _make_finding(
    title: str = "Test finding",
    category: str = "pod-health",
    affected_resources: list[str] | None = None,
    *,
    fid: str = "test-finding",
) -> Finding:
    return Finding(
        id=fid,
        title=title,
        severity=Severity.MEDIUM,
        category=category,
        affected_resources=affected_resources or [],
    )


# ── _finding_fingerprint ──────────────────────────────────────────────────────


class TestFindingFingerprint:
    def test_returns_40_char_hex_string(self) -> None:
        f = _make_finding()
        fp = _finding_fingerprint(f)
        assert len(fp) == 40  # noqa: PLR2004
        assert all(c in "0123456789abcdef" for c in fp)

    def test_identical_findings_produce_same_fingerprint(self) -> None:
        f1 = _make_finding(title="CrashLoop", category="pod-health", affected_resources=["pod/my-pod"])
        f2 = _make_finding(title="CrashLoop", category="pod-health", affected_resources=["pod/my-pod"])
        assert _finding_fingerprint(f1) == _finding_fingerprint(f2)

    def test_case_insensitive_title(self) -> None:
        f1 = _make_finding(title="CrashLoop")
        f2 = _make_finding(title="crashloop")
        assert _finding_fingerprint(f1) == _finding_fingerprint(f2)

    def test_whitespace_stripped_title(self) -> None:
        f1 = _make_finding(title="  CrashLoop  ")
        f2 = _make_finding(title="CrashLoop")
        assert _finding_fingerprint(f1) == _finding_fingerprint(f2)

    def test_different_titles_produce_different_fingerprints(self) -> None:
        f1 = _make_finding(title="CrashLoop")
        f2 = _make_finding(title="OOMKilled")
        assert _finding_fingerprint(f1) != _finding_fingerprint(f2)

    def test_different_categories_produce_different_fingerprints(self) -> None:
        f1 = _make_finding(category="pod-health")
        f2 = _make_finding(category="networking")
        assert _finding_fingerprint(f1) != _finding_fingerprint(f2)

    def test_resource_kind_and_name_extracted(self) -> None:
        f1 = _make_finding(affected_resources=["pod/payment-svc-abc"])
        f2 = _make_finding(affected_resources=["deployment/payment-svc"])
        # Same title/category but different resource type → different fingerprints
        assert _finding_fingerprint(f1) != _finding_fingerprint(f2)

    def test_no_affected_resources(self) -> None:
        f = _make_finding(affected_resources=[])
        fp = _finding_fingerprint(f)
        assert isinstance(fp, str) and len(fp) == 40  # noqa: PLR2004

    def test_resource_without_slash(self) -> None:
        """A resource string without '/' is treated as kind-only, name is empty."""
        f = _make_finding(affected_resources=["payment-svc"])
        fp = _finding_fingerprint(f)
        assert isinstance(fp, str) and len(fp) == 40  # noqa: PLR2004

    def test_only_first_resource_used(self) -> None:
        f1 = _make_finding(affected_resources=["pod/a", "pod/b"])
        f2 = _make_finding(affected_resources=["pod/a", "pod/c"])
        assert _finding_fingerprint(f1) == _finding_fingerprint(f2)


# ── _dedup_findings ───────────────────────────────────────────────────────────


class TestDedupFindings:
    def test_empty_list_returns_empty(self) -> None:
        assert _dedup_findings([]) == []

    def test_no_duplicates_unchanged(self) -> None:
        findings = [
            _make_finding(title="A", fid="a"),
            _make_finding(title="B", fid="b"),
        ]
        result = _dedup_findings(findings)
        assert len(result) == 2  # noqa: PLR2004

    def test_exact_duplicate_removed(self) -> None:
        f = _make_finding(title="CrashLoop", category="pod-health", fid="dup")
        findings = [f, f]
        result = _dedup_findings(findings)
        assert len(result) == 1

    def test_first_occurrence_wins(self) -> None:
        f1 = _make_finding(title="CrashLoop", category="pod-health", fid="first")
        f2 = _make_finding(title="CrashLoop", category="pod-health", fid="second")
        result = _dedup_findings([f1, f2])
        assert result[0].id == "first"

    def test_original_list_not_mutated(self) -> None:
        f = _make_finding(title="CrashLoop", fid="dup")
        original = [f, f]
        _dedup_findings(original)
        assert len(original) == 2  # noqa: PLR2004

    def test_multiple_duplicates_all_removed(self) -> None:
        f = _make_finding(title="CrashLoop", fid="x")
        findings = [f, f, f, _make_finding(title="OOM", fid="y")]
        result = _dedup_findings(findings)
        assert len(result) == 2  # noqa: PLR2004

    def test_case_insensitive_dedup(self) -> None:
        f1 = _make_finding(title="crashloop", fid="a")
        f2 = _make_finding(title="CrashLoop", fid="b")
        result = _dedup_findings([f1, f2])
        assert len(result) == 1

    def test_different_resource_not_deduped(self) -> None:
        f1 = _make_finding(title="CrashLoop", affected_resources=["pod/svc-a"], fid="a")
        f2 = _make_finding(title="CrashLoop", affected_resources=["pod/svc-b"], fid="b")
        result = _dedup_findings([f1, f2])
        assert len(result) == 2  # noqa: PLR2004
