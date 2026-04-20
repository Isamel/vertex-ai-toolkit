"""Tests for SPEC-V2-REPO-02 — Tiered file-processing pipeline.

Acceptance criteria:
1. Unit test of the classifier against a 50-file fixture (parametrized).
2. Each classifier rule fires correctly.
3. File kind detection for each supported kind.
4. Binary detection.
5. exclude_globs work.
6. catastrophic_size skip works.
7. streaming threshold triggers chunked_streaming outcome.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vaig.core.config import RepoInvestigationConfig
from vaig.core.repo_pipeline import (
    ClassificationResult,
    FileMeta,
    Relevance,
    TierOutcome,
    classify_file,
    detect_file_kind,
    is_binary_file,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

DEFAULT_CFG = RepoInvestigationConfig()


def _meta(
    path: str,
    size: int = 1024,
    kind: str = "yaml",
    is_binary: bool = False,
) -> FileMeta:
    return FileMeta(path=path, size=size, kind=kind, is_binary=is_binary)


def _classify(
    path: str,
    size: int = 1024,
    kind: str = "yaml",
    is_binary: bool = False,
    cfg: RepoInvestigationConfig | None = None,
) -> ClassificationResult:
    return classify_file(_meta(path, size, kind, is_binary), cfg or DEFAULT_CFG)


# ── 50-file fixture (parametrized) ────────────────────────────────────────────

# Each tuple: (path, size, kind, is_binary, expected_outcome, expected_reason_prefix)
_FIFTY_FILE_FIXTURE: list[
    tuple[str, int, str, bool, TierOutcome, str]
] = [
    # Binary files → skip
    ("app.bin", 100, "unknown", True, TierOutcome.SKIP, "binary"),
    ("image.png", 50_000, "unknown", True, TierOutcome.SKIP, "binary"),
    ("font.ttf", 200_000, "unknown", True, TierOutcome.SKIP, "binary"),
    # Catastrophic size → skip
    ("huge.yaml", 600_000_000, "yaml", False, TierOutcome.SKIP, "catastrophic_size"),
    ("gigantic.txt", 501_000_000, "text", False, TierOutcome.SKIP, "catastrophic_size"),
    # Streaming threshold
    ("big.yaml", 3_000_000, "yaml", False, TierOutcome.CHUNKED_STREAMING, "large_text_stream"),
    ("big.txt", 2_500_000, "text", False, TierOutcome.CHUNKED_STREAMING, "large_text_stream"),
    # Helm chart roots
    ("helm/Chart.yaml", 1024, "yaml", False, TierOutcome.CHUNKED, "helm_chart_root"),
    ("charts/myapp/Chart.yaml", 1024, "yaml", False, TierOutcome.CHUNKED, "helm_chart_root"),
    ("Chart.yaml", 1024, "yaml", False, TierOutcome.CHUNKED, "helm_chart_root"),
    # Chart.lock is excluded by default config's **/*.lock glob before helm_chart_root fires
    ("charts/foo/Chart.lock", 512, "yaml", False, TierOutcome.SKIP, "exclude_glob"),
    # Helm values
    ("helm/values.yaml", 2048, "yaml", False, TierOutcome.CHUNKED, "helm_values"),
    ("helm/values.yml", 2048, "yaml", False, TierOutcome.CHUNKED, "helm_values"),
    ("helm/values-prod.yaml", 2048, "yaml", False, TierOutcome.CHUNKED, "helm_values"),
    ("values.yaml", 2048, "yaml", False, TierOutcome.CHUNKED, "helm_values"),
    # Istio routing — by kind
    ("networking/vs.yaml", 1024, "istio_crd", False, TierOutcome.CHUNKED, "istio_routing"),
    # Istio routing — by path
    ("networking/virtualservice-frontend.yaml", 1024, "yaml", False, TierOutcome.CHUNKED, "istio_routing"),
    ("networking/virtualservice-backend.yml", 1024, "yaml", False, TierOutcome.CHUNKED, "istio_routing"),
    # ArgoCD app — only when kind == argocd
    ("argocd/Application-prod.yaml", 1024, "argocd", False, TierOutcome.CHUNKED, "argocd_app"),
    ("argocd/Application.yml", 1024, "argocd", False, TierOutcome.CHUNKED, "argocd_app"),
    # ArgoCD path match but wrong kind → not argocd_app, falls to k8s_manifest or default
    ("argocd/Application-prod.yaml", 1024, "yaml", False, TierOutcome.HEADER, "default"),
    # k8s manifest by kind
    ("deploy/deployment.yaml", 1024, "k8s_manifest", False, TierOutcome.CHUNKED, "k8s_manifest"),
    ("k8s/service.yaml", 1024, "k8s_manifest", False, TierOutcome.CHUNKED, "k8s_manifest"),
    # kustomization by kind
    ("overlays/kustomization.yaml", 1024, "kustomization", False, TierOutcome.CHUNKED, "k8s_manifest"),
    # Helm templates — medium
    ("charts/app/templates/deployment.yaml", 2048, "yaml", False, TierOutcome.CHUNKED, "helm_template"),
    ("charts/app/templates/service.yml", 2048, "yaml", False, TierOutcome.CHUNKED, "helm_template"),
    # README — header
    ("README.md", 3000, "text", False, TierOutcome.HEADER, "readme"),
    ("docs/README.md", 3000, "text", False, TierOutcome.HEADER, "readme"),
    ("README.rst", 2000, "text", False, TierOutcome.HEADER, "readme"),
    # CHANGELOG — header
    ("CHANGELOG.md", 5000, "text", False, TierOutcome.HEADER, "changelog"),
    ("CHANGELOG", 3000, "text", False, TierOutcome.HEADER, "changelog"),
    ("path/to/CHANGELOG.md", 3000, "text", False, TierOutcome.HEADER, "changelog"),
    # Default — header
    ("src/main.py", 1024, "text", False, TierOutcome.HEADER, "default"),
    ("config/app.conf", 512, "text", False, TierOutcome.HEADER, "default"),
    ("Makefile", 2048, "text", False, TierOutcome.HEADER, "default"),
    ("scripts/deploy.sh", 1024, "text", False, TierOutcome.HEADER, "default"),
    ("docs/architecture.md", 4096, "text", False, TierOutcome.HEADER, "default"),
    ("terraform/main.tf", 1024, "terraform_gke", False, TierOutcome.HEADER, "default"),
    ("pyproject.toml", 1024, "text", False, TierOutcome.HEADER, "default"),
    ("go.mod", 512, "text", False, TierOutcome.HEADER, "default"),
    (".github/workflows/ci.yaml", 2048, "yaml", False, TierOutcome.HEADER, "default"),
    ("docker-compose.yml", 1024, "yaml", False, TierOutcome.HEADER, "default"),
    ("k8s/configmap.yaml", 512, "yaml", False, TierOutcome.HEADER, "default"),
    ("Dockerfile", 1024, "text", False, TierOutcome.HEADER, "default"),
    ("helm/templates/notes.txt", 512, "text", False, TierOutcome.HEADER, "default"),
    ("LICENSE", 10_000, "text", False, TierOutcome.HEADER, "default"),
    ("src/utils/helper.go", 2048, "text", False, TierOutcome.HEADER, "default"),
    ("infra/outputs.tf", 1024, "text", False, TierOutcome.HEADER, "default"),
    (".env.example", 512, "text", False, TierOutcome.HEADER, "default"),
    ("tests/fixtures/sample.json", 4096, "text", False, TierOutcome.HEADER, "default"),
]

assert len(_FIFTY_FILE_FIXTURE) == 50, f"Fixture has {len(_FIFTY_FILE_FIXTURE)} entries, expected 50"


@pytest.mark.parametrize(
    "path,size,kind,is_binary,expected_outcome,expected_reason_prefix",
    _FIFTY_FILE_FIXTURE,
    ids=[f[0] for f in _FIFTY_FILE_FIXTURE],
)
def test_classifier_50_file_fixture(
    path: str,
    size: int,
    kind: str,
    is_binary: bool,
    expected_outcome: TierOutcome,
    expected_reason_prefix: str,
) -> None:
    """AC-1: classifier produces expected outcome for each of the 50 fixture files."""
    result = _classify(path, size, kind, is_binary)
    assert result.outcome == expected_outcome, (
        f"{path!r}: expected {expected_outcome}, got {result.outcome} (reason={result.reason!r})"
    )
    assert result.reason == expected_reason_prefix or result.reason.startswith(expected_reason_prefix), (
        f"{path!r}: expected reason starting with {expected_reason_prefix!r}, got {result.reason!r}"
    )


# ── Individual rule tests ─────────────────────────────────────────────────────


class TestBinaryRule:
    """Rule: binary → skip."""

    def test_binary_meta_skips(self) -> None:
        result = _classify("app.bin", is_binary=True)
        assert result.outcome == TierOutcome.SKIP
        assert result.reason == "binary"

    def test_non_binary_not_skipped_by_binary_rule(self) -> None:
        result = _classify("file.yaml", is_binary=False)
        assert result.outcome != TierOutcome.SKIP or result.reason != "binary"


class TestExcludeGlobRule:
    """Rule: exclude_glob → skip."""

    def test_exclude_glob_skips(self) -> None:
        cfg = RepoInvestigationConfig(exclude_globs=["**/node_modules/**"])
        result = classify_file(_meta("frontend/node_modules/lodash/index.js"), cfg)
        assert result.outcome == TierOutcome.SKIP
        assert result.reason == "exclude_glob"

    def test_multiple_exclude_globs(self) -> None:
        cfg = RepoInvestigationConfig(exclude_globs=["**/.git/**", "**/vendor/**"])
        result = classify_file(_meta("project/vendor/pkg/main.go"), cfg)
        assert result.outcome == TierOutcome.SKIP
        assert result.reason == "exclude_glob"

    def test_non_excluded_file_not_skipped(self) -> None:
        cfg = RepoInvestigationConfig(exclude_globs=["**/node_modules/**"])
        result = classify_file(_meta("src/app.ts"), cfg)
        assert result.reason != "exclude_glob"

    def test_default_exclude_globs_skip_dotgit(self) -> None:
        """**/.git/** pattern correctly excludes .git internals."""
        cfg = RepoInvestigationConfig(exclude_globs=["**/.git/**"])
        result = classify_file(_meta(".git/config"), cfg)
        assert result.outcome == TierOutcome.SKIP


class TestCatastrophicSizeRule:
    """Rule: catastrophic_size → skip."""

    def test_above_max_bytes_skips(self) -> None:
        cfg = RepoInvestigationConfig(max_file_bytes_absolute=500_000_000)
        result = classify_file(_meta("huge.yaml", size=500_000_001), cfg)
        assert result.outcome == TierOutcome.SKIP
        assert result.reason == "catastrophic_size"

    def test_exactly_at_max_not_skipped(self) -> None:
        cfg = RepoInvestigationConfig(max_file_bytes_absolute=500_000_000)
        result = classify_file(_meta("file.yaml", size=500_000_000), cfg)
        assert result.reason != "catastrophic_size"

    def test_custom_max_file_bytes(self) -> None:
        cfg = RepoInvestigationConfig(max_file_bytes_absolute=1024)
        result = classify_file(_meta("file.yaml", size=1025), cfg)
        assert result.outcome == TierOutcome.SKIP
        assert result.reason == "catastrophic_size"


class TestStreamingThresholdRule:
    """Rule: large text → chunked_streaming."""

    def test_above_streaming_threshold(self) -> None:
        cfg = RepoInvestigationConfig(streaming_threshold_bytes=2_000_000)
        result = classify_file(_meta("big.yaml", size=2_000_001), cfg)
        assert result.outcome == TierOutcome.CHUNKED_STREAMING
        assert result.reason == "large_text_stream"
        assert result.relevance == Relevance.AUTO

    def test_exactly_at_streaming_threshold_not_streaming(self) -> None:
        cfg = RepoInvestigationConfig(streaming_threshold_bytes=2_000_000)
        result = classify_file(_meta("file.yaml", size=2_000_000), cfg)
        assert result.reason != "large_text_stream"

    def test_custom_streaming_threshold(self) -> None:
        cfg = RepoInvestigationConfig(streaming_threshold_bytes=100)
        result = classify_file(_meta("file.yaml", size=101), cfg)
        assert result.outcome == TierOutcome.CHUNKED_STREAMING


class TestHelmChartRootRule:
    """Rule: Chart.yaml / Chart.lock → chunked, high."""

    # Note: Chart.lock needs a config without the **/*.lock exclude glob,
    # since the default config excludes all *.lock files before this rule fires.
    _cfg_no_lock_exclude = RepoInvestigationConfig(exclude_globs=[])

    @pytest.mark.parametrize("path", [
        "Chart.yaml",
        "helm/Chart.yaml",
        "charts/myapp/Chart.yaml",
    ])
    def test_chart_yaml_files(self, path: str) -> None:
        result = _classify(path)
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.HIGH
        assert result.reason == "helm_chart_root"

    @pytest.mark.parametrize("path", [
        "Chart.lock",
        "charts/foo/Chart.lock",
    ])
    def test_chart_lock_files_without_lock_exclude(self, path: str) -> None:
        """Chart.lock fires helm_chart_root when *.lock is not excluded."""
        result = classify_file(_meta(path), self._cfg_no_lock_exclude)
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.HIGH
        assert result.reason == "helm_chart_root"

    def test_chart_lock_excluded_by_default_config(self) -> None:
        """With default config, Chart.lock is excluded by **/*.lock glob."""
        result = _classify("Chart.lock")
        assert result.outcome == TierOutcome.SKIP
        assert result.reason == "exclude_glob"


class TestHelmValuesRule:
    """Rule: values*.y*ml → chunked, high."""

    @pytest.mark.parametrize("path", [
        "values.yaml",
        "helm/values.yaml",
        "helm/values.yml",
        "helm/values-prod.yaml",
        "helm/values-staging.yml",
    ])
    def test_helm_values_files(self, path: str) -> None:
        result = _classify(path)
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.HIGH
        assert result.reason == "helm_values"


class TestIstioRoutingRule:
    """Rule: istio_crd kind or virtualservice path → chunked, high."""

    def test_istio_crd_kind(self) -> None:
        result = _classify("networking/vs.yaml", kind="istio_crd")
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.HIGH
        assert result.reason == "istio_routing"

    @pytest.mark.parametrize("path", [
        "networking/virtualservice-frontend.yaml",
        "k8s/virtualservice-backend.yml",
        "virtualservice.yaml",
    ])
    def test_virtualservice_path(self, path: str) -> None:
        result = _classify(path)
        assert result.outcome == TierOutcome.CHUNKED
        assert result.reason == "istio_routing"


class TestArgoCDAppRule:
    """Rule: Application*.y*ml + kind==argocd → chunked, high."""

    def test_argocd_app_yaml(self) -> None:
        result = _classify("argocd/Application-prod.yaml", kind="argocd")
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.HIGH
        assert result.reason == "argocd_app"

    def test_argocd_path_but_wrong_kind_not_matched(self) -> None:
        """Path matches but kind != argocd → rule must not fire."""
        result = _classify("argocd/Application.yaml", kind="yaml")
        assert result.reason != "argocd_app"


class TestK8sManifestRule:
    """Rule: k8s_manifest / kustomization kind → chunked, med."""

    @pytest.mark.parametrize("kind", ["k8s_manifest", "kustomization"])
    def test_k8s_kinds(self, kind: str) -> None:
        result = _classify("k8s/resource.yaml", kind=kind)
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.MEDIUM
        assert result.reason == "k8s_manifest"


class TestHelmTemplateRule:
    """Rule: templates/*.y*ml → chunked, med."""

    @pytest.mark.parametrize("path", [
        "charts/app/templates/deployment.yaml",
        "charts/app/templates/service.yml",
        "templates/ingress.yaml",
    ])
    def test_helm_templates(self, path: str) -> None:
        result = _classify(path)
        assert result.outcome == TierOutcome.CHUNKED
        assert result.relevance == Relevance.MEDIUM
        assert result.reason == "helm_template"


class TestReadmeRule:
    """Rule: README* → header, low."""

    @pytest.mark.parametrize("path", [
        "README.md",
        "README.rst",
        "README",
        "docs/README.md",
        "subdir/README.txt",
    ])
    def test_readme_files(self, path: str) -> None:
        result = _classify(path)
        assert result.outcome == TierOutcome.HEADER
        assert result.reason == "readme"


class TestChangelogRule:
    """Rule: CHANGELOG* → header, low."""

    @pytest.mark.parametrize("path", [
        "CHANGELOG.md",
        "CHANGELOG",
        "CHANGELOG.txt",
        "path/CHANGELOG.md",
    ])
    def test_changelog_files(self, path: str) -> None:
        result = _classify(path)
        assert result.outcome == TierOutcome.HEADER
        assert result.reason == "changelog"


class TestDefaultRule:
    """Default rule: anything else → header, low."""

    def test_default_plain_python(self) -> None:
        result = _classify("src/main.py")
        assert result.outcome == TierOutcome.HEADER
        assert result.reason == "default"

    def test_default_makefile(self) -> None:
        result = _classify("Makefile")
        assert result.outcome == TierOutcome.HEADER
        assert result.reason == "default"


# ── File kind detection tests ─────────────────────────────────────────────────


class TestDetectFileKind:
    """detect_file_kind() returns correct kind for each supported pattern."""

    def test_argocd(self) -> None:
        peek = "apiVersion: argoproj.io/v1alpha1\nkind: Application\n"
        assert detect_file_kind(peek) == "argocd"

    def test_istio_crd(self) -> None:
        peek = "apiVersion: networking.istio.io/v1alpha3\nkind: VirtualService\n"
        assert detect_file_kind(peek) == "istio_crd"

    def test_k8s_manifest_apps_v1(self) -> None:
        peek = "apiVersion: apps/v1\nkind: Deployment\n"
        assert detect_file_kind(peek) == "k8s_manifest"

    def test_k8s_manifest_v1(self) -> None:
        peek = "apiVersion: v1\nkind: Service\n"
        assert detect_file_kind(peek) == "k8s_manifest"

    def test_k8s_manifest_batch_v1(self) -> None:
        peek = "apiVersion: batch/v1\nkind: Job\n"
        assert detect_file_kind(peek) == "k8s_manifest"

    def test_kustomization(self) -> None:
        peek = "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\n"
        assert detect_file_kind(peek) == "kustomization"

    def test_terraform_gke(self) -> None:
        peek = 'resource "google_container_cluster" "primary" {\n'
        assert detect_file_kind(peek) == "terraform_gke"

    def test_generic_yaml(self) -> None:
        peek = "foo: bar\nbaz: 123\n"
        assert detect_file_kind(peek) == "yaml"

    def test_plain_text(self) -> None:
        peek = "Hello world\nThis is a plain text file.\n"
        assert detect_file_kind(peek) == "text"

    def test_empty_string(self) -> None:
        assert detect_file_kind("") == "text"

    def test_priority_argocd_over_k8s(self) -> None:
        """ArgoCD must win if both argocd and k8s signals are present."""
        peek = "apiVersion: argoproj.io/v1alpha1\napiVersion: apps/v1\n"
        assert detect_file_kind(peek) == "argocd"


# ── Binary detection tests ────────────────────────────────────────────────────


class TestIsBinaryFile:
    """is_binary_file() correctly identifies binary vs text files."""

    def _write_tmp(self, content: bytes) -> Path:
        import os
        fd, name = tempfile.mkstemp(suffix=".dat")
        os.write(fd, content)
        os.close(fd)
        return Path(name)

    def test_text_file_not_binary(self) -> None:
        path = self._write_tmp(b"hello world\nthis is text\n")
        assert is_binary_file(path) is False

    def test_null_byte_is_binary(self) -> None:
        path = self._write_tmp(b"hello\x00world")
        assert is_binary_file(path) is True

    def test_png_header_is_binary(self) -> None:
        # PNG magic bytes
        path = self._write_tmp(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert is_binary_file(path) is True

    def test_yaml_text_file_not_binary(self) -> None:
        path = self._write_tmp(b"apiVersion: v1\nkind: Service\nmetadata:\n  name: foo\n")
        assert is_binary_file(path) is False

    def test_empty_file_not_binary(self) -> None:
        path = self._write_tmp(b"")
        assert is_binary_file(path) is False

    def test_high_non_text_ratio_is_binary(self) -> None:
        # Build content with >30% non-printable bytes (but no null byte)
        non_text = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] * 50)
        text = b"hello world " * 10  # 120 bytes text
        content = non_text + text  # 520 bytes, ~77% non-text
        path = self._write_tmp(content)
        assert is_binary_file(path) is True

    def test_nonexistent_file_returns_false(self) -> None:
        path = Path("/tmp/does_not_exist_xyz_12345.bin")
        assert is_binary_file(path) is False


# ── ClassificationResult model tests ─────────────────────────────────────────


class TestClassificationResult:
    """ClassificationResult model validation."""

    def test_result_contains_file_meta(self) -> None:
        meta = _meta("Chart.yaml")
        result = classify_file(meta, DEFAULT_CFG)
        assert result.file == meta

    def test_result_has_outcome_and_reason(self) -> None:
        result = _classify("Chart.yaml")
        assert isinstance(result.outcome, TierOutcome)
        assert isinstance(result.reason, str)
        assert result.reason != ""

    def test_result_relevance_default_is_low(self) -> None:
        result = _classify("random.txt")
        assert result.relevance == Relevance.LOW


# ── Invariant: every file gets exactly one outcome ────────────────────────────


def test_every_file_gets_exactly_one_outcome() -> None:
    """Every FileMeta must match exactly one outcome (no exceptions)."""
    test_files = [
        _meta(path, size, kind, is_binary)
        for path, size, kind, is_binary, *_ in _FIFTY_FILE_FIXTURE
    ]
    cfg = DEFAULT_CFG
    for meta in test_files:
        result = classify_file(meta, cfg)
        assert result is not None
        assert result.outcome in list(TierOutcome)
        assert result.reason != ""
