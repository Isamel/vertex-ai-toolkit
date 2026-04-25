"""Tests for detect_file_kind() — Helm template detection (Bug A fix)."""

from __future__ import annotations

from vaig.core.repo_chunkers import FallbackLineChunker, chunk_file, get_chunker
from vaig.core.repo_pipeline import detect_file_kind

# ── Helm template detection ────────────────────────────────────────────────────


def test_helm_template_with_values_classified_as_helm_template() -> None:
    """A YAML-looking file with {{ }} Go template syntax → helm_template."""
    content = "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: {{ .Values.replicaCount }}\n"
    assert detect_file_kind(content) == "helm_template"


def test_helm_template_does_not_trigger_yaml_chunker_error() -> None:
    """Helm template files should not produce ChunkingError log warnings."""
    content = "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: {{ .Values.replicaCount }}\n"
    kind = detect_file_kind(content)
    # Should be chunked without error — helm_template → FallbackLineChunker
    chunks = chunk_file(content, "templates/deployment.yaml", kind)
    assert len(chunks) > 0


def test_pure_k8s_manifest_not_misclassified_as_helm() -> None:
    """A plain k8s manifest with no {{ }} should stay k8s_manifest."""
    content = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: my-app\n"
    # No {{ }} → should still be k8s_manifest, not helm_template
    assert detect_file_kind(content) == "k8s_manifest"


def test_helm_template_starting_with_directive() -> None:
    """A Helm template that starts with {{- should be classified as helm_template."""
    content = "{{- if .Values.enabled }}\napiVersion: apps/v1\n{{- end }}\n"
    assert detect_file_kind(content) == "helm_template"


def test_helm_template_uses_fallback_chunker() -> None:
    """get_chunker('helm_template') should return FallbackLineChunker."""
    chunker = get_chunker("helm_template")
    assert isinstance(chunker, FallbackLineChunker)


def test_helm_template_multiple_braces() -> None:
    """Various Helm template patterns should all be detected."""
    content = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: {{ .Release.Name }}-config\n"
        "  namespace: {{ .Values.namespace | default \"default\" }}\n"
    )
    assert detect_file_kind(content) == "helm_template"


def test_argocd_without_templates_still_argocd() -> None:
    """ArgoCD manifests without {{ }} stay as argocd."""
    content = "apiVersion: argoproj.io/v1alpha1\nkind: Application\n"
    assert detect_file_kind(content) == "argocd"


def test_istio_without_templates_still_istio() -> None:
    """Istio manifests without {{ }} stay as istio_crd."""
    content = "apiVersion: networking.istio.io/v1beta1\nkind: VirtualService\n"
    assert detect_file_kind(content) == "istio_crd"
