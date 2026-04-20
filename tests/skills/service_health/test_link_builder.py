"""Tests for link_builder.py — pure URL generation functions.

Covers:
- All context present → correct link with correct URL
- Each required key missing → link omitted, no exception
- Empty context → empty list
- build_external_links aggregate builder
"""

from __future__ import annotations

import pytest

from vaig.skills.service_health.link_builder import (
    LinkContext,
    build_argocd_links,
    build_datadog_links,
    build_external_links,
    build_gcp_links,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def full_ctx() -> LinkContext:
    return {
        "project_id": "my-project",
        "cluster": "prod-cluster",
        "namespace": "payments",
        "service": "payment-svc",
        "datadog_org": "myorg",
        "argocd_server": "argocd.example.com",
        "argocd_app": "payment-svc",
    }


# ── build_gcp_links ───────────────────────────────────────────


class TestBuildGcpLinks:
    def test_full_context_returns_three_links(self, full_ctx: LinkContext) -> None:
        links = build_gcp_links(full_ctx)
        assert len(links) == 3
        labels = {link.label for link in links}
        assert "GCP Logs Explorer" in labels
        assert "GCP Monitoring" in labels
        assert "GKE Workloads" in labels

    def test_all_links_have_gcp_system(self, full_ctx: LinkContext) -> None:
        for link in build_gcp_links(full_ctx):
            assert link.system == "gcp"

    def test_logs_url_contains_project_id(self, full_ctx: LinkContext) -> None:
        links = build_gcp_links(full_ctx)
        logs_link = next(l for l in links if l.label == "GCP Logs Explorer")
        assert "my-project" in logs_link.url

    def test_monitoring_url_contains_project_id(self, full_ctx: LinkContext) -> None:
        links = build_gcp_links(full_ctx)
        mon_link = next(l for l in links if l.label == "GCP Monitoring")
        assert "my-project" in mon_link.url

    def test_gke_url_contains_cluster_and_namespace(self, full_ctx: LinkContext) -> None:
        links = build_gcp_links(full_ctx)
        gke_link = next(l for l in links if l.label == "GKE Workloads")
        assert "prod-cluster" in gke_link.url or "prod" in gke_link.url
        assert "payments" in gke_link.url

    def test_missing_project_id_returns_empty(self) -> None:
        links = build_gcp_links({"cluster": "c", "namespace": "n"})
        assert links == []

    def test_missing_cluster_omits_gke_link(self, full_ctx: LinkContext) -> None:
        ctx = {**full_ctx}
        del ctx["cluster"]
        links = build_gcp_links(ctx)
        labels = {l.label for l in links}
        assert "GKE Workloads" not in labels
        assert "GCP Logs Explorer" in labels

    def test_missing_namespace_omits_gke_link(self, full_ctx: LinkContext) -> None:
        ctx = {**full_ctx}
        del ctx["namespace"]
        links = build_gcp_links(ctx)
        labels = {l.label for l in links}
        assert "GKE Workloads" not in labels

    def test_empty_context_returns_empty(self) -> None:
        assert build_gcp_links({}) == []

    def test_links_have_icon(self, full_ctx: LinkContext) -> None:
        for link in build_gcp_links(full_ctx):
            assert link.icon != ""


# ── build_datadog_links ───────────────────────────────────────


class TestBuildDatadogLinks:
    def test_full_context_returns_two_links(self, full_ctx: LinkContext) -> None:
        links = build_datadog_links(full_ctx)
        assert len(links) == 2

    def test_all_links_have_datadog_system(self, full_ctx: LinkContext) -> None:
        for link in build_datadog_links(full_ctx):
            assert link.system == "datadog"

    def test_apm_url_contains_service_name(self, full_ctx: LinkContext) -> None:
        links = build_datadog_links(full_ctx)
        apm = next(l for l in links if "APM" in l.label)
        assert "payment-svc" in apm.url

    def test_dashboard_link_present(self, full_ctx: LinkContext) -> None:
        links = build_datadog_links(full_ctx)
        labels = {l.label for l in links}
        assert "Datadog Dashboards" in labels

    def test_missing_datadog_org_returns_empty(self) -> None:
        links = build_datadog_links({"service": "my-svc"})
        assert links == []

    def test_missing_service_omits_apm_link(self, full_ctx: LinkContext) -> None:
        ctx = {**full_ctx}
        del ctx["service"]
        links = build_datadog_links(ctx)
        labels = {l.label for l in links}
        assert not any("APM" in label for label in labels)
        assert "Datadog Dashboards" in labels

    def test_empty_context_returns_empty(self) -> None:
        assert build_datadog_links({}) == []

    def test_links_have_icon(self, full_ctx: LinkContext) -> None:
        for link in build_datadog_links(full_ctx):
            assert link.icon != ""


# ── build_argocd_links ────────────────────────────────────────


class TestBuildArgoCDLinks:
    def test_full_context_returns_one_link(self, full_ctx: LinkContext) -> None:
        links = build_argocd_links(full_ctx)
        assert len(links) == 1
        assert links[0].system == "argocd"

    def test_url_contains_app_name(self, full_ctx: LinkContext) -> None:
        links = build_argocd_links(full_ctx)
        assert "payment-svc" in links[0].url

    def test_url_contains_server(self, full_ctx: LinkContext) -> None:
        links = build_argocd_links(full_ctx)
        assert "argocd.example.com" in links[0].url

    def test_server_without_scheme_gets_https(self) -> None:
        ctx = {"argocd_server": "argocd.example.com", "argocd_app": "myapp"}
        links = build_argocd_links(ctx)
        assert links[0].url.startswith("https://")

    def test_server_with_existing_scheme_kept(self) -> None:
        ctx = {"argocd_server": "https://argocd.example.com", "argocd_app": "myapp"}
        links = build_argocd_links(ctx)
        # Should not double-up the scheme
        assert links[0].url.count("https://") == 1

    def test_missing_argocd_server_returns_empty(self) -> None:
        links = build_argocd_links({"argocd_app": "myapp"})
        assert links == []

    def test_missing_argocd_app_returns_empty(self) -> None:
        links = build_argocd_links({"argocd_server": "argocd.example.com"})
        assert links == []

    def test_empty_context_returns_empty(self) -> None:
        assert build_argocd_links({}) == []

    def test_link_has_icon(self, full_ctx: LinkContext) -> None:
        links = build_argocd_links(full_ctx)
        assert links[0].icon != ""


# ── build_external_links ──────────────────────────────────────


class TestBuildExternalLinks:
    def test_full_context_populates_all_groups(self, full_ctx: LinkContext) -> None:
        el = build_external_links(full_ctx)
        assert len(el.gcp) == 3
        assert len(el.datadog) == 2
        assert len(el.argocd) == 1

    def test_empty_context_returns_all_empty_groups(self) -> None:
        el = build_external_links({})
        assert el.gcp == []
        assert el.datadog == []
        assert el.argocd == []

    def test_partial_context_only_gcp(self) -> None:
        ctx: LinkContext = {"project_id": "my-project"}
        el = build_external_links(ctx)
        # Logs + Monitoring but NOT GKE (missing cluster/namespace)
        assert len(el.gcp) == 2
        assert el.datadog == []
        assert el.argocd == []

    def test_never_raises(self) -> None:
        """build_external_links must never raise, even with garbage input."""
        garbage: LinkContext = {"project_id": "", "service": "   "}
        el = build_external_links(garbage)
        assert el is not None
