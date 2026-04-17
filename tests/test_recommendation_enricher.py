"""Unit tests for the recommendation enrichment module.

Tests cover successful enrichment, finding matching, invalid/empty responses,
timeout handling, and early-return when there are no recommendations.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from vaig.skills.service_health.recommendation_enricher import (
    _build_enrichment_prompt,
    enrich_recommendations,
)
from vaig.skills.service_health.schema import (
    ActionUrgency,
    Confidence,
    ExecutiveSummary,
    Finding,
    HealthReport,
    RecommendedAction,
    ReportMetadata,
    ServiceStatus,
    Severity,
)


@dataclass
class FakeGenerateResult:
    """Mimics the result from GeminiClient.async_generate."""

    text: str


def _make_stub_client(response_text: str) -> AsyncMock:
    """Create a stub client that returns the given text from async_generate."""
    client = AsyncMock()
    client.async_generate = AsyncMock(
        return_value=FakeGenerateResult(text=response_text)
    )
    # async_initialize should be a no-op
    client.async_initialize = AsyncMock()
    return client


def _make_finding(
    finding_id: str = "test-finding-1",
    title: str = "Pod CrashLoopBackOff",
    severity: Severity = Severity.HIGH,
) -> Finding:
    return Finding(
        id=finding_id,
        title=title,
        severity=severity,
        description="Pod is crash looping",
        root_cause="Application startup failure",
        evidence=["Error: connection refused to db:5432"],
        confidence=Confidence.HIGH,
        impact="Service degraded",
        affected_resources=["production/deployment/api-gateway"],
    )


def _make_action(
    related_findings: list[str] | None = None,
) -> RecommendedAction:
    return RecommendedAction(
        priority=1,
        title="Check pod logs",
        description="Inspect logs for crash reason",
        urgency=ActionUrgency.IMMEDIATE,
        command="kubectl logs -n production deployment/api-gateway --previous --tail=50",
        expected_output="original expected output",
        interpretation="original interpretation",
        why="To identify crash root cause",
        risk="None — read-only",
        related_findings=related_findings or ["test-finding-1"],
    )


def _make_report(
    findings: list[Finding] | None = None,
    recommendations: list[RecommendedAction] | None = None,
) -> HealthReport:
    """Create a minimal HealthReport for testing."""
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status="DEGRADED",
            scope="test",
            summary_text="test summary",
        ),
        findings=findings if findings is not None else [_make_finding()],
        recommendations=recommendations if recommendations is not None else [_make_action()],
        root_cause_hypotheses=[],
        timeline=[],
        metadata={},
    )


class TestEnrichRecommendations:
    @pytest.mark.asyncio
    async def test_successful_enrichment(self):
        """Stub client returns valid JSON — recommendations should be updated."""
        response = json.dumps(
            {
                "expected_output": "NAME         READY   STATUS    RESTARTS\napi-gw-abc   0/1     CrashLoop 5",
                "interpretation": "Look at STATUS column. If CrashLoopBackOff → check logs next.",
            }
        )
        client = _make_stub_client(response)
        report = _make_report()

        result = await enrich_recommendations(report, client, model="test-model")

        assert result.recommendations[0].expected_output.startswith("NAME")
        assert "CrashLoop" in result.recommendations[0].expected_output
        assert "STATUS column" in result.recommendations[0].interpretation

    @pytest.mark.asyncio
    async def test_related_findings_matching(self):
        """The correct finding should be matched via related_findings IDs."""
        finding1 = _make_finding(finding_id="f1", title="OOMKilled")
        finding2 = _make_finding(finding_id="f2", title="CrashLoop")
        action = _make_action(related_findings=["f2"])

        response = json.dumps(
            {
                "expected_output": "enriched output",
                "interpretation": "enriched interpretation",
            }
        )
        client = _make_stub_client(response)
        report = _make_report(
            findings=[finding1, finding2], recommendations=[action]
        )

        await enrich_recommendations(report, client, model="test-model")

        # Verify the prompt was built with finding2, not finding1
        call_args = client.async_generate.call_args
        prompt = call_args[0][0]
        assert "CrashLoop" in prompt
        assert "OOMKilled" not in prompt

    @pytest.mark.asyncio
    async def test_invalid_json_preserves_original(self):
        """Invalid JSON response should preserve original values."""
        client = _make_stub_client("this is not valid json at all")
        report = _make_report()

        result = await enrich_recommendations(report, client, model="test-model")

        assert result.recommendations[0].expected_output == "original expected output"
        assert (
            result.recommendations[0].interpretation == "original interpretation"
        )

    @pytest.mark.asyncio
    async def test_empty_fields_preserves_original(self):
        """Empty enrichment fields should preserve original values."""
        response = json.dumps(
            {
                "expected_output": "",
                "interpretation": "",
            }
        )
        client = _make_stub_client(response)
        report = _make_report()

        result = await enrich_recommendations(report, client, model="test-model")

        assert result.recommendations[0].expected_output == "original expected output"
        assert (
            result.recommendations[0].interpretation == "original interpretation"
        )

    @pytest.mark.asyncio
    async def test_timeout_preserves_original(self):
        """Timeout should preserve original values."""

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)
            return FakeGenerateResult(text="{}")

        client = AsyncMock()
        client.async_generate = slow_generate
        client.async_initialize = AsyncMock()
        report = _make_report()

        result = await enrich_recommendations(
            report, client, model="test-model", timeout_per_call=0.1
        )

        assert result.recommendations[0].expected_output == "original expected output"

    @pytest.mark.asyncio
    async def test_no_recommendations_early_return(self):
        """Report with no recommendations should return immediately."""
        client = _make_stub_client("should not be called")
        report = _make_report(recommendations=[])

        result = await enrich_recommendations(report, client, model="test-model")

        client.async_generate.assert_not_called()
        assert result.recommendations == []

    @pytest.mark.asyncio
    async def test_non_dict_json_preserves_original(self):
        """Non-dict JSON (e.g., array) should preserve original values."""
        client = _make_stub_client("[1, 2, 3]")
        report = _make_report()

        result = await enrich_recommendations(report, client, model="test-model")

        assert result.recommendations[0].expected_output == "original expected output"


class TestBuildEnrichmentPrompt:
    """Tests for _build_enrichment_prompt — context wiring and prompt construction."""

    def test_context_section_included_when_context_provided(self):
        """Prompt should contain a '## Context' section with namespace and cluster."""
        finding = _make_finding()
        action = _make_action()
        context = {"namespace": "production", "cluster_name": "gke-us-east1"}

        prompt = _build_enrichment_prompt(finding, action, context=context)

        assert "## Context" in prompt
        assert "**Namespace**: production" in prompt
        assert "**Cluster**: gke-us-east1" in prompt

    def test_context_section_omitted_when_no_context(self):
        """Prompt should NOT contain '## Context' when context is None."""
        finding = _make_finding()
        action = _make_action()

        prompt = _build_enrichment_prompt(finding, action, context=None)

        assert "## Context" not in prompt

    def test_prompt_includes_service_category_remediation(self):
        """Prompt should include service, category, and remediation from the finding."""
        finding = Finding(
            id="svc-finding",
            title="High CPU",
            severity=Severity.HIGH,
            description="CPU usage is excessive",
            root_cause="Inefficient loop",
            evidence=["cpu at 95%"],
            confidence=Confidence.HIGH,
            impact="Latency increase",
            affected_resources=["production/deployment/api"],
            service="payment-service",
            category="resource-usage",
            remediation="Increase CPU limit or optimize code",
        )
        action = _make_action(related_findings=["svc-finding"])

        prompt = _build_enrichment_prompt(finding, action, context=None)

        assert "**Service**: payment-service" in prompt
        assert "**Category**: resource-usage" in prompt
        assert "Increase CPU limit or optimize code" in prompt

    @pytest.mark.asyncio
    async def test_namespace_resolved_from_service_with_fallback(self):
        """Namespace should be resolved from finding.service via service_statuses,
        falling back to the first service's namespace when there's no match."""
        finding_matched = _make_finding(finding_id="f1", title="Matched")
        finding_matched.service = "api-gateway"

        finding_unmatched = _make_finding(finding_id="f2", title="Unmatched")
        finding_unmatched.service = "unknown-service"

        action_matched = _make_action(related_findings=["f1"])
        action_unmatched = _make_action(related_findings=["f2"])

        response = json.dumps({
            "expected_output": "enriched",
            "interpretation": "enriched",
        })
        client = _make_stub_client(response)

        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status="DEGRADED",
                scope="test",
                summary_text="test",
            ),
            findings=[finding_matched, finding_unmatched],
            recommendations=[action_matched, action_unmatched],
            root_cause_hypotheses=[],
            timeline=[],
            metadata=ReportMetadata(cluster_name="test-cluster"),
            service_statuses=[
                ServiceStatus(service="frontend", namespace="web-ns"),
                ServiceStatus(service="api-gateway", namespace="api-ns"),
            ],
        )

        await enrich_recommendations(report, client, model="test-model")

        # Two calls: one per recommendation
        assert client.async_generate.call_count == 2

        # First call should have api-ns (matched via service name)
        first_prompt = client.async_generate.call_args_list[0][0][0]
        assert "**Namespace**: api-ns" in first_prompt

        # Second call should fallback to web-ns (first service's namespace)
        second_prompt = client.async_generate.call_args_list[1][0][0]
        assert "**Namespace**: web-ns" in second_prompt


# ── Root-cause finding fallback (SH-03) ─────────────────────────────────────


def _make_finding_with_causal(
    finding_id: str,
    caused_by: list[str] | None = None,
) -> Finding:
    return Finding(
        id=finding_id,
        title=f"Finding {finding_id}",
        severity=Severity.HIGH,
        description="desc",
        root_cause="cause",
        evidence=[],
        confidence=Confidence.HIGH,
        impact="impact",
        affected_resources=[],
        caused_by=caused_by or [],
    )


def _make_action_no_related() -> RecommendedAction:
    """Action with no related_findings so the enricher must fall back."""
    return RecommendedAction(
        priority=1,
        title="Generic action",
        description="Do something",
        urgency=ActionUrgency.IMMEDIATE,
        command="kubectl get pods",
        expected_output="pods listed",
        interpretation="check status",
        why="diagnosing",
        risk="none",
        related_findings=[],  # explicitly empty — forces fallback
    )


class TestRootCauseFallback:
    """Verify the 3-step finding resolution fallback in enrich_recommendations."""

    @pytest.mark.asyncio
    async def test_falls_back_to_root_cause_finding_when_no_related(self):
        """Step 2: should pick the root-cause finding (caused_by=[]) over a child."""
        root = _make_finding_with_causal("root-f", caused_by=[])
        child = _make_finding_with_causal("child-f", caused_by=["root-f"])

        enriched_json = json.dumps(
            {
                "expected_output": "new expected",
                "interpretation": "new interpretation",
            }
        )
        client = _make_stub_client(enriched_json)

        report = _make_report(
            findings=[root, child],
            recommendations=[_make_action_no_related()],
        )

        await enrich_recommendations(report, client)

        # The enrichment prompt should reference the root-cause finding, not the child
        prompt_used = client.async_generate.call_args_list[0][0][0]
        assert "root-f" in prompt_used or "Finding root-f" in prompt_used

    @pytest.mark.asyncio
    async def test_falls_back_to_first_finding_when_no_root_causes(self):
        """Step 3: all findings have upstream causes → fall back to findings[0]."""
        f1 = _make_finding_with_causal("f1", caused_by=["f2"])
        f2 = _make_finding_with_causal("f2", caused_by=["f1"])

        enriched_json = json.dumps(
            {"expected_output": "x", "interpretation": "y"}
        )
        client = _make_stub_client(enriched_json)

        report = _make_report(
            findings=[f1, f2],
            recommendations=[_make_action_no_related()],
        )

        await enrich_recommendations(report, client)

        # Should still call the LLM (didn't skip) — meaning it resolved to findings[0]
        assert client.async_generate.call_count == 1

    @pytest.mark.asyncio
    async def test_related_findings_id_match_wins_over_root_cause(self):
        """Step 1 takes priority: explicit related_findings ID match beats root_causes."""
        root = _make_finding_with_causal("root-f", caused_by=[])
        child = _make_finding_with_causal("child-f", caused_by=["root-f"])

        enriched_json = json.dumps(
            {"expected_output": "x", "interpretation": "y"}
        )
        client = _make_stub_client(enriched_json)

        # Action explicitly points to the child finding
        action = RecommendedAction(
            priority=1,
            title="Fix child",
            description="desc",
            urgency=ActionUrgency.IMMEDIATE,
            command="cmd",
            expected_output="out",
            interpretation="interp",
            why="why",
            risk="none",
            related_findings=["child-f"],
        )

        report = _make_report(findings=[root, child], recommendations=[action])
        await enrich_recommendations(report, client)

        prompt_used = client.async_generate.call_args_list[0][0][0]
        assert "child-f" in prompt_used or "Finding child-f" in prompt_used
