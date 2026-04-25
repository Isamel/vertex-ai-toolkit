"""Tests for ExternalLink / ExternalLinks schema models.

Covers:
- Valid ExternalLink instantiation
- Invalid system literal is accepted (system is a plain str, not a Literal)
- ExternalLinks partial groups default to empty lists
- HealthReport backward-compat deserialization without external_links
- HealthReportGeminiSchema pruning of excluded fields and orphaned $defs
- HealthReportGeminiSchema model_dump() retains post-hoc fields (HTML regression)
"""

from __future__ import annotations

import json
import logging

import pytest

from vaig.skills.service_health.schema import (
    ExternalLink,
    ExternalLinks,
    HealthReport,
    HealthReportGeminiSchema,
    ReportMetadata,
    ServiceHealthStatus,
    ServiceStatus,
)

# ── ExternalLink ──────────────────────────────────────────────


class TestExternalLink:
    def test_valid_gcp_link(self) -> None:
        link = ExternalLink(label="GCP Logs", url="https://console.cloud.google.com/logs", system="gcp")
        assert link.label == "GCP Logs"
        assert link.url == "https://console.cloud.google.com/logs"
        assert link.system == "gcp"
        assert link.icon == ""

    def test_valid_datadog_link_with_icon(self) -> None:
        link = ExternalLink(label="DD APM", url="https://app.datadoghq.com/apm", system="datadog", icon="<svg/>")
        assert link.system == "datadog"
        assert link.icon == "<svg/>"

    def test_valid_argocd_link(self) -> None:
        link = ExternalLink(label="ArgoCD", url="https://argocd.example.com/applications/myapp", system="argocd")
        assert link.system == "argocd"

    def test_invalid_system_literal_accepted(self) -> None:
        # system is now a plain str to reduce Gemini schema complexity;
        # any string value is accepted at instantiation time.
        link = ExternalLink(label="Bad", url="https://example.com", system="splunk")
        assert link.system == "splunk"

    def test_empty_system_accepted(self) -> None:
        # Empty string is accepted since system is a plain str (no Literal constraint).
        link = ExternalLink(label="Bad", url="https://example.com", system="")
        assert link.system == ""

    def test_all_system_literals_accepted(self) -> None:
        for system in ("gcp", "datadog", "argocd"):
            link = ExternalLink(label="x", url="https://example.com", system=system)
            assert link.system == system


# ── ExternalLinks ─────────────────────────────────────────────


class TestExternalLinks:
    def test_empty_defaults(self) -> None:
        el = ExternalLinks()
        assert el.gcp == []
        assert el.datadog == []
        assert el.argocd == []

    def test_partial_gcp_only(self) -> None:
        gcp_link = ExternalLink(label="GCP", url="https://console.cloud.google.com", system="gcp")
        el = ExternalLinks(gcp=[gcp_link])
        assert len(el.gcp) == 1
        assert el.datadog == []
        assert el.argocd == []

    def test_all_groups_populated(self) -> None:
        gcp = ExternalLink(label="G", url="https://gcp.com", system="gcp")
        dd = ExternalLink(label="D", url="https://dd.com", system="datadog")
        argo = ExternalLink(label="A", url="https://argo.com", system="argocd")
        el = ExternalLinks(gcp=[gcp], datadog=[dd], argocd=[argo])
        assert len(el.gcp) == 1
        assert len(el.datadog) == 1
        assert len(el.argocd) == 1


# ── HealthReport backward compatibility ───────────────────────


class TestHealthReportBackwardCompat:
    _MINIMAL_PAYLOAD = {
        "executive_summary": {
            "overall_status": "HEALTHY",
            "scope": "Cluster-wide",
            "summary_text": "All good",
        }
    }

    def test_deserialization_without_external_links(self) -> None:
        """Legacy reports without external_links must deserialize without error."""
        report = HealthReport.model_validate(self._MINIMAL_PAYLOAD)
        assert report.external_links is None

    def test_deserialization_with_external_links_null(self) -> None:
        payload = {**self._MINIMAL_PAYLOAD, "external_links": None}
        report = HealthReport.model_validate(payload)
        assert report.external_links is None

    def test_deserialization_with_external_links_present(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "external_links": {
                "gcp": [{"label": "GCP Logs", "url": "https://gcp.com", "system": "gcp"}],
            },
        }
        report = HealthReport.model_validate(payload)
        assert report.external_links is not None
        assert len(report.external_links.gcp) == 1
        assert report.external_links.gcp[0].label == "GCP Logs"

    def test_json_roundtrip(self) -> None:
        """Serialize and deserialize back without data loss."""
        gcp_link = ExternalLink(label="GCP", url="https://gcp.com", system="gcp")
        el = ExternalLinks(gcp=[gcp_link])
        payload = {**self._MINIMAL_PAYLOAD, "external_links": el.model_dump()}
        report = HealthReport.model_validate(payload)
        dumped = json.loads(report.model_dump_json())
        assert dumped["external_links"]["gcp"][0]["label"] == "GCP"


# ── HealthReportGeminiSchema pruning ──────────────────────────


class TestHealthReportGeminiSchemaPruning:
    """Verify that model_json_schema() removes excluded fields and orphaned $defs."""

    def _get_schema(self) -> dict:
        return HealthReportGeminiSchema.model_json_schema()

    def test_schema_is_json_serializable(self) -> None:
        schema = self._get_schema()
        # Should not raise
        json.dumps(schema)

    def test_excluded_fields_not_in_properties(self) -> None:
        schema = self._get_schema()
        props = schema.get("properties", {})
        excluded = {"metadata", "evidence_gaps", "recent_changes", "external_links", "investigation_coverage"}
        for field in excluded:
            assert field not in props, f"Excluded field '{field}' should not appear in schema properties"

    def test_orphaned_defs_removed(self) -> None:
        schema = self._get_schema()
        defs = schema.get("$defs", {})
        # These types are only reachable via excluded fields
        orphans = {"ReportMetadata", "ToolUsageSummary", "ExternalLink", "ExternalLinks", "EvidenceGap", "ChangeEvent"}
        for orphan in orphans:
            assert orphan not in defs, f"Orphaned $def '{orphan}' should have been pruned from schema"

    def test_non_excluded_fields_present(self) -> None:
        schema = self._get_schema()
        props = schema.get("properties", {})
        assert "executive_summary" in props
        assert "findings" in props
        assert "recommendations" in props

    def test_model_dump_retains_post_hoc_fields(self) -> None:
        """Regression: model_dump() must NOT strip post-hoc fields.

        PR #276 added ``exclude=True`` to the field re-declarations in
        ``HealthReportGeminiSchema``, which caused ``model_dump()`` to omit
        ``metadata``, ``evidence_gaps``, ``recent_changes``, ``external_links``,
        and ``investigation_coverage``.  The HTML report template reads
        ``REPORT_DATA.metadata.project_id`` (and friends) — their absence
        caused a silent JS crash and a blank page.
        """
        minimal_payload = {
            "executive_summary": {
                "overall_status": "HEALTHY",
                "scope": "Cluster-wide",
                "summary_text": "All good",
            }
        }
        instance = HealthReportGeminiSchema.model_validate(minimal_payload)
        dumped = instance.model_dump()
        post_hoc = {"metadata", "evidence_gaps", "recent_changes", "external_links", "investigation_coverage"}
        for field in post_hoc:
            assert field in dumped, (
                f"model_dump() must retain post-hoc field '{field}' "
                f"(HTML report template depends on it)"
            )


# ── AUDIT-04: overall_severity_reason ────────────────────────


class TestOverallSeverityReason:
    """AUDIT-04 — overall_severity_reason field in HealthReport."""

    _MINIMAL_PAYLOAD: dict = {
        "executive_summary": {
            "overall_status": "DEGRADED",
            "scope": "Cluster-wide",
            "summary_text": "Issues detected",
        }
    }

    def test_field_defaults_to_none(self) -> None:
        report = HealthReport.model_validate(self._MINIMAL_PAYLOAD)
        assert report.overall_severity_reason is None

    def test_field_accepts_string(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "overall_severity_reason": (
                "DEGRADED because 1 HIGH finding with confidence=CONFIRMED "
                "and 2 MEDIUM findings in distinct namespaces."
            ),
        }
        report = HealthReport.model_validate(payload)
        assert report.overall_severity_reason is not None
        assert "1 HIGH" in report.overall_severity_reason
        assert "CONFIRMED" in report.overall_severity_reason

    def test_field_present_in_gemini_schema(self) -> None:
        """overall_severity_reason is reporter-populated, must be in Gemini schema."""
        schema = HealthReportGeminiSchema.model_json_schema()
        props = schema.get("properties", {})
        assert "overall_severity_reason" in props, (
            "overall_severity_reason must be present in the Gemini schema "
            "(it is filled by the reporter LLM, not post-hoc)"
        )

    def test_markdown_renders_reason_when_present(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "overall_severity_reason": "DEGRADED because 1 HIGH finding with confidence=CONFIRMED.",
        }
        report = HealthReport.model_validate(payload)
        md = report.to_markdown()
        assert "DEGRADED because 1 HIGH finding with confidence=CONFIRMED." in md

    def test_markdown_omits_reason_when_none(self) -> None:
        report = HealthReport.model_validate(self._MINIMAL_PAYLOAD)
        md = report.to_markdown()
        # No stray italic line under Status line
        lines = md.splitlines()
        status_idx = next(i for i, l in enumerate(lines) if "**Status**" in l)
        next_line = lines[status_idx + 1] if status_idx + 1 < len(lines) else ""
        assert not (next_line.strip().startswith("*") and next_line.strip().endswith("*")), (
            "No italic reason line should appear when overall_severity_reason is None"
        )

    def test_roundtrip_preserves_reason(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "overall_severity_reason": "HEALTHY because no findings above LOW severity.",
        }
        report = HealthReport.model_validate(payload)
        dumped = report.model_dump()
        assert dumped["overall_severity_reason"] == "HEALTHY because no findings above LOW severity."


# ── AUDIT-05: ServiceStatus.degraded_reason ───────────────────


_MINIMAL_EXEC = {
    "overall_status": "HEALTHY",
    "scope": "Cluster-wide",
    "summary_text": "All good",
}


class TestServiceStatusDegradedReason:
    """AUDIT-05 — degraded_reason field on ServiceStatus."""

    def test_field_defaults_to_none(self) -> None:
        svc = ServiceStatus(service="payment-svc", namespace="prod", status="HEALTHY")
        assert svc.degraded_reason is None

    def test_field_accepts_string_for_degraded(self) -> None:
        svc = ServiceStatus(
            service="payment-svc",
            namespace="prod",
            status="DEGRADED",
            degraded_reason="15.79% APM error rate over last 15 min.",
        )
        assert svc.degraded_reason == "15.79% APM error rate over last 15 min."
        assert svc.status == ServiceHealthStatus.DEGRADED

    def test_field_accepts_string_for_failed(self) -> None:
        svc = ServiceStatus(
            service="payment-svc",
            namespace="prod",
            status="FAILED",
            degraded_reason="All pods in CrashLoopBackOff.",
        )
        assert svc.degraded_reason == "All pods in CrashLoopBackOff."

    def test_max_length_truncated_not_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        """degraded_reason > 160 chars must be truncated with ellipsis, not rejected."""
        long_reason = "x" * 161
        with caplog.at_level(logging.WARNING, logger="vaig.skills.service_health.schema"):
            svc = ServiceStatus(
                service="svc",
                namespace="ns",
                status="DEGRADED",
                degraded_reason=long_reason,
            )
        assert svc.degraded_reason is not None
        assert len(svc.degraded_reason) <= 160
        assert svc.degraded_reason.endswith("…")
        assert "truncated" in caplog.text

    def test_validator_warns_when_non_healthy_reason_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="vaig.skills.service_health.schema"):
            ServiceStatus(service="broken-svc", namespace="prod", status="DEGRADED")
        assert "broken-svc" in caplog.text
        assert "degraded_reason" in caplog.text

    def test_no_warning_when_healthy_without_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="vaig.skills.service_health.schema"):
            ServiceStatus(service="ok-svc", namespace="prod", status="HEALTHY")
        assert "ok-svc" not in caplog.text

    def test_no_warning_when_degraded_with_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="vaig.skills.service_health.schema"):
            ServiceStatus(
                service="svc",
                namespace="prod",
                status="DEGRADED",
                degraded_reason="High error rate",
            )
        assert "degraded_reason" not in caplog.text

    def test_roundtrip_preserves_degraded_reason(self) -> None:
        svc = ServiceStatus(
            service="svc",
            namespace="prod",
            status="DEGRADED",
            degraded_reason="Latency spike detected.",
        )
        dumped = svc.model_dump()
        restored = ServiceStatus.model_validate(dumped)
        assert restored.degraded_reason == "Latency spike detected."

    def test_markdown_renders_reason_inline_for_degraded(self) -> None:
        payload = {
            "executive_summary": _MINIMAL_EXEC,
            "service_statuses": [
                {
                    "service": "payment-svc",
                    "namespace": "prod",
                    "status": "DEGRADED",
                    "degraded_reason": "15.79% APM error rate over last 15 min.",
                }
            ],
        }
        report = HealthReport.model_validate(payload)
        md = report.to_markdown()
        assert "🟡 *15.79% APM error rate over last 15 min.*" in md

    def test_markdown_renders_reason_inline_for_failed(self) -> None:
        payload = {
            "executive_summary": _MINIMAL_EXEC,
            "service_statuses": [
                {
                    "service": "broken-svc",
                    "namespace": "prod",
                    "status": "FAILED",
                    "degraded_reason": "All pods in CrashLoopBackOff.",
                }
            ],
        }
        report = HealthReport.model_validate(payload)
        md = report.to_markdown()
        assert "🔴 *All pods in CrashLoopBackOff.*" in md

    def test_markdown_no_inline_reason_for_healthy(self) -> None:
        payload = {
            "executive_summary": _MINIMAL_EXEC,
            "service_statuses": [
                {
                    "service": "ok-svc",
                    "namespace": "prod",
                    "status": "HEALTHY",
                }
            ],
        }
        report = HealthReport.model_validate(payload)
        md = report.to_markdown()
        # Healthy status shows emoji only, no italic subline
        assert "🟢 *" not in md

    def test_markdown_no_inline_reason_when_degraded_reason_empty(self) -> None:
        """DEGRADED with no reason → no italic injected (just emoji)."""
        payload = {
            "executive_summary": _MINIMAL_EXEC,
            "service_statuses": [
                {
                    "service": "svc",
                    "namespace": "prod",
                    "status": "DEGRADED",
                }
            ],
        }
        report = HealthReport.model_validate(payload)
        md = report.to_markdown()
        assert "🟡 *" not in md

    def test_field_present_in_gemini_schema(self) -> None:
        schema = HealthReportGeminiSchema.model_json_schema()
        service_status_def = schema.get("$defs", {}).get("ServiceStatus", {})
        assert "degraded_reason" in service_status_def.get("properties", {}), (
            "degraded_reason must be present in the Gemini schema for ServiceStatus"
        )


# ── AUDIT-07: Run determinism metadata ───────────────────────────────────────


class TestAudit07RunDeterminismMetadata:
    """Tests for AUDIT-07 — run_seed, model_versions, pipeline_version."""

    def test_report_metadata_defaults(self) -> None:
        """ReportMetadata default values are sane and non-breaking."""
        meta = ReportMetadata()
        assert meta.run_seed is None
        assert meta.model_versions == {}
        assert meta.pipeline_version == "unknown"
        assert meta.autonomous_enabled is False
        assert meta.autonomous_steps_executed is None

    def test_report_metadata_round_trip(self) -> None:
        """ReportMetadata serialises and deserialises AUDIT-07 fields correctly."""
        meta = ReportMetadata(
            run_seed=42,
            model_versions={"health_analyzer": "gemini-2.5-pro-002"},
            pipeline_version="a1b2c3d",
            autonomous_enabled=True,
            autonomous_steps_executed=5,
        )
        dumped = meta.model_dump()
        restored = ReportMetadata.model_validate(dumped)
        assert restored.run_seed == 42
        assert restored.model_versions == {"health_analyzer": "gemini-2.5-pro-002"}
        assert restored.pipeline_version == "a1b2c3d"
        assert restored.autonomous_enabled is True
        assert restored.autonomous_steps_executed == 5

    def test_health_report_carries_audit07_fields(self) -> None:
        """HealthReport propagates AUDIT-07 metadata fields through model_dump."""
        payload = {
            "executive_summary": _MINIMAL_EXEC,
            "metadata": {
                "pipeline_version": "deadbeef",
                "model_versions": {"health_analyzer": "gemini-2.5-pro-001"},
                "run_seed": 7,
                "autonomous_enabled": True,
                "autonomous_steps_executed": 3,
            },
        }
        report = HealthReport.model_validate(payload)
        assert report.metadata.pipeline_version == "deadbeef"
        assert report.metadata.model_versions == {"health_analyzer": "gemini-2.5-pro-001"}
        assert report.metadata.run_seed == 7
        assert report.metadata.autonomous_enabled is True
        assert report.metadata.autonomous_steps_executed == 3

    def test_gemini_schema_excludes_metadata(self) -> None:
        """metadata (containing AUDIT-07 fields) is excluded from the Gemini schema."""
        schema = HealthReportGeminiSchema.model_json_schema()
        props = schema.get("properties", {})
        assert "metadata" not in props, (
            "metadata (with AUDIT-07 fields) must be excluded from Gemini schema"
        )

    def test_model_dump_retains_audit07_fields(self) -> None:
        """model_dump() must retain AUDIT-07 fields — HTML renderer reads them."""
        report = HealthReport.model_validate({"executive_summary": _MINIMAL_EXEC})
        report.metadata.pipeline_version = "abc1234"
        report.metadata.model_versions = {"health_analyzer": "gemini-2.5-flash"}
        dumped = report.model_dump()
        meta_dump = dumped.get("metadata", {})
        assert meta_dump.get("pipeline_version") == "abc1234"
        assert meta_dump.get("model_versions") == {"health_analyzer": "gemini-2.5-flash"}

    def test_gemini_schema_size_unchanged_by_audit07(self) -> None:
        """AUDIT-07 fields do not increase the Gemini schema size (post-hoc)."""
        schema_str = json.dumps(HealthReportGeminiSchema.model_json_schema())
        # Schema must remain under 20 KB — new post-hoc fields must NOT appear
        assert len(schema_str) < 20_000, (
            f"Gemini schema grew unexpectedly: {len(schema_str)} chars "
            "(AUDIT-07 fields must be post-hoc only)"
        )

    def test_gemini_schema_no_prefix_items(self) -> None:
        """Gemini schema must not contain 'prefixItems' (JSON Schema draft 2020-12 keyword).

        Regression guard for the RepoSnippet.line_range tuple[int,int] bug —
        Gemini's validator uses draft-07 and rejects prefixItems with
        'Extra inputs are not permitted'.
        """
        schema_str = json.dumps(HealthReportGeminiSchema.model_json_schema())
        assert "prefixItems" not in schema_str, (
            "Gemini schema contains 'prefixItems' — a draft-2020-12 keyword unsupported "
            "by Gemini. Use flat scalar fields instead of tuple types."
        )


# ── AUDIT-16: Schema state budget CI guard ───────────────────────────────────


def _count_gemini_states(schema: dict) -> int:
    """Approximate the Gemini state count using the same heuristic as the runtime.

    Heuristic (verified empirically):
      - Each scalar property in an object contributes 1 state.
      - Each array property contributes ``array_depth * element_states`` states,
        where ``array_depth`` is the nesting depth and ``element_states`` is the
        count for the element type.
      - $defs are shared; we count their states once and substitute inline.

    This is intentionally conservative — Gemini's exact count is undisclosed —
    so we stay safely below the observed 32 768 hard limit.
    """
    defs: dict[str, dict] = schema.get("$defs", {})

    def _states_for(node: dict, depth: int = 0) -> int:
        if depth > 10:  # guard against infinite recursion
            return 1
        ref = node.get("$ref", "")
        if ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            return _states_for(defs.get(def_name, {}), depth + 1)
        node_type = node.get("type", "")
        if node_type == "object":
            props = node.get("properties", {})
            return max(sum(_states_for(v, depth + 1) for v in props.values()), 1)
        if node_type == "array":
            items = node.get("items", {})
            return (depth + 1) * _states_for(items, depth + 1)
        # anyOf / oneOf / allOf
        for combiner in ("anyOf", "oneOf", "allOf"):
            branches = node.get(combiner)
            if branches:
                return sum(_states_for(b, depth + 1) for b in branches)
        return 1  # scalar / enum / string / number / boolean / null

    return _states_for(schema)


_STATE_BUDGET_MAX = 25_000
_STATE_BUDGET_WARN = int(_STATE_BUDGET_MAX * 0.80)  # 20 000

_CHAR_BUDGET_MAX = 20_000
_DEFS_BUDGET_MAX = 25


class TestAudit16SchemaStateBudget:
    """AUDIT-16 — Schema state budget CI guard.

    Ensures the Gemini schema does not grow beyond thresholds that would
    trigger Gemini's "too many states" error (observed hard limit ~32 768).
    """

    def _get_schema(self) -> dict:
        return HealthReportGeminiSchema.model_json_schema()

    # ── hard budgets ──────────────────────────────────────────────────────────

    def test_char_count_within_budget(self) -> None:
        """Schema serialised to JSON must stay below the char budget."""
        schema_str = json.dumps(self._get_schema())
        char_count = len(schema_str)
        assert char_count <= _CHAR_BUDGET_MAX, (
            f"Gemini schema exceeds char budget: {char_count} chars "
            f"(max {_CHAR_BUDGET_MAX}). "
            "Remove or simplify fields to restore budget."
        )

    def test_defs_count_within_budget(self) -> None:
        """Number of $defs must stay at or below the $defs budget."""
        schema = self._get_schema()
        defs = schema.get("$defs", {})
        defs_count = len(defs)
        assert defs_count <= _DEFS_BUDGET_MAX, (
            f"Gemini schema has {defs_count} $defs (max {_DEFS_BUDGET_MAX}). "
            f"Excess defs: {sorted(set(defs) - set(list(defs)[:_DEFS_BUDGET_MAX]))}. "
            "Prune or inline types to restore budget."
        )

    def test_state_count_within_budget(self) -> None:
        """Approximate Gemini state count must stay below the hard budget."""
        schema = self._get_schema()
        state_count = _count_gemini_states(schema)
        assert state_count <= _STATE_BUDGET_MAX, (
            f"Gemini schema state count {state_count} exceeds budget {_STATE_BUDGET_MAX}. "
            "This will likely trigger Gemini's 'too many states' error. "
            "Simplify nested models or remove fields."
        )

    # ── soft warning at 80 % ──────────────────────────────────────────────────

    def test_state_count_below_warning_threshold(self) -> None:
        """Emit a UserWarning when state count exceeds 80 % of the budget."""
        import warnings

        schema = self._get_schema()
        state_count = _count_gemini_states(schema)
        if state_count > _STATE_BUDGET_WARN:
            warnings.warn(
                f"Gemini schema state count {state_count} exceeds 80 % of the "
                f"budget ({_STATE_BUDGET_WARN}/{_STATE_BUDGET_MAX}). "
                "Consider simplifying the schema before it hits the hard limit.",
                UserWarning,
                stacklevel=2,
            )
        # The test itself never fails here — the warning is the signal.
        # Hard failure is handled by test_state_count_within_budget above.
        assert state_count <= _STATE_BUDGET_MAX, (
            f"State count {state_count} is beyond hard budget {_STATE_BUDGET_MAX}."
        )

    # ── snapshot / regression guard ───────────────────────────────────────────

    def test_schema_metrics_snapshot(self) -> None:
        """Snapshot test: flag unexpected growth in any metric.

        Thresholds are set ~10 % above the current measured values so that
        a single new field is caught before it blows the hard budget.

        Current baseline (post PR #276 BFS optimisation):
          chars ~16 975 · $defs 21 · states (heuristic, varies by impl)
        """
        schema = self._get_schema()
        schema_str = json.dumps(schema)
        defs = schema.get("$defs", {})

        char_count = len(schema_str)
        defs_count = len(defs)

        # ~10 % headroom above the post-BFS baseline
        _SNAPSHOT_CHAR_MAX = 20_500  # ATT-11: AttachmentCitation added to Finding
        _SNAPSHOT_DEFS_MAX = 23

        assert char_count <= _SNAPSHOT_CHAR_MAX, (
            f"Schema char count grew to {char_count} (snapshot ceiling {_SNAPSHOT_CHAR_MAX}). "
            "If this is intentional, update _SNAPSHOT_CHAR_MAX in AUDIT-16 tests."
        )
        assert defs_count <= _SNAPSHOT_DEFS_MAX, (
            f"Schema $defs count grew to {defs_count} (snapshot ceiling {_SNAPSHOT_DEFS_MAX}). "
            "If this is intentional, update _SNAPSHOT_DEFS_MAX in AUDIT-16 tests."
        )
