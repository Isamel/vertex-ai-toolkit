"""Tests for vaig.core.export_transformers — pure data transformation functions.

All tests are unit tests with no I/O, no GCP dependencies, and no async.
Each test class covers one transformer function.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from vaig.core.export_transformers import (
    transform_feedback_record,
    transform_health_report,
    transform_telemetry_record,
    transform_tool_call_record,
)

# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _make_telemetry(**overrides: Any) -> dict[str, Any]:
    """Return a fully-populated telemetry record suitable for testing."""
    base: dict[str, Any] = {
        "timestamp": "2025-06-01T12:00:00+00:00",
        "event_type": "tool_call",
        "tool_name": "get_pods",
        "agent_name": "service-health-agent",
        "duration_ms": 123.4,
        "success": True,
        "error_message": None,
        "metadata": {"namespace": "default", "cluster": "dev"},
        "session_id": "sess-abc123",
        "run_id": "run-001",
    }
    base.update(overrides)
    return base


def _make_tool_call(**overrides: Any) -> dict[str, Any]:
    """Return a fully-populated tool call record suitable for testing."""
    base: dict[str, Any] = {
        "timestamp": "2025-06-01T13:00:00+00:00",
        "tool_name": "describe_pod",
        "agent_name": "coding-agent",
        "input_params": {"pod_name": "nginx-abc", "namespace": "prod"},
        "output_summary": "Pod nginx-abc is Running",
        "duration_ms": 55.0,
        "success": True,
        "error_message": None,
        "run_id": "run-002",
        "session_id": "sess-xyz789",
    }
    base.update(overrides)
    return base


def _make_health_report_dict(**overrides: Any) -> dict[str, Any]:
    """Return a minimal HealthReport.to_dict()-style dict for testing."""
    base: dict[str, Any] = {
        "executive_summary": {
            "overall_status": "DEGRADED",
            "issues_found": 3,
            "critical_count": 1,
            "warning_count": 2,
            "scope": "namespace:prod",
            "summary": "One critical crashloop detected",
        },
        "findings": [
            {
                "category": "pod-health",
                "severity": "CRITICAL",
                "title": "CrashLoopBackOff: payment-svc",
                "description": "Payment service is crash-looping",
                "recommendation": "Check logs for OOMKilled",
            },
            {
                "category": "scaling",
                "severity": "WARNING",
                "title": "HPA at max replicas",
                "description": "HPA cannot scale further",
                "recommendation": "Increase max replicas or reduce resource requests",
            },
        ],
        "metadata": {"run_at": "2025-06-01T14:00:00Z", "version": "1.2.0"},
    }
    base.update(overrides)
    return base


def _make_feedback(**overrides: Any) -> dict[str, Any]:
    """Return a fully-populated feedback record for testing."""
    base: dict[str, Any] = {
        "rating": 4,
        "comment": "Report was mostly accurate but missed one issue.",
        "auto_quality_score": 0.87,
        "report_summary": "DEGRADED: 1 critical, 2 warnings in namespace:prod",
        "metadata": {"user": "alice", "source": "cli"},
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════
# TestTransformTelemetry
# ══════════════════════════════════════════════════════════════════════════


class TestTransformTelemetry:
    """Tests for transform_telemetry_record()."""

    # ── Happy path ──────────────────────────────────────────────────────

    def test_happy_path_returns_all_expected_keys(self) -> None:
        row = transform_telemetry_record(_make_telemetry())
        expected_keys = {
            "timestamp", "event_type", "tool_name", "agent_name",
            "duration_ms", "success", "error_message", "metadata",
            "session_id", "run_id",
        }
        assert set(row.keys()) == expected_keys

    def test_happy_path_event_type_passthrough(self) -> None:
        row = transform_telemetry_record(_make_telemetry(event_type="api_call"))
        assert row["event_type"] == "api_call"

    def test_happy_path_tool_name_passthrough(self) -> None:
        row = transform_telemetry_record(_make_telemetry(tool_name="list_nodes"))
        assert row["tool_name"] == "list_nodes"

    def test_happy_path_agent_name_passthrough(self) -> None:
        row = transform_telemetry_record(_make_telemetry(agent_name="orchestrator"))
        assert row["agent_name"] == "orchestrator"

    def test_happy_path_duration_ms_is_float(self) -> None:
        row = transform_telemetry_record(_make_telemetry(duration_ms=99))
        assert isinstance(row["duration_ms"], float)
        assert row["duration_ms"] == 99.0

    def test_happy_path_success_true(self) -> None:
        row = transform_telemetry_record(_make_telemetry(success=True))
        assert row["success"] is True

    def test_happy_path_success_false(self) -> None:
        row = transform_telemetry_record(_make_telemetry(success=False, error_message="timeout"))
        assert row["success"] is False

    def test_happy_path_session_id_passthrough(self) -> None:
        row = transform_telemetry_record(_make_telemetry(session_id="sess-999"))
        assert row["session_id"] == "sess-999"

    def test_happy_path_run_id_passthrough(self) -> None:
        row = transform_telemetry_record(_make_telemetry(run_id="run-xyz"))
        assert row["run_id"] == "run-xyz"

    # ── Timestamp normalisation ──────────────────────────────────────────

    def test_iso_string_timestamp_parsed_to_datetime(self) -> None:
        row = transform_telemetry_record(_make_telemetry(timestamp="2025-06-01T12:00:00+00:00"))
        assert isinstance(row["timestamp"], datetime)

    def test_naive_iso_string_gets_utc_tzinfo(self) -> None:
        row = transform_telemetry_record(_make_telemetry(timestamp="2025-06-01T12:00:00"))
        ts = row["timestamp"]
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_datetime_object_passthrough(self) -> None:
        dt = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        row = transform_telemetry_record(_make_telemetry(timestamp=dt))
        assert row["timestamp"] == dt

    def test_naive_datetime_gets_utc(self) -> None:
        naive = datetime(2025, 6, 1, 12, 0, 0)
        row = transform_telemetry_record(_make_telemetry(timestamp=naive))
        assert row["timestamp"].tzinfo is not None

    def test_unix_float_timestamp_parsed(self) -> None:
        unix_ts = 1_748_779_200.0  # 2025-06-01 12:00:00 UTC
        row = transform_telemetry_record(_make_telemetry(timestamp=unix_ts))
        assert isinstance(row["timestamp"], datetime)
        assert row["timestamp"].year == 2025

    # ── Metadata serialisation ───────────────────────────────────────────

    def test_metadata_dict_becomes_json_string(self) -> None:
        row = transform_telemetry_record(_make_telemetry(metadata={"key": "value"}))
        assert isinstance(row["metadata"], str)
        parsed = json.loads(row["metadata"])
        assert parsed == {"key": "value"}

    def test_metadata_already_string_passthrough(self) -> None:
        row = transform_telemetry_record(_make_telemetry(metadata='{"already": "serialised"}'))
        assert row["metadata"] == '{"already": "serialised"}'

    def test_metadata_json_field_alias_used_when_metadata_missing(self) -> None:
        """telemetry schema uses metadata_json; both should be accepted."""
        record = {
            "timestamp": "2025-06-01T12:00:00+00:00",
            "event_type": "tool_call",
            "metadata_json": '{"ns": "default"}',
        }
        row = transform_telemetry_record(record)
        assert row["metadata"] == '{"ns": "default"}'

    # ── Missing optional fields use defaults ─────────────────────────────

    def test_missing_optional_fields_use_defaults(self) -> None:
        row = transform_telemetry_record({
            "timestamp": "2025-06-01T12:00:00+00:00",
            "event_type": "session",
        })
        assert row["tool_name"] == ""
        assert row["agent_name"] == ""
        assert row["duration_ms"] == 0.0
        assert row["error_message"] is None
        assert row["session_id"] == ""
        assert row["run_id"] == ""

    def test_missing_timestamp_falls_back_to_now(self) -> None:
        before = datetime.now(UTC)
        row = transform_telemetry_record({"event_type": "session"})
        after = datetime.now(UTC)
        assert before <= row["timestamp"] <= after

    def test_success_inferred_from_absent_error_when_success_missing(self) -> None:
        row = transform_telemetry_record({"event_type": "tool_call"})
        assert row["success"] is True  # no error_message → infer True

    def test_success_int_zero_treated_as_false(self) -> None:
        row = transform_telemetry_record(_make_telemetry(success=0))
        assert row["success"] is False

    def test_success_int_one_treated_as_true(self) -> None:
        row = transform_telemetry_record(_make_telemetry(success=1))
        assert row["success"] is True

    def test_success_string_false_treated_as_false(self) -> None:
        row = transform_telemetry_record(_make_telemetry(success="false"))
        assert row["success"] is False

    def test_error_message_from_error_msg_alias(self) -> None:
        """Some telemetry rows store error under 'error_msg' (old schema)."""
        record = _make_telemetry()
        del record["error_message"]
        record["error_msg"] = "connection refused"
        row = transform_telemetry_record(record)
        assert row["error_message"] == "connection refused"

    def test_agent_name_falls_back_to_event_name(self) -> None:
        """Telemetry uses event_name; agent_name is preferred but optional."""
        record = {
            "timestamp": "2025-06-01T12:00:00+00:00",
            "event_type": "tool_call",
            "event_name": "get_pods",
        }
        row = transform_telemetry_record(record)
        assert row["agent_name"] == "get_pods"


# ══════════════════════════════════════════════════════════════════════════
# TestTransformToolCall
# ══════════════════════════════════════════════════════════════════════════


class TestTransformToolCall:
    """Tests for transform_tool_call_record()."""

    # ── Happy path ──────────────────────────────────────────────────────

    def test_happy_path_returns_all_expected_keys(self) -> None:
        row = transform_tool_call_record(_make_tool_call())
        expected_keys = {
            "timestamp", "tool_name", "agent_name", "input_params",
            "output_summary", "duration_ms", "success", "error_message",
            "run_id", "session_id",
        }
        assert set(row.keys()) == expected_keys

    def test_happy_path_tool_name(self) -> None:
        row = transform_tool_call_record(_make_tool_call(tool_name="scale_deployment"))
        assert row["tool_name"] == "scale_deployment"

    def test_happy_path_agent_name(self) -> None:
        row = transform_tool_call_record(_make_tool_call(agent_name="k8s-agent"))
        assert row["agent_name"] == "k8s-agent"

    def test_happy_path_duration_ms_is_float(self) -> None:
        row = transform_tool_call_record(_make_tool_call(duration_ms=200))
        assert isinstance(row["duration_ms"], float)
        assert row["duration_ms"] == 200.0

    def test_happy_path_success_true(self) -> None:
        row = transform_tool_call_record(_make_tool_call(success=True))
        assert row["success"] is True

    def test_happy_path_input_params_dict_serialised(self) -> None:
        params = {"pod_name": "nginx", "namespace": "prod"}
        row = transform_tool_call_record(_make_tool_call(input_params=params))
        assert isinstance(row["input_params"], str)
        assert json.loads(row["input_params"]) == params

    def test_happy_path_output_summary_passthrough(self) -> None:
        row = transform_tool_call_record(_make_tool_call(output_summary="OK: 5 pods running"))
        assert row["output_summary"] == "OK: 5 pods running"

    # ── Output summary truncation ────────────────────────────────────────

    def test_output_summary_truncated_at_10000_chars(self) -> None:
        long_output = "x" * 15_000
        row = transform_tool_call_record(_make_tool_call(output_summary=long_output))
        assert len(row["output_summary"]) == 10_000

    def test_output_summary_exactly_10000_not_truncated(self) -> None:
        exact = "y" * 10_000
        row = transform_tool_call_record(_make_tool_call(output_summary=exact))
        assert len(row["output_summary"]) == 10_000

    def test_output_summary_under_10000_not_truncated(self) -> None:
        short = "z" * 500
        row = transform_tool_call_record(_make_tool_call(output_summary=short))
        assert row["output_summary"] == short

    # ── Missing / defaults ───────────────────────────────────────────────

    def test_missing_optional_fields_use_defaults(self) -> None:
        row = transform_tool_call_record({
            "timestamp": "2025-06-01T12:00:00+00:00",
            "tool_name": "get_pods",
        })
        assert row["agent_name"] == ""
        assert row["duration_ms"] == 0.0
        assert row["error_message"] is None
        assert row["run_id"] == ""
        assert row["session_id"] == ""

    def test_missing_input_params_defaults_to_empty_json_object(self) -> None:
        row = transform_tool_call_record({"tool_name": "noop"})
        assert json.loads(row["input_params"]) == {}

    def test_missing_output_summary_defaults_to_empty_string(self) -> None:
        row = transform_tool_call_record({"tool_name": "noop"})
        assert row["output_summary"] == ""

    def test_timestamp_string_parsed_to_datetime(self) -> None:
        row = transform_tool_call_record(_make_tool_call(timestamp="2025-06-01T13:00:00Z"))
        # Must parse the 'Z' suffix and return a UTC-aware datetime at the exact moment
        assert row["timestamp"] == datetime(2025, 6, 1, 13, 0, 0, tzinfo=UTC)

    def test_error_message_none_when_absent(self) -> None:
        row = transform_tool_call_record({"tool_name": "noop"})
        assert row["error_message"] is None

    def test_error_message_string_when_present(self) -> None:
        row = transform_tool_call_record(_make_tool_call(error_message="timeout"))
        assert row["error_message"] == "timeout"

    def test_success_false_string_parsed(self) -> None:
        row = transform_tool_call_record(_make_tool_call(success="false"))
        assert row["success"] is False


# ══════════════════════════════════════════════════════════════════════════
# TestTransformHealthReport
# ══════════════════════════════════════════════════════════════════════════


class TestTransformHealthReport:
    """Tests for transform_health_report()."""

    # ── Happy path ──────────────────────────────────────────────────────

    def test_happy_path_returns_all_expected_keys(self) -> None:
        row = transform_health_report(
            _make_health_report_dict(),
            run_id="run-health-001",
            cluster_name="my-cluster",
            namespace="prod",
        )
        expected_keys = {
            "timestamp", "run_id", "cluster_name", "namespace",
            "overall_status", "summary", "findings", "metadata", "report_markdown",
        }
        assert set(row.keys()) == expected_keys

    def test_happy_path_run_id_from_parameter(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="run-42")
        assert row["run_id"] == "run-42"

    def test_happy_path_cluster_name_from_parameter(self) -> None:
        row = transform_health_report(
            _make_health_report_dict(), run_id="r", cluster_name="prod-cluster"
        )
        assert row["cluster_name"] == "prod-cluster"

    def test_happy_path_namespace_from_parameter(self) -> None:
        row = transform_health_report(
            _make_health_report_dict(), run_id="r", namespace="kube-system"
        )
        assert row["namespace"] == "kube-system"

    def test_happy_path_overall_status_extracted(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert row["overall_status"] == "DEGRADED"

    def test_happy_path_findings_list_is_list(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert isinstance(row["findings"], list)
        assert len(row["findings"]) == 2

    def test_happy_path_finding_has_required_keys(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        finding = row["findings"][0]
        assert set(finding.keys()) == {"category", "severity", "title", "description", "recommendation"}

    def test_happy_path_finding_values(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        first = row["findings"][0]
        assert first["category"] == "pod-health"
        assert first["severity"] == "CRITICAL"
        assert first["title"] == "CrashLoopBackOff: payment-svc"

    def test_happy_path_metadata_serialised_to_json(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert isinstance(row["metadata"], str)
        parsed = json.loads(row["metadata"])
        assert parsed["version"] == "1.2.0"

    # ── Empty findings list ──────────────────────────────────────────────

    def test_empty_findings_returns_empty_list(self) -> None:
        report = _make_health_report_dict(findings=[])
        row = transform_health_report(report, run_id="r")
        assert row["findings"] == []

    def test_missing_findings_key_returns_empty_list(self) -> None:
        report = _make_health_report_dict()
        del report["findings"]
        row = transform_health_report(report, run_id="r")
        assert row["findings"] == []

    # ── Truncation ───────────────────────────────────────────────────────

    def test_summary_truncated_at_50000_chars(self) -> None:
        long_summary = "s" * 60_000
        report = {"executive_summary": {"summary": long_summary}}
        row = transform_health_report(report, run_id="r")
        assert len(row["summary"]) == 50_000

    def test_report_markdown_truncated_at_100000_chars(self) -> None:
        long_md = "#" * 120_000
        report = _make_health_report_dict(report_markdown=long_md)
        row = transform_health_report(report, run_id="r")
        assert len(row["report_markdown"]) == 100_000

    def test_report_markdown_none_when_absent(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert row["report_markdown"] is None

    def test_report_markdown_passthrough_when_within_limit(self) -> None:
        md = "# Report\n\nAll good."
        report = _make_health_report_dict(report_markdown=md)
        row = transform_health_report(report, run_id="r")
        assert row["report_markdown"] == md

    # ── Timestamp is current UTC ─────────────────────────────────────────

    def test_timestamp_is_iso_string(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert isinstance(row["timestamp"], str)
        # Must be parseable as ISO-8601
        parsed = datetime.fromisoformat(row["timestamp"])
        assert parsed.tzinfo is not None

    def test_timestamp_is_approximately_now(self) -> None:
        before = datetime.now(UTC)
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        after = datetime.now(UTC)
        parsed = datetime.fromisoformat(row["timestamp"])
        assert before <= parsed <= after

    # ── Enum value handling (Pydantic model_dump output) ─────────────────

    def test_overall_status_from_enum_dict_value(self) -> None:
        """Pydantic may serialise enums as {'value': 'HEALTHY'}."""
        report = _make_health_report_dict()
        report["executive_summary"]["overall_status"] = {"value": "HEALTHY"}
        row = transform_health_report(report, run_id="r")
        assert row["overall_status"] == "HEALTHY"

    def test_severity_from_enum_dict_in_finding(self) -> None:
        """Pydantic may serialise severity enum as {'value': 'WARNING'}."""
        report = _make_health_report_dict()
        report["findings"][0]["severity"] = {"value": "CRITICAL"}
        row = transform_health_report(report, run_id="r")
        assert row["findings"][0]["severity"] == "CRITICAL"

    # ── Default parameters ────────────────────────────────────────────────

    def test_cluster_name_defaults_to_empty_string(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert row["cluster_name"] == ""

    def test_namespace_defaults_to_empty_string(self) -> None:
        row = transform_health_report(_make_health_report_dict(), run_id="r")
        assert row["namespace"] == ""


# ══════════════════════════════════════════════════════════════════════════
# TestTransformFeedback
# ══════════════════════════════════════════════════════════════════════════


class TestTransformFeedback:
    """Tests for transform_feedback_record()."""

    # ── Happy path ──────────────────────────────────────────────────────

    def test_happy_path_returns_all_expected_keys(self) -> None:
        row = transform_feedback_record(_make_feedback(), run_id="run-fb-001")
        expected_keys = {
            "timestamp", "run_id", "rating", "comment",
            "auto_quality_score", "report_summary", "metadata",
        }
        assert set(row.keys()) == expected_keys

    def test_happy_path_run_id_from_parameter(self) -> None:
        row = transform_feedback_record(_make_feedback(), run_id="run-77")
        assert row["run_id"] == "run-77"

    def test_happy_path_rating_passthrough(self) -> None:
        row = transform_feedback_record(_make_feedback(rating=5), run_id="r")
        assert row["rating"] == 5

    def test_happy_path_comment_passthrough(self) -> None:
        row = transform_feedback_record(_make_feedback(comment="Great report!"), run_id="r")
        assert row["comment"] == "Great report!"

    def test_happy_path_auto_quality_score_is_float(self) -> None:
        row = transform_feedback_record(_make_feedback(auto_quality_score=0.95), run_id="r")
        assert isinstance(row["auto_quality_score"], float)
        assert row["auto_quality_score"] == 0.95

    def test_happy_path_report_summary_passthrough(self) -> None:
        row = transform_feedback_record(
            _make_feedback(report_summary="HEALTHY: 0 issues"), run_id="r"
        )
        assert row["report_summary"] == "HEALTHY: 0 issues"

    def test_happy_path_metadata_dict_serialised(self) -> None:
        row = transform_feedback_record(_make_feedback(metadata={"user": "bob"}), run_id="r")
        assert isinstance(row["metadata"], str)
        assert json.loads(row["metadata"]) == {"user": "bob"}

    # ── Rating clamping ──────────────────────────────────────────────────

    def test_rating_1_not_clamped(self) -> None:
        row = transform_feedback_record(_make_feedback(rating=1), run_id="r")
        assert row["rating"] == 1

    def test_rating_5_not_clamped(self) -> None:
        row = transform_feedback_record(_make_feedback(rating=5), run_id="r")
        assert row["rating"] == 5

    def test_rating_above_5_clamped_to_5(self) -> None:
        row = transform_feedback_record(_make_feedback(rating=10), run_id="r")
        assert row["rating"] == 5

    def test_rating_below_1_clamped_to_1(self) -> None:
        row = transform_feedback_record(_make_feedback(rating=-3), run_id="r")
        assert row["rating"] == 1

    def test_rating_zero_clamped_to_1(self) -> None:
        row = transform_feedback_record(_make_feedback(rating=0), run_id="r")
        assert row["rating"] == 1

    def test_missing_rating_returns_sentinel_zero(self) -> None:
        """Absent rating → 0 (sentinel meaning 'not provided')."""
        fb = _make_feedback()
        del fb["rating"]
        row = transform_feedback_record(fb, run_id="r")
        assert row["rating"] == 0

    def test_invalid_rating_string_returns_sentinel_zero(self) -> None:
        row = transform_feedback_record(_make_feedback(rating="great"), run_id="r")
        assert row["rating"] == 0

    def test_rating_string_number_parsed(self) -> None:
        """A rating supplied as a numeric string ('4') should be accepted."""
        row = transform_feedback_record(_make_feedback(rating="4"), run_id="r")
        assert row["rating"] == 4

    # ── Missing optional fields ──────────────────────────────────────────

    def test_missing_comment_defaults_to_empty_string(self) -> None:
        fb = _make_feedback()
        del fb["comment"]
        row = transform_feedback_record(fb, run_id="r")
        assert row["comment"] == ""

    def test_missing_auto_quality_score_defaults_to_zero(self) -> None:
        fb = _make_feedback()
        del fb["auto_quality_score"]
        row = transform_feedback_record(fb, run_id="r")
        assert row["auto_quality_score"] == 0.0

    def test_missing_report_summary_defaults_to_empty_string(self) -> None:
        fb = _make_feedback()
        del fb["report_summary"]
        row = transform_feedback_record(fb, run_id="r")
        assert row["report_summary"] == ""

    def test_missing_metadata_defaults_to_empty_json_object(self) -> None:
        fb = _make_feedback()
        del fb["metadata"]
        row = transform_feedback_record(fb, run_id="r")
        assert json.loads(row["metadata"]) == {}

    # ── Timestamp is current UTC ─────────────────────────────────────────

    def test_timestamp_is_iso_string(self) -> None:
        row = transform_feedback_record(_make_feedback(), run_id="r")
        assert isinstance(row["timestamp"], str)
        # Must be parseable as ISO-8601
        parsed = datetime.fromisoformat(row["timestamp"])
        assert parsed.tzinfo is not None

    def test_timestamp_is_approximately_now(self) -> None:
        before = datetime.now(UTC)
        row = transform_feedback_record(_make_feedback(), run_id="r")
        after = datetime.now(UTC)
        parsed = datetime.fromisoformat(row["timestamp"])
        assert before <= parsed <= after
