"""Tests for CheckOutput — string-only Terraform contract schema."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vaig.cli.check_schema import CheckOutput

# ── Fixtures ─────────────────────────────────────────────────


def _make_health_report(
    *,
    overall_status: str = "HEALTHY",
    critical_count: int = 0,
    warning_count: int = 0,
    issues_found: int = 0,
    services_checked: int = 5,
    summary_text: str = "All services healthy",
    scope: str = "Namespace: default",
) -> MagicMock:
    """Build a mock HealthReport with an executive_summary."""
    es = MagicMock()
    es.overall_status.value = overall_status
    es.critical_count = critical_count
    es.warning_count = warning_count
    es.issues_found = issues_found
    es.services_checked = services_checked
    es.summary_text = summary_text
    es.scope = scope

    report = MagicMock()
    report.executive_summary = es
    return report


# ── from_health_report() mapping ─────────────────────────────


class TestFromHealthReport:
    """Verify from_health_report() maps all fields correctly."""

    def test_maps_all_fields(self) -> None:
        report = _make_health_report(
            overall_status="CRITICAL",
            critical_count=3,
            warning_count=2,
            issues_found=5,
            services_checked=10,
            summary_text="Multiple services degraded",
            scope="Namespace: production",
        )
        output = CheckOutput.from_health_report(report)

        assert output.status == "CRITICAL"
        assert output.critical_count == "3"
        assert output.warning_count == "2"
        assert output.issues_found == "5"
        assert output.services_checked == "10"
        assert output.summary_text == "Multiple services degraded"
        assert output.scope == "Namespace: production"
        assert output.cached == "false"
        assert output.version  # non-empty

    def test_cached_flag_true(self) -> None:
        report = _make_health_report()
        output = CheckOutput.from_health_report(report, cached=True)
        assert output.cached == "true"

    def test_cached_flag_false(self) -> None:
        report = _make_health_report()
        output = CheckOutput.from_health_report(report, cached=False)
        assert output.cached == "false"

    def test_timestamp_is_iso_format(self) -> None:
        report = _make_health_report()
        output = CheckOutput.from_health_report(report)
        # ISO 8601 timestamps contain 'T' and end with timezone info
        assert "T" in output.timestamp


# ── String-only serialisation ────────────────────────────────


class TestStringOnlyValues:
    """Every serialised JSON value MUST be a string — no int, bool, nested."""

    def test_all_values_are_strings(self) -> None:
        report = _make_health_report(critical_count=7, services_checked=42)
        output = CheckOutput.from_health_report(report)
        data = json.loads(output.model_dump_json())

        for key, value in data.items():
            assert isinstance(value, str), (
                f"Field '{key}' has type {type(value).__name__}, expected str"
            )

    def test_numeric_fields_are_string_representations(self) -> None:
        report = _make_health_report(critical_count=0, warning_count=12)
        output = CheckOutput.from_health_report(report)
        data = output.model_dump()

        assert data["critical_count"] == "0"
        assert data["warning_count"] == "12"

    def test_json_roundtrip_produces_flat_map(self) -> None:
        """Terraform external data source requires a flat map of strings."""
        report = _make_health_report()
        output = CheckOutput.from_health_report(report)
        raw_json = output.model_dump_json()
        parsed = json.loads(raw_json)

        # No nested objects or arrays
        for key, value in parsed.items():
            assert not isinstance(value, (dict, list)), (
                f"Field '{key}' contains a nested structure"
            )


# ── from_error() factory ─────────────────────────────────────


class TestFromError:
    """Verify from_error() produces valid schema for error/timeout states."""

    def test_timeout_error(self) -> None:
        output = CheckOutput.from_error("TIMEOUT", "Health check timed out after 120s")

        assert output.status == "TIMEOUT"
        assert output.summary_text == "Health check timed out after 120s"
        assert output.critical_count == "0"
        assert output.cached == "false"

    def test_generic_error(self) -> None:
        output = CheckOutput.from_error("ERROR", "Connection refused")

        assert output.status == "ERROR"
        assert output.summary_text == "Connection refused"

    def test_error_output_is_all_strings(self) -> None:
        output = CheckOutput.from_error("ERROR", "Something went wrong")
        data = json.loads(output.model_dump_json())

        for key, value in data.items():
            assert isinstance(value, str), (
                f"Error field '{key}' has type {type(value).__name__}, expected str"
            )

    def test_error_type_uppercased(self) -> None:
        output = CheckOutput.from_error("timeout", "msg")
        assert output.status == "TIMEOUT"


# ── Schema stability ─────────────────────────────────────────


class TestSchemaStability:
    """Internal HealthReport changes must not break CheckOutput."""

    def test_extra_executive_summary_fields_ignored(self) -> None:
        """If ExecutiveSummary adds new fields, from_health_report still works."""
        report = _make_health_report()
        # Simulate a new field added to executive_summary
        report.executive_summary.new_future_field = "some value"

        # Should not raise — we only access known fields
        output = CheckOutput.from_health_report(report)
        assert output.status == "HEALTHY"

    def test_output_field_set_is_stable(self) -> None:
        """The set of field names must match the expected contract."""
        expected_fields = {
            "status",
            "critical_count",
            "warning_count",
            "issues_found",
            "services_checked",
            "summary_text",
            "scope",
            "timestamp",
            "version",
            "cached",
        }
        assert set(CheckOutput.model_fields.keys()) == expected_fields
