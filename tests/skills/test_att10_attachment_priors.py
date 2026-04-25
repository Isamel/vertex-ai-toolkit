"""Tests for SPEC-ATT-10 §6.5.1 — AttachmentPriors model and extractor.

Covers:
- Schema: all sub-models (ResourceSpec, ProbeSpec, Hotspot, HistoricalIncident,
  ChangeSignal, NarrativeHint, AttachmentPriors)
- HealthReport.attachment_priors field presence and exclusion from Gemini schema
- Extractor: fingerprint, cache, parse_priors_json, extract_priors
"""

from __future__ import annotations

import json

import pytest

from vaig.core.attachment_priors_extractor import (
    clear_cache,
    fingerprint,
    get_cached,
    parse_priors_json,
    set_cached,
)
from vaig.skills.service_health.schema import (
    AttachmentPriors,
    ChangeSignal,
    HealthReport,
    HealthReportGeminiSchema,
    HistoricalIncident,
    Hotspot,
    NarrativeHint,
    ProbeSpec,
    ResourceSpec,
)

# ── ResourceSpec ─────────────────────────────────────────────


class TestResourceSpec:
    def test_defaults_are_empty_strings(self) -> None:
        rs = ResourceSpec()
        assert rs.cpu_request == ""
        assert rs.cpu_limit == ""
        assert rs.memory_request == ""
        assert rs.memory_limit == ""

    def test_all_fields_populated(self) -> None:
        rs = ResourceSpec(
            cpu_request="250m",
            cpu_limit="500m",
            memory_request="128Mi",
            memory_limit="512Mi",
        )
        assert rs.cpu_request == "250m"
        assert rs.memory_limit == "512Mi"

    def test_extra_fields_ignored(self) -> None:
        rs = ResourceSpec.model_validate({"cpu_request": "100m", "unknown": "x"})
        assert rs.cpu_request == "100m"
        assert not hasattr(rs, "unknown")


# ── ProbeSpec ────────────────────────────────────────────────


class TestProbeSpec:
    def test_defaults(self) -> None:
        ps = ProbeSpec()
        assert ps.readiness_path == ""
        assert ps.liveness_path == ""
        assert ps.initial_delay_seconds is None
        assert ps.timeout_seconds is None

    def test_full_spec(self) -> None:
        ps = ProbeSpec(
            readiness_path="/ready",
            liveness_path="/live",
            initial_delay_seconds=10,
            timeout_seconds=5,
            period_seconds=15,
            failure_threshold=3,
        )
        assert ps.readiness_path == "/ready"
        assert ps.failure_threshold == 3


# ── Hotspot ──────────────────────────────────────────────────


class TestHotspot:
    def test_required_fields(self) -> None:
        h = Hotspot(entity="DB pool", concern="oversaturation")
        assert h.entity == "DB pool"
        assert h.concern == "oversaturation"
        assert h.source_ref == ""
        assert h.description == ""

    def test_with_source_ref(self) -> None:
        h = Hotspot(entity="cache", concern="eviction", source_ref="runbook.md:L42")
        assert h.source_ref == "runbook.md:L42"


# ── HistoricalIncident ───────────────────────────────────────


class TestHistoricalIncident:
    def test_required_symptom_pattern(self) -> None:
        inc = HistoricalIncident(symptom_pattern="OOM kills + latency spike")
        assert inc.symptom_pattern == "OOM kills + latency spike"
        assert inc.root_cause == ""
        assert inc.fix_applied == ""
        assert inc.date == ""

    def test_full_incident(self) -> None:
        inc = HistoricalIncident(
            symptom_pattern="503s on checkout",
            root_cause="DB pool exhaustion",
            fix_applied="Increased pool size to 50",
            source_ref="postmortem-2026-02.md:§3",
            date="2026-02-18",
        )
        assert inc.fix_applied == "Increased pool size to 50"
        assert inc.date == "2026-02-18"


# ── ChangeSignal ─────────────────────────────────────────────


class TestChangeSignal:
    def test_required_field_path(self) -> None:
        cs = ChangeSignal(field_path="resources.limits.memory")
        assert cs.field_path == "resources.limits.memory"
        assert cs.old_value == ""
        assert cs.new_value == ""

    def test_with_values(self) -> None:
        cs = ChangeSignal(
            field_path="resources.limits.memory",
            old_value="512Mi",
            new_value="256Mi",
            source_ref="values.after.yaml:L22",
        )
        assert cs.old_value == "512Mi"
        assert cs.new_value == "256Mi"


# ── NarrativeHint ────────────────────────────────────────────


class TestNarrativeHint:
    def test_required_hint(self) -> None:
        nh = NarrativeHint(hint="Check DB pool when latency spikes")
        assert nh.hint == "Check DB pool when latency spikes"
        assert nh.source_ref == ""

    def test_with_source_ref(self) -> None:
        nh = NarrativeHint(hint="Restart pod if OOM", source_ref="runbook.md:L99")
        assert nh.source_ref == "runbook.md:L99"


# ── AttachmentPriors ─────────────────────────────────────────


class TestAttachmentPriors:
    def test_empty_defaults(self) -> None:
        ap = AttachmentPriors()
        assert ap.expected_versions == {}
        assert ap.expected_replica_counts == {}
        assert ap.expected_env_vars == {}
        assert ap.expected_resource_limits == {}
        assert ap.expected_probes == {}
        assert ap.runbook_hotspots == []
        assert ap.historical_incidents == []
        assert ap.change_signals == []
        assert ap.narrative_hints == []

    def test_full_priors(self) -> None:
        ap = AttachmentPriors(
            expected_versions={"checkout-service": "2.3.1"},
            expected_replica_counts={"checkout-deployment": 3},
            expected_env_vars={"checkout": {"LOG_LEVEL": "info"}},
            expected_resource_limits={"checkout": ResourceSpec(cpu_limit="500m", memory_limit="512Mi")},
            expected_probes={"checkout": ProbeSpec(readiness_path="/ready", timeout_seconds=5)},
            runbook_hotspots=[Hotspot(entity="DB pool", concern="oversaturation")],
            historical_incidents=[HistoricalIncident(symptom_pattern="OOM kills")],
            change_signals=[ChangeSignal(field_path="memory.limit", old_value="512Mi", new_value="256Mi")],
            narrative_hints=[NarrativeHint(hint="check pool metrics")],
        )
        assert ap.expected_versions["checkout-service"] == "2.3.1"
        assert ap.expected_replica_counts["checkout-deployment"] == 3
        assert ap.runbook_hotspots[0].entity == "DB pool"
        assert ap.change_signals[0].new_value == "256Mi"

    def test_json_roundtrip(self) -> None:
        ap = AttachmentPriors(
            expected_versions={"svc": "1.0"},
            runbook_hotspots=[Hotspot(entity="cache", concern="eviction")],
        )
        dumped = json.loads(ap.model_dump_json())
        restored = AttachmentPriors.model_validate(dumped)
        assert restored.expected_versions["svc"] == "1.0"
        assert restored.runbook_hotspots[0].concern == "eviction"

    def test_extra_fields_ignored(self) -> None:
        ap = AttachmentPriors.model_validate({"expected_versions": {"x": "1"}, "unknown": True})
        assert ap.expected_versions["x"] == "1"


# ── HealthReport.attachment_priors ───────────────────────────


class TestHealthReportAttachmentPriors:
    _MINIMAL: dict = {
        "executive_summary": {
            "overall_status": "HEALTHY",
            "scope": "cluster",
            "summary_text": "ok",
        }
    }

    def test_defaults_to_none(self) -> None:
        report = HealthReport.model_validate(self._MINIMAL)
        assert report.attachment_priors is None

    def test_accepts_attachment_priors(self) -> None:
        ap = AttachmentPriors(expected_versions={"svc": "2.0"})
        payload = {**self._MINIMAL, "attachment_priors": ap.model_dump()}
        report = HealthReport.model_validate(payload)
        assert report.attachment_priors is not None
        assert report.attachment_priors.expected_versions["svc"] == "2.0"

    def test_excluded_from_gemini_schema(self) -> None:
        schema = HealthReportGeminiSchema.model_json_schema()
        props = schema.get("properties", {})
        assert "attachment_priors" not in props, "attachment_priors must be excluded from the Gemini response_schema"


# ── Extractor: fingerprint ────────────────────────────────────


class TestFingerprint:
    def test_same_input_same_output(self) -> None:
        assert fingerprint("hello") == fingerprint("hello")

    def test_different_input_different_output(self) -> None:
        assert fingerprint("hello") != fingerprint("world")

    def test_length_is_16(self) -> None:
        assert len(fingerprint("any text")) == 16

    def test_hex_chars_only(self) -> None:
        fp = fingerprint("test")
        assert all(c in "0123456789abcdef" for c in fp)


# ── Extractor: cache ──────────────────────────────────────────


class TestCache:
    def setup_method(self) -> None:
        clear_cache()

    def test_miss_returns_none(self) -> None:
        assert get_cached("nonexistent") is None

    def test_set_and_get(self) -> None:
        ap = AttachmentPriors(expected_versions={"svc": "1.0"})
        set_cached("fp123", ap)
        result = get_cached("fp123")
        assert result is not None
        assert result.expected_versions["svc"] == "1.0"

    def test_clear_empties_cache(self) -> None:
        ap = AttachmentPriors()
        set_cached("fp456", ap)
        clear_cache()
        assert get_cached("fp456") is None


# ── Extractor: parse_priors_json ──────────────────────────────


class TestParsePriorsJson:
    def test_plain_json(self) -> None:
        raw = json.dumps({"expected_versions": {"svc": "3.0"}})
        ap = parse_priors_json(raw)
        assert ap.expected_versions["svc"] == "3.0"

    def test_json_with_markdown_fence(self) -> None:
        raw = '```json\n{"expected_replica_counts": {"dep": 2}}\n```'
        ap = parse_priors_json(raw)
        assert ap.expected_replica_counts["dep"] == 2

    def test_json_with_plain_fence(self) -> None:
        raw = '```\n{"runbook_hotspots": [{"entity": "db", "concern": "slow"}]}\n```'
        ap = parse_priors_json(raw)
        assert ap.runbook_hotspots[0].entity == "db"

    def test_empty_object(self) -> None:
        ap = parse_priors_json("{}")
        assert ap.expected_versions == {}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_priors_json("not json at all")


# ── Extractor: extract_priors with mock client ───────────────


class _MockResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockClient:
    def __init__(self, response: str, *, fail: bool = False) -> None:
        self._response = response
        self._fail = fail
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: object) -> _MockResult:
        self.call_count += 1
        if self._fail:
            raise RuntimeError("LLM unavailable")
        return _MockResult(self._response)


class TestExtractPriors:
    def setup_method(self) -> None:
        clear_cache()

    def test_extracts_from_valid_response(self) -> None:
        from vaig.core.attachment_priors_extractor import extract_priors

        client = _MockClient(json.dumps({"expected_versions": {"checkout": "1.2.3"}}))
        priors = extract_priors("some attachment text", client)
        assert priors.expected_versions["checkout"] == "1.2.3"

    def test_second_call_uses_cache(self) -> None:
        from vaig.core.attachment_priors_extractor import extract_priors

        client = _MockClient(json.dumps({"expected_versions": {"svc": "1.0"}}))
        text = "identical attachment text"
        extract_priors(text, client)
        extract_priors(text, client)
        assert client.call_count == 1  # second call should be cached

    def test_different_text_calls_llm_again(self) -> None:
        from vaig.core.attachment_priors_extractor import extract_priors

        client = _MockClient(json.dumps({}))
        extract_priors("text A", client)
        extract_priors("text B", client)
        assert client.call_count == 2

    def test_llm_failure_returns_empty_priors(self) -> None:
        from vaig.core.attachment_priors_extractor import extract_priors

        client = _MockClient("", fail=True)
        priors = extract_priors("attachment text", client)
        assert priors.expected_versions == {}
        assert priors.runbook_hotspots == []

    def test_bad_json_response_returns_empty_priors(self) -> None:
        from vaig.core.attachment_priors_extractor import extract_priors

        client = _MockClient("this is not json")
        priors = extract_priors("attachment text", client)
        assert priors.expected_versions == {}
