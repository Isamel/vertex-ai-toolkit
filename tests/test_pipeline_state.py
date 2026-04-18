"""Unit tests for PipelineState and apply_state_patch (Phase 1: Core Models)."""

from __future__ import annotations

import pytest

from vaig.core.models import PipelineState, apply_state_patch


class TestPipelineState:
    """Tests for PipelineState model — creation, defaults, immutability."""

    def test_default_instantiation(self) -> None:
        state = PipelineState()
        assert state.findings == ()
        assert state.metrics == {}
        assert state.errors == ()

    def test_custom_instantiation(self) -> None:
        state = PipelineState(
            findings=[{"key": "value"}],
            metrics={"latency_ms": 120},
            errors=["oops"],
        )
        assert state.findings == ({"key": "value"},)
        assert state.metrics == {"latency_ms": 120}
        assert state.errors == ("oops",)

    def test_frozen_raises_on_assignment(self) -> None:
        state = PipelineState()
        with pytest.raises(Exception):  # ValidationError (frozen model)
            state.errors = ("mutating",)  # type: ignore[misc]

    def test_model_copy_produces_new_instance(self) -> None:
        state = PipelineState(errors=["first"])
        new_state = state.model_copy(update={"errors": (*state.errors, "second")})
        assert new_state.errors == ("first", "second")
        assert state.errors == ("first",)  # original unchanged

    def test_is_pydantic_base_model(self) -> None:
        from pydantic import BaseModel
        assert issubclass(PipelineState, BaseModel)


class TestApplyStatePatch:
    """Tests for apply_state_patch merge function."""

    # ── None handling ────────────────────────────────────────

    def test_none_state_returns_none(self) -> None:
        result = apply_state_patch(None, {"errors": ["x"]})
        assert result is None

    def test_none_patch_returns_state_unchanged(self) -> None:
        state = PipelineState(errors=["existing"])
        result = apply_state_patch(state, None)
        assert result is state  # same object — no copy needed

    def test_both_none_returns_none(self) -> None:
        result = apply_state_patch(None, None)
        assert result is None

    # ── Tuple field extension ─────────────────────────────────

    def test_findings_are_extended(self) -> None:
        state = PipelineState(findings=[{"a": 1}])
        result = apply_state_patch(state, {"findings": [{"b": 2}]})
        assert result is not None
        assert result.findings == ({"a": 1}, {"b": 2})

    def test_errors_are_extended(self) -> None:
        state = PipelineState(errors=["err1"])
        result = apply_state_patch(state, {"errors": ["err2", "err3"]})
        assert result is not None
        assert result.errors == ("err1", "err2", "err3")

    def test_list_fields_with_empty_patch_list(self) -> None:
        state = PipelineState(findings=[{"x": 1}])
        result = apply_state_patch(state, {"findings": []})
        assert result is not None
        assert result.findings == ({"x": 1},)

    # ── Dict field shallow merge ─────────────────────────────

    def test_metrics_shallow_merge_adds_new_keys(self) -> None:
        state = PipelineState(metrics={"a": 1})
        result = apply_state_patch(state, {"metrics": {"b": 2}})
        assert result is not None
        assert result.metrics == {"a": 1, "b": 2}

    def test_metrics_shallow_merge_patch_wins_on_conflict(self) -> None:
        state = PipelineState(metrics={"a": 1, "b": 99})
        result = apply_state_patch(state, {"metrics": {"b": 2}})
        assert result is not None
        assert result.metrics == {"a": 1, "b": 2}

    def test_metrics_empty_patch_dict_leaves_state_unchanged(self) -> None:
        state = PipelineState(metrics={"x": 42})
        result = apply_state_patch(state, {"metrics": {}})
        assert result is not None
        assert result.metrics == {"x": 42}

    # ── Immutability — original state untouched ──────────────

    def test_original_state_is_never_mutated(self) -> None:
        state = PipelineState(
            findings=[{"original": True}],
            metrics={"k": 1},
            errors=["e1"],
        )
        apply_state_patch(
            state,
            {
                "findings": [{"extra": True}],
                "metrics": {"k": 99, "new": 2},
                "errors": ["e2"],
            },
        )
        # Original state must be unchanged
        assert state.findings == ({"original": True},)
        assert state.metrics == {"k": 1}
        assert state.errors == ("e1",)

    # ── PipelineState as patch ───────────────────────────────

    def test_pipeline_state_instance_as_patch(self) -> None:
        state = PipelineState(errors=["base"])
        patch = PipelineState(errors=["extra"], metrics={"m": 1})
        result = apply_state_patch(state, patch)
        assert result is not None
        assert result.errors == ("base", "extra")
        assert result.metrics == {"m": 1}

    # ── Empty state + full patch ─────────────────────────────

    def test_empty_state_with_full_patch(self) -> None:
        state = PipelineState()
        result = apply_state_patch(
            state,
            {
                "findings": [{"issue": "high cpu"}],
                "metrics": {"cpu": 95},
                "errors": ["timeout"],
            },
        )
        assert result is not None
        assert result.findings == ({"issue": "high cpu"},)
        assert result.metrics == {"cpu": 95}
        assert result.errors == ("timeout",)

    # ── Patch with unknown keys ignored gracefully ───────────

    def test_patch_with_unknown_keys_ignored(self) -> None:
        state = PipelineState(errors=["e"])
        # Unknown keys in a plain-dict patch should not blow up — they are
        # simply not present in PipelineState and are discarded.
        result = apply_state_patch(state, {"errors": ["f"], "unknown_field": "ignored"})
        assert result is not None
        assert result.errors == ("e", "f")


class TestPipelineStateNewFields:
    """Tests for the new PipelineState fields (affected_resources, management_context, flags, agent_outputs)."""

    def test_default_new_fields(self) -> None:
        state = PipelineState()
        assert state.affected_resources == ()
        assert state.management_context == {}
        assert state.flags == {}
        assert state.agent_outputs == {}

    def test_custom_affected_resources(self) -> None:
        state = PipelineState(affected_resources=["pod/web-1", "svc/api"])
        assert state.affected_resources == ("pod/web-1", "svc/api")

    def test_custom_management_context(self) -> None:
        state = PipelineState(management_context={"cluster": "prod-us", "namespace": "default"})
        assert state.management_context == {"cluster": "prod-us", "namespace": "default"}

    def test_custom_flags(self) -> None:
        state = PipelineState(flags={"has_critical": True, "needs_restart": False})
        assert state.flags == {"has_critical": True, "needs_restart": False}

    def test_custom_agent_outputs(self) -> None:
        state = PipelineState(agent_outputs={"gatherer": "raw output text"})
        assert state.agent_outputs == {"gatherer": "raw output text"}

    def test_frozen_prevents_new_field_mutation(self) -> None:
        state = PipelineState(flags={"ok": True})
        with pytest.raises(Exception):
            state.flags = {"changed": False}  # type: ignore[misc]

    def test_model_copy_with_affected_resources(self) -> None:
        state = PipelineState(affected_resources=["pod/a"])
        new = state.model_copy(update={"affected_resources": (*state.affected_resources, "pod/b")})
        assert new.affected_resources == ("pod/a", "pod/b")
        assert state.affected_resources == ("pod/a",)

    def test_full_state_construction_with_all_fields(self) -> None:
        state = PipelineState(
            findings=[{"sev": "high"}],
            metrics={"cpu": 90},
            errors=["timeout"],
            affected_resources=["pod/x"],
            management_context={"cluster": "c1"},
            flags={"critical": True},
            agent_outputs={"agent_a": "output_a"},
        )
        assert len(state.findings) == 1
        assert state.metrics == {"cpu": 90}
        assert state.errors == ("timeout",)
        assert state.affected_resources == ("pod/x",)
        assert state.management_context == {"cluster": "c1"}
        assert state.flags == {"critical": True}
        assert state.agent_outputs == {"agent_a": "output_a"}


class TestToContextString:
    """Tests for PipelineState.to_context_string() method."""

    def test_empty_state_shows_zero_findings(self) -> None:
        state = PipelineState()
        ctx = state.to_context_string()
        assert "Findings: 0" in ctx

    def test_findings_count_shown(self) -> None:
        state = PipelineState(findings=[{"a": 1}, {"b": 2}, {"c": 3}])
        ctx = state.to_context_string()
        assert "Findings: 3" in ctx

    def test_affected_resources_shown(self) -> None:
        state = PipelineState(affected_resources=["pod/web", "svc/api"])
        ctx = state.to_context_string()
        assert "Affected resources: pod/web, svc/api" in ctx

    def test_flags_only_shows_active(self) -> None:
        state = PipelineState(flags={"has_critical": True, "needs_restart": False, "is_degraded": True})
        ctx = state.to_context_string()
        assert "Flags:" in ctx
        assert "has_critical" in ctx
        assert "is_degraded" in ctx
        assert "needs_restart" not in ctx

    def test_flags_all_false_omitted(self) -> None:
        state = PipelineState(flags={"ok": False, "done": False})
        ctx = state.to_context_string()
        assert "Flags:" not in ctx

    def test_management_context_shown(self) -> None:
        state = PipelineState(management_context={"cluster": "prod", "namespace": "web"})
        ctx = state.to_context_string()
        assert "Context:" in ctx
        assert "cluster=prod" in ctx
        assert "namespace=web" in ctx

    def test_errors_count_shown(self) -> None:
        state = PipelineState(errors=["err1", "err2"])
        ctx = state.to_context_string()
        assert "Errors: 2" in ctx

    def test_full_context_string(self) -> None:
        state = PipelineState(
            findings=[{"sev": "high"}],
            affected_resources=["pod/x"],
            flags={"critical": True},
            management_context={"cluster": "prod"},
            errors=["timeout"],
        )
        ctx = state.to_context_string()
        assert "Findings: 1" in ctx
        assert "Affected resources: pod/x" in ctx
        assert "Flags: critical" in ctx
        assert "Context: cluster=prod" in ctx
        assert "Errors: 1" in ctx

    def test_context_string_omits_empty_sections(self) -> None:
        state = PipelineState(findings=[{"a": 1}])
        ctx = state.to_context_string()
        assert "Affected resources:" not in ctx
        assert "Flags:" not in ctx
        assert "Context:" not in ctx
        assert "Errors:" not in ctx


class TestApplyStatePatchNewFields:
    """Tests for apply_state_patch with new fields."""

    def test_affected_resources_are_extended(self) -> None:
        state = PipelineState(affected_resources=["pod/a"])
        result = apply_state_patch(state, {"affected_resources": ["pod/b", "pod/c"]})
        assert result is not None
        assert result.affected_resources == ("pod/a", "pod/b", "pod/c")

    def test_management_context_shallow_merge(self) -> None:
        state = PipelineState(management_context={"cluster": "old"})
        result = apply_state_patch(state, {"management_context": {"cluster": "new", "ns": "web"}})
        assert result is not None
        assert result.management_context == {"cluster": "new", "ns": "web"}

    def test_flags_shallow_merge(self) -> None:
        state = PipelineState(flags={"has_critical": False})
        result = apply_state_patch(state, {"flags": {"has_critical": True, "done": True}})
        assert result is not None
        assert result.flags == {"has_critical": True, "done": True}

    def test_agent_outputs_shallow_merge(self) -> None:
        state = PipelineState(agent_outputs={"agent_a": "output_a"})
        result = apply_state_patch(state, {"agent_outputs": {"agent_b": "output_b"}})
        assert result is not None
        assert result.agent_outputs == {"agent_a": "output_a", "agent_b": "output_b"}

    def test_agent_outputs_overwrite_existing_key(self) -> None:
        state = PipelineState(agent_outputs={"agent_a": "old"})
        result = apply_state_patch(state, {"agent_outputs": {"agent_a": "new"}})
        assert result is not None
        assert result.agent_outputs == {"agent_a": "new"}

    def test_full_patch_with_all_new_fields(self) -> None:
        state = PipelineState()
        patch = {
            "findings": [{"sev": "high"}],
            "errors": ["err1"],
            "affected_resources": ["pod/x"],
            "management_context": {"cluster": "prod"},
            "flags": {"critical": True},
            "agent_outputs": {"gatherer": "data"},
            "metrics": {"cpu": 85},
        }
        result = apply_state_patch(state, patch)
        assert result is not None
        assert result.findings == ({"sev": "high"},)
        assert result.errors == ("err1",)
        assert result.affected_resources == ("pod/x",)
        assert result.management_context == {"cluster": "prod"}
        assert result.flags == {"critical": True}
        assert result.agent_outputs == {"gatherer": "data"}
        assert result.metrics == {"cpu": 85}

    def test_pipeline_state_as_patch_with_new_fields(self) -> None:
        state = PipelineState(affected_resources=["pod/a"], flags={"x": True})
        patch = PipelineState(affected_resources=["pod/b"], flags={"y": False})
        result = apply_state_patch(state, patch)
        assert result is not None
        assert result.affected_resources == ("pod/a", "pod/b")
        assert result.flags == {"x": True, "y": False}

    def test_original_state_untouched_with_new_fields(self) -> None:
        state = PipelineState(
            affected_resources=["pod/a"],
            management_context={"k": "v"},
            flags={"f": True},
            agent_outputs={"a": "1"},
        )
        apply_state_patch(state, {
            "affected_resources": ["pod/b"],
            "management_context": {"k": "new"},
            "flags": {"f": False, "g": True},
            "agent_outputs": {"b": "2"},
        })
        assert state.affected_resources == ("pod/a",)
        assert state.management_context == {"k": "v"}
        assert state.flags == {"f": True}
        assert state.agent_outputs == {"a": "1"}

    def test_non_sequence_affected_resources_ignored(self) -> None:
        state = PipelineState(affected_resources=["pod/a"])
        result = apply_state_patch(state, {"affected_resources": "not-a-list"})
        assert result is not None
        assert result.affected_resources == ("pod/a",)

    def test_non_dict_flags_ignored(self) -> None:
        state = PipelineState(flags={"ok": True})
        result = apply_state_patch(state, {"flags": "not-a-dict"})
        assert result is not None
        assert result.flags == {"ok": True}


class TestPipelineStateJsonRoundtrip:
    """Tests for JSON serialization roundtrip with new fields."""

    def test_json_roundtrip_with_all_fields(self) -> None:
        state = PipelineState(
            findings=[{"sev": "critical", "msg": "OOM"}],
            metrics={"cpu": 95, "memory_mb": 512},
            errors=["timeout", "dns failure"],
            affected_resources=["pod/web-1", "svc/api"],
            management_context={"cluster": "prod-us", "namespace": "default"},
            flags={"has_critical": True, "needs_restart": False},
            agent_outputs={"gatherer": "raw data", "analyzer": "analysis text"},
        )
        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored == state
        assert restored.affected_resources == ("pod/web-1", "svc/api")
        assert restored.management_context == {"cluster": "prod-us", "namespace": "default"}
        assert restored.flags == {"has_critical": True, "needs_restart": False}
        assert restored.agent_outputs == {"gatherer": "raw data", "analyzer": "analysis text"}

    def test_json_roundtrip_empty_state(self) -> None:
        state = PipelineState()
        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored == state
        assert restored.affected_resources == ()
        assert restored.management_context == {}
        assert restored.flags == {}
        assert restored.agent_outputs == {}


# ── Public API surface check ─────────────────────────────────────────────────

def test_importable_from_vaig_core() -> None:
    from vaig.core import PipelineState as CorePipelineState
    from vaig.core import apply_state_patch as core_apply_state_patch

    assert CorePipelineState is PipelineState
    assert core_apply_state_patch is apply_state_patch


# ── T-02: PipelineState.investigation_plan (SH-09) ──────────────────────────


class TestPipelineStateInvestigationPlan:
    """Tests for the investigation_plan field added in T-02 (SH-09)."""

    def test_default_is_none(self) -> None:
        state = PipelineState()
        assert state.investigation_plan is None

    def test_accepts_none_explicitly(self) -> None:
        state = PipelineState(investigation_plan=None)
        assert state.investigation_plan is None

    def test_accepts_arbitrary_dict(self) -> None:
        plan_data = {"plan_id": "test-plan", "steps": []}
        state = PipelineState(investigation_plan=plan_data)
        assert state.investigation_plan == plan_data

    def test_apply_state_patch_sets_investigation_plan(self) -> None:
        from vaig.skills.service_health.schema import InvestigationPlan

        state = PipelineState()
        plan = InvestigationPlan(plan_id="patch-plan", created_from="run-1", steps=[])
        patched = apply_state_patch(state, {"investigation_plan": plan})
        assert patched is not None
        assert patched.investigation_plan is plan

    def test_apply_state_patch_clears_investigation_plan_with_none(self) -> None:
        from vaig.skills.service_health.schema import InvestigationPlan

        plan = InvestigationPlan(plan_id="existing", created_from="run-0", steps=[])
        state = PipelineState(investigation_plan=plan)
        # Patching with None is ignored by apply_state_patch (guard: is not None)
        patched = apply_state_patch(state, {"investigation_plan": None})
        assert patched is not None
        assert patched.investigation_plan is plan  # unchanged — None patch is ignored

    def test_investigation_plan_json_roundtrip(self) -> None:
        plan_data = {"plan_id": "rtrip", "steps": ["s1", "s2"]}
        state = PipelineState(investigation_plan=plan_data)
        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored.investigation_plan == plan_data
