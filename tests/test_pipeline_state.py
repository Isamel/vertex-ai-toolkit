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


# ── Public API surface check ─────────────────────────────────────────────────

def test_importable_from_vaig_core() -> None:
    from vaig.core import PipelineState as CorePipelineState
    from vaig.core import apply_state_patch as core_apply_state_patch

    assert CorePipelineState is PipelineState
    assert core_apply_state_patch is apply_state_patch
