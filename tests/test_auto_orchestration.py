"""Tests for AutoActivationPolicy, ActivationMode, and CapabilityEntry."""

from __future__ import annotations

import pytest

from vaig.core.auto_orchestration import (
    ActivationMode,
    AutoActivationPolicy,
    CapabilityEntry,
)


class TestActivationModeEnum:
    """ActivationMode is a StrEnum with the four required members."""

    def test_auto_always_value(self) -> None:
        assert ActivationMode.AUTO_ALWAYS == "auto_always"

    def test_auto_triggered_value(self) -> None:
        assert ActivationMode.AUTO_TRIGGERED == "auto_triggered"

    def test_auto_on_input_value(self) -> None:
        assert ActivationMode.AUTO_ON_INPUT == "auto_on_input"

    def test_opt_in_value(self) -> None:
        assert ActivationMode.OPT_IN == "opt_in"

    def test_is_str_enum(self) -> None:
        assert isinstance(ActivationMode.AUTO_ALWAYS, str)


class TestCapabilityEntry:
    """CapabilityEntry is a dataclass with the required fields."""

    def test_defaults(self) -> None:
        entry = CapabilityEntry(
            name="test",
            condition=lambda ctx: True,
            feature_flag="test_flag",
        )
        assert entry.mode == ActivationMode.AUTO_TRIGGERED

    def test_custom_mode(self) -> None:
        entry = CapabilityEntry(
            name="test",
            condition=lambda ctx: False,
            feature_flag="flag",
            mode=ActivationMode.AUTO_ALWAYS,
        )
        assert entry.mode == ActivationMode.AUTO_ALWAYS


class TestAutoActivationPolicyRegister:
    """register() adds and replaces capabilities."""

    def test_register_adds_capability(self) -> None:
        policy = AutoActivationPolicy()
        entry = CapabilityEntry("cap1", lambda ctx: True, "flag1")
        policy.register(entry)
        assert policy.is_enabled("cap1", {"flag1": True}) is True

    def test_register_replaces_existing(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(CapabilityEntry("cap", lambda ctx: True, "f"))
        policy.register(CapabilityEntry("cap", lambda ctx: False, "f"))
        assert policy.is_enabled("cap", {}) is False


class TestAutoActivationPolicyEvaluate:
    """evaluate() returns the set of active capability names."""

    def test_evaluate_empty_returns_empty_set(self) -> None:
        policy = AutoActivationPolicy()
        assert policy.evaluate({}) == set()

    def test_auto_always_always_active(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry("always", lambda ctx: False, "f", ActivationMode.AUTO_ALWAYS)
        )
        assert "always" in policy.evaluate({})

    def test_auto_triggered_active_when_condition_true(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry(
                "triggered",
                lambda ctx: "key" in ctx,
                "f",
                ActivationMode.AUTO_TRIGGERED,
            )
        )
        assert "triggered" in policy.evaluate({"key": "value"})
        assert "triggered" not in policy.evaluate({})

    def test_auto_on_input_active_when_input_key_present(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry(
                "on_input",
                lambda ctx: False,
                "f",
                ActivationMode.AUTO_ON_INPUT,
            )
        )
        assert "on_input" in policy.evaluate({"has_user_input": True})
        assert "on_input" not in policy.evaluate({"has_user_input": False})
        assert "on_input" not in policy.evaluate({})

    def test_opt_in_never_auto_active(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry(
                "opt",
                lambda ctx: True,
                "f",
                ActivationMode.OPT_IN,
            )
        )
        assert "opt" not in policy.evaluate({})
        assert "opt" not in policy.evaluate({"f": True})

    def test_opt_in_activates_when_explicitly_opted_in(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry(
                "my_cap",
                lambda ctx: True,
                "ff_x",
                ActivationMode.OPT_IN,
            )
        )
        # Active when opt_in key is set for the capability
        assert "my_cap" in policy.evaluate({"ff_x": True, "opt_in": {"my_cap": True}})
        # Inactive without opt_in key
        assert "my_cap" not in policy.evaluate({"ff_x": True})
        # Inactive when opt_in key is False
        assert "my_cap" not in policy.evaluate({"ff_x": True, "opt_in": {"my_cap": False}})
        # Feature flag suppression still applies
        assert "my_cap" not in policy.evaluate({"ff_x": False, "opt_in": {"my_cap": True}})

    def test_feature_flag_false_suppresses_capability(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry(
                "flagged",
                lambda ctx: True,
                "my_flag",
                ActivationMode.AUTO_ALWAYS,
            )
        )
        # Suppressed when flag is falsy
        assert "flagged" not in policy.evaluate({"my_flag": False})
        # Active when flag is truthy
        assert "flagged" in policy.evaluate({"my_flag": True})
        # Active when flag is absent
        assert "flagged" in policy.evaluate({})

    def test_condition_exception_treated_as_inactive(self) -> None:
        def bad_condition(ctx: dict) -> bool:
            raise RuntimeError("boom")

        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry("boom_cap", bad_condition, "f", ActivationMode.AUTO_TRIGGERED)
        )
        # Should not raise — exception is caught internally
        result = policy.evaluate({})
        assert "boom_cap" not in result


class TestSetMode:
    """set_mode() updates an existing capability's activation mode."""

    def test_set_mode_changes_behavior(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(
            CapabilityEntry("cap", lambda ctx: False, "f", ActivationMode.AUTO_TRIGGERED)
        )
        assert "cap" not in policy.evaluate({})
        policy.set_mode("cap", ActivationMode.AUTO_ALWAYS)
        assert "cap" in policy.evaluate({})

    def test_set_mode_unknown_capability_raises_key_error(self) -> None:
        policy = AutoActivationPolicy()
        with pytest.raises(ValueError, match="not registered"):
            policy.set_mode("ghost", ActivationMode.AUTO_ALWAYS)


class TestIsEnabled:
    """is_enabled() applies same rules as evaluate() for a single capability."""

    def test_unknown_capability_returns_false(self) -> None:
        policy = AutoActivationPolicy()
        assert policy.is_enabled("nonexistent", {}) is False

    def test_auto_always_returns_true(self) -> None:
        policy = AutoActivationPolicy()
        policy.register(CapabilityEntry("c", lambda ctx: False, "f", ActivationMode.AUTO_ALWAYS))
        assert policy.is_enabled("c", {}) is True
