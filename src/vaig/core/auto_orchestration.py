"""Auto-activation policy for dynamically enabling agent capabilities.

Provides :class:`AutoActivationPolicy` for registering capabilities with
activation conditions and modes, and evaluating them against runtime context.

Usage::

    from vaig.core.auto_orchestration import (
        ActivationMode,
        AutoActivationPolicy,
        CapabilityEntry,
    )

    policy = AutoActivationPolicy()
    policy.register(
        CapabilityEntry(
            name="k8s_diagnostics",
            condition=lambda ctx: "cluster" in ctx,
            feature_flag="k8s_enabled",
            mode=ActivationMode.AUTO_TRIGGERED,
        )
    )
    enabled = policy.evaluate({"cluster": "prod-us-east"})
    # enabled == {"k8s_diagnostics"} if the feature flag is not disabling it
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ActivationMode(StrEnum):
    """Controls how a capability is activated.

    Attributes:
        AUTO_ALWAYS: Capability is always active regardless of context.
        AUTO_TRIGGERED: Capability activates when its condition function returns True.
        AUTO_ON_INPUT: Capability activates when the context contains an ``input`` key.
        OPT_IN: Capability is disabled by default; must be explicitly enabled.
    """

    AUTO_ALWAYS = "auto_always"
    AUTO_TRIGGERED = "auto_triggered"
    AUTO_ON_INPUT = "auto_on_input"
    OPT_IN = "opt_in"


@dataclass
class CapabilityEntry:
    """Descriptor for a single auto-activatable capability.

    Attributes:
        name: Unique capability identifier.
        condition: Callable that accepts the runtime context dict and returns
            ``True`` when this capability should activate.
        feature_flag: String key looked up in the runtime context to check
            whether the capability has been explicitly disabled.  When the
            context contains ``{feature_flag: False}``, the capability is
            suppressed regardless of other settings.
        mode: :class:`ActivationMode` controlling activation logic.
    """

    name: str
    condition: Callable[[dict[str, Any]], bool]
    feature_flag: str
    mode: ActivationMode = ActivationMode.AUTO_TRIGGERED


class AutoActivationPolicy:
    """Registry and evaluator for auto-activatable capabilities.

    Capabilities are registered with a name, condition, feature flag, and
    activation mode.  At runtime, :meth:`evaluate` returns the set of
    capability names that are active given the provided context.

    Example::

        policy = AutoActivationPolicy()
        policy.register(CapabilityEntry(
            name="summariser",
            condition=lambda ctx: len(ctx.get("text", "")) > 5000,
            feature_flag="summariser_enabled",
            mode=ActivationMode.AUTO_TRIGGERED,
        ))
        active = policy.evaluate({"text": "very long text ..."})
    """

    def __init__(self) -> None:
        self._entries: dict[str, CapabilityEntry] = {}

    def register(self, entry: CapabilityEntry) -> None:
        """Register a capability.

        If a capability with the same name is already registered, it is
        replaced and a debug log is emitted.

        Args:
            entry: :class:`CapabilityEntry` to register.
        """
        if entry.name in self._entries:
            logger.debug("AutoActivationPolicy: overwriting capability %r", entry.name)
        self._entries[entry.name] = entry

    def evaluate(self, context: dict[str, Any]) -> set[str]:
        """Return the set of capability names that are active for *context*.

        Evaluation rules (applied in order):
        1. If the capability's ``feature_flag`` key is present in *context*
           and its value is falsy, the capability is suppressed.
        2. ``OPT_IN`` capabilities activate only when ``context["opt_in"][capability_name]`` is True.
        3. ``AUTO_ALWAYS`` capabilities are always active (unless suppressed by rule 1).
        4. ``AUTO_ON_INPUT`` capabilities activate when ``context.get("has_user_input") is True``.
        5. ``AUTO_TRIGGERED`` capabilities activate when ``entry.condition(context)`` is True.

        Args:
            context: Runtime context mapping (e.g., agent state, input flags).

        Returns:
            Set of active capability names.
        """
        active: set[str] = set()
        for name, _entry in self._entries.items():
            if not self.is_enabled(name, context):
                continue
            active.add(name)
        return active

    def set_mode(self, capability: str, mode: ActivationMode) -> None:
        """Update the activation mode for a registered capability.

        Args:
            capability: Name of the capability to update.
            mode: New :class:`ActivationMode`.

        Raises:
            ValueError: If *capability* is not registered.
        """
        if capability not in self._entries:
            raise ValueError(f"Capability {capability!r} is not registered")
        self._entries[capability].mode = mode

    def is_enabled(self, capability: str, context: dict[str, Any]) -> bool:
        """Return True if *capability* is active in *context*.

        Applies the same evaluation rules as :meth:`evaluate` for a single
        named capability.

        Args:
            capability: Name of the capability to check.
            context: Runtime context mapping.

        Returns:
            True if active, False if suppressed or unknown.
        """
        entry = self._entries.get(capability)
        if entry is None:
            logger.debug("AutoActivationPolicy.is_enabled: unknown capability %r", capability)
            return False

        # Rule 1: feature flag suppression
        flag_value = context.get(entry.feature_flag)
        if flag_value is not None and not flag_value:
            return False

        mode = entry.mode

        # Rule 2: OPT_IN — only activate when explicitly opted-in via context
        if mode == ActivationMode.OPT_IN:
            return context.get("opt_in", {}).get(capability) is True

        # Rule 3: AUTO_ALWAYS — always active
        if mode == ActivationMode.AUTO_ALWAYS:
            return True

        # Rule 4: AUTO_ON_INPUT — active when "has_user_input" is True
        if mode == ActivationMode.AUTO_ON_INPUT:
            return context.get("has_user_input") is True

        # Rule 5: AUTO_TRIGGERED — delegate to condition
        try:
            return bool(entry.condition(context))
        except Exception:
            logger.exception(
                "AutoActivationPolicy: condition for capability %r raised an exception; "
                "treating as inactive",
                capability,
            )
            return False
