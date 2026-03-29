"""Core domain models — shared state and merge utilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineState(BaseModel):
    """Typed shared state that flows through the agent pipeline.

    Phase 1: Infrastructure only — fields are empty containers by default.
    Phase 2+: Skills populate via ``get_initial_state()`` and agents
    write incremental deltas via ``AgentResult.state_patch``.

    The model is *immutable* (``frozen=True``).  Use ``model_copy(update=...)``
    to produce a new instance with changed fields — never mutate in-place.

    ``findings``, ``errors``, and ``affected_resources`` use ``tuple`` instead
    of ``list`` to prevent accidental mutation when the same
    :class:`PipelineState` instance is broadcast to multiple agents in
    parallel/fanout strategies.

    Example::

        state = PipelineState()
        new_state = state.model_copy(update={"errors": (*state.errors, "boom")})
    """

    model_config = ConfigDict(frozen=True)

    findings: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    """Structured findings emitted by agents during the pipeline run."""

    metrics: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary numeric or categorical metrics accumulated across agents."""

    errors: tuple[str, ...] = Field(default_factory=tuple)
    """Non-fatal error messages collected during the pipeline run."""

    affected_resources: tuple[str, ...] = Field(default_factory=tuple)
    """Resource identifiers (e.g. pod names, service URIs) affected by findings."""

    management_context: dict[str, str] = Field(default_factory=dict)
    """Key-value pairs of management/operational context (e.g. cluster, namespace)."""

    flags: dict[str, bool] = Field(default_factory=dict)
    """Boolean flags set during the pipeline (e.g. ``needs_restart``, ``has_critical``)."""

    agent_outputs: dict[str, str] = Field(default_factory=dict)
    """Maps agent name → raw output text for downstream consumption."""

    def to_context_string(self) -> str:
        """Serialize the state into a human-readable string for prompt injection.

        Returns a concise multi-line summary covering findings count,
        affected resources, active flags, management context, and errors.
        """
        parts: list[str] = []

        parts.append(f"Findings: {len(self.findings)}")

        if self.affected_resources:
            parts.append(
                f"Affected resources: {', '.join(str(r) for r in self.affected_resources if r is not None)}"
            )

        if self.flags:
            active = [k for k, v in sorted(self.flags.items(), key=lambda kv: str(kv[0])) if v]
            if active:
                parts.append(f"Flags: {', '.join(active)}")

        if self.management_context:
            ctx_items = [
                f"{str(k)}={str(v)}"
                for k, v in sorted(self.management_context.items(), key=lambda kv: str(kv[0]))
            ]
            parts.append(f"Context: {'; '.join(ctx_items)}")

        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")

        return "\n".join(parts)


# ── Merge utility ─────────────────────────────────────────────────────────────


def apply_state_patch(
    state: PipelineState | None,
    patch: dict[str, Any] | BaseModel | None,
) -> PipelineState | None:
    """Merge *patch* into *state* and return a new :class:`PipelineState`.

    Merge strategy:
    - **Tuple fields** (``findings``, ``errors``, ``affected_resources``):
      items from *patch* are *appended* to the existing tuple.  The patch
      value must be a ``list`` or ``tuple``; other types are silently ignored.
    - **Dict fields** (``metrics``, ``management_context``, ``flags``,
      ``agent_outputs``): a shallow ``update`` is performed so patch keys
      overwrite / add to existing keys.  The patch value must be a ``dict``;
      other types are silently ignored.
    - **None handling**: if *state* is ``None`` the function returns ``None``;
      if *patch* is ``None`` the original *state* is returned unchanged.

    The original *state* object is never mutated.

    Args:
        state: Current pipeline state, or ``None`` when state is not in use.
        patch: A plain ``dict``, a :class:`~pydantic.BaseModel` instance
            (e.g. :class:`PipelineState`), or ``None`` if the agent did not
            emit any state delta.

    Returns:
        A new :class:`PipelineState` with the patch applied, or ``None`` /
        the original state as described above.
    """
    if state is None:
        return None
    if patch is None:
        return state

    # Normalise patch to a plain dict
    if isinstance(patch, BaseModel):
        patch_dict: dict[str, Any] = patch.model_dump()
    else:
        patch_dict = dict(patch)

    # ── Tuple fields — extend (append new items) ─────────────────────────
    # Defensive: only extend when the patch value is actually a list/tuple.

    raw_findings = patch_dict.get("findings", ())
    if isinstance(raw_findings, (list, tuple)):
        new_findings: tuple[dict[str, Any], ...] = tuple(state.findings) + tuple(raw_findings)
    else:
        new_findings = tuple(state.findings)

    raw_errors = patch_dict.get("errors", ())
    if isinstance(raw_errors, (list, tuple)):
        new_errors: tuple[str, ...] = tuple(state.errors) + tuple(raw_errors)
    else:
        new_errors = tuple(state.errors)

    raw_affected = patch_dict.get("affected_resources", ())
    if isinstance(raw_affected, (list, tuple)):
        new_affected: tuple[str, ...] = tuple(state.affected_resources) + tuple(
            str(item) for item in raw_affected if item is not None
        )
    else:
        new_affected = tuple(state.affected_resources)

    # ── Dict fields — shallow merge (patch keys win) ─────────────────────
    # Defensive: only merge when the patch value is actually a dict.

    raw_metrics = patch_dict.get("metrics", {})
    if isinstance(raw_metrics, dict):
        new_metrics: dict[str, Any] = {**state.metrics, **raw_metrics}
    else:
        new_metrics = dict(state.metrics)

    raw_mgmt = patch_dict.get("management_context", {})
    if isinstance(raw_mgmt, dict):
        new_mgmt: dict[str, str] = {**state.management_context, **raw_mgmt}
    else:
        new_mgmt = dict(state.management_context)

    raw_flags = patch_dict.get("flags", {})
    if isinstance(raw_flags, dict):
        new_flags: dict[str, bool] = {**state.flags, **raw_flags}
    else:
        new_flags = dict(state.flags)

    raw_outputs = patch_dict.get("agent_outputs", {})
    if isinstance(raw_outputs, dict):
        new_outputs: dict[str, str] = {**state.agent_outputs, **raw_outputs}
    else:
        new_outputs = dict(state.agent_outputs)

    return state.model_copy(
        update={
            "findings": new_findings,
            "metrics": new_metrics,
            "errors": new_errors,
            "affected_resources": new_affected,
            "management_context": new_mgmt,
            "flags": new_flags,
            "agent_outputs": new_outputs,
        }
    )
