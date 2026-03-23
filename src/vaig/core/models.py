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

    Example::

        state = PipelineState()
        new_state = state.model_copy(update={"errors": [*state.errors, "boom"]})
    """

    model_config = ConfigDict(frozen=True)

    findings: list[dict[str, Any]] = Field(default_factory=list)
    """Structured findings emitted by agents during the pipeline run."""

    metrics: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary numeric or categorical metrics accumulated across agents."""

    errors: list[str] = Field(default_factory=list)
    """Non-fatal error messages collected during the pipeline run."""


# ── Merge utility ─────────────────────────────────────────────────────────────


def apply_state_patch(
    state: PipelineState | None,
    patch: dict[str, Any] | PipelineState | None,
) -> PipelineState | None:
    """Merge *patch* into *state* and return a new :class:`PipelineState`.

    Merge strategy:
    - **List fields** (``findings``, ``errors``): items from *patch* are
      *appended* to the existing list.
    - **Dict fields** (``metrics``): a shallow ``update`` is performed so
      patch keys overwrite / add to existing keys.
    - **None handling**: if *state* is ``None`` the function returns ``None``;
      if *patch* is ``None`` the original *state* is returned unchanged.

    The original *state* object is never mutated.

    Args:
        state: Current pipeline state, or ``None`` when state is not in use.
        patch: A plain ``dict``, a :class:`PipelineState` instance, or
            ``None`` if the agent did not emit any state delta.

    Returns:
        A new :class:`PipelineState` with the patch applied, or ``None`` /
        the original state as described above.
    """
    if state is None:
        return None
    if patch is None:
        return state

    # Normalise patch to a plain dict
    if isinstance(patch, PipelineState):
        patch_dict: dict[str, Any] = patch.model_dump()
    else:
        patch_dict = dict(patch)

    # List fields — extend (append new items)
    new_findings = list(state.findings) + list(patch_dict.get("findings", []))
    new_errors = list(state.errors) + list(patch_dict.get("errors", []))

    # Dict fields — shallow merge (patch keys win)
    new_metrics = {**state.metrics, **patch_dict.get("metrics", {})}

    return state.model_copy(
        update={
            "findings": new_findings,
            "metrics": new_metrics,
            "errors": new_errors,
        }
    )
