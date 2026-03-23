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

    ``findings`` and ``errors`` use ``tuple`` instead of ``list`` to prevent
    accidental mutation when the same :class:`PipelineState` instance is
    broadcast to multiple agents in parallel/fanout strategies.

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


# ── Merge utility ─────────────────────────────────────────────────────────────


def apply_state_patch(
    state: PipelineState | None,
    patch: dict[str, Any] | BaseModel | None,
) -> PipelineState | None:
    """Merge *patch* into *state* and return a new :class:`PipelineState`.

    Merge strategy:
    - **List/tuple fields** (``findings``, ``errors``): items from *patch* are
      *appended* to the existing tuple.  The patch value must be a ``list``
      or ``tuple``; other types are silently ignored.
    - **Dict fields** (``metrics``): a shallow ``update`` is performed so
      patch keys overwrite / add to existing keys.  The patch value must be a
      ``dict``; other types are silently ignored.
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

    # List/tuple fields — extend (append new items)
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

    # Dict fields — shallow merge (patch keys win)
    # Defensive: only merge when the patch value is actually a dict.
    raw_metrics = patch_dict.get("metrics", {})
    if isinstance(raw_metrics, dict):
        new_metrics: dict[str, Any] = {**state.metrics, **raw_metrics}
    else:
        new_metrics = dict(state.metrics)

    return state.model_copy(
        update={
            "findings": new_findings,
            "metrics": new_metrics,
            "errors": new_errors,
        }
    )
