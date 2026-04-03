"""Dynamic agent routing based on capability keyword matching.

Routes parallel gatherer agents based on the user query, reducing token
spend and latency on focused queries.  Only configs that explicitly declare
``parallel_group`` AND ``capabilities`` are subject to filtering — all other
configs (sequential agents, or gatherers without a capabilities list) pass
through unconditionally.

Matching strategy: tokenize the query into lowercase alphanumeric words,
then apply a bidirectional substring check against each capability keyword.
A capability *c* matches a query token *t* when ``c in t`` OR ``t in c``
(case-insensitive, both already lowercased).

Safe-all fallback: if no parallel gatherers match, ALL original configs are
returned unchanged.  This prevents accidentally skipping every agent on
queries that are too short or use unexpected phrasing.

Example::

    >>> configs = [
    ...     {"name": "node_gatherer",     "parallel_group": "gather", "capabilities": ["node", "cpu"]},
    ...     {"name": "logging_gatherer",  "parallel_group": "gather", "capabilities": ["log", "logging"]},
    ...     {"name": "analyzer",          "parallel_group": None},
    ... ]
    >>> route_agents("check logs for pod X", configs)
    [{"name": "logging_gatherer", ...}, {"name": "analyzer", ...}]
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.core.effectiveness import ToolEffectivenessService

logger = logging.getLogger(__name__)

# ── Tokenisation ─────────────────────────────────────────────────────────────
# Matches runs of lowercase alphanumeric characters.  Punctuation, spaces, and
# other separators are treated as token boundaries.  This intentionally uses a
# simple approach (no stemming) to keep the router dependency-free.
_RE_QUERY_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(query: str) -> list[str]:
    """Tokenize a query string into lowercase alphanumeric words.

    Args:
        query: The raw user input string.

    Returns:
        A list of lowercase word tokens.  Empty list if query is blank.
    """
    return _RE_QUERY_TOKEN.findall(query.lower())


def _is_gatherer(config: dict[str, Any]) -> bool:
    """Return True if *config* is a parallel gatherer with declared capabilities.

    A config is considered a *filterable gatherer* when it has BOTH:
    * ``parallel_group`` key present AND truthy (marks it as a parallel agent), AND
    * ``capabilities`` key present AND is a non-empty list of at least one string.

    Configs missing either key, or with an empty/non-list capabilities value,
    are treated as pass-through (not filtered).
    """
    if not config.get("parallel_group"):
        return False
    caps = config.get("capabilities")
    if not isinstance(caps, list):
        return False
    return any(isinstance(c, str) and c for c in caps)


def _gatherer_matches(config: dict[str, Any], tokens: list[str]) -> bool:
    """Return True if any query token matches any capability of this gatherer.

    Bidirectional substring match (case-insensitive, both sides lowercased):
    - token is a substring of capability, OR
    - capability is a substring of token.

    This catches:
    * "log" matches capability "logs" (capability contains token)
    * "logging" matches capability "log" (token contains capability)
    * "pods" matches capability "pod" (token contains capability)
    * "pod" matches capability "pods" (capability contains token)

    Non-string items in capabilities are skipped (coerced out).

    Args:
        config: An agent config dict.
        tokens:  Lowercase query tokens from :func:`_tokenize`.

    Returns:
        True if at least one token ↔ capability pair matches.
    """
    capabilities: list[str] = [c.lower() for c in config.get("capabilities", []) if isinstance(c, str)]
    return any(cap in token or token in cap for token in tokens for cap in capabilities)


def route_agents(
    query: str,
    configs: list[dict[str, Any]],
    effectiveness_service: ToolEffectivenessService | None = None,
) -> list[dict[str, Any]]:
    """Filter agent configs to those relevant to the given query.

    Only parallel gatherer configs (those with **both** ``parallel_group``
    and ``capabilities`` keys) are subject to filtering.  Sequential agents
    and gatherers without a ``capabilities`` list always pass through.

    When no parallel gatherers match (safe-all fallback), ALL original
    configs are returned unchanged — this prevents accidentally skipping
    every agent.

    When *effectiveness_service* is provided, matched gatherers whose
    declared tools are ALL in SKIP tier are removed.  Gatherers with
    mixed tiers (some SKIP, some non-SKIP) are kept.  This implements
    R-EFF-05.

    Args:
        query:   The user's input query string.
        configs: List of agent configuration dicts as returned by
                 ``BaseSkill.get_agents_config()``.
        effectiveness_service: Optional effectiveness scorer — when ``None``
            (default) effectiveness filtering is skipped entirely.

    Returns:
        A filtered list of configs.  Never empty when *configs* is non-empty
        (safe-all fallback guarantees at least all configs are returned).
    """
    if not configs:
        return configs

    tokens = _tokenize(query)

    # Split configs into filterable gatherers vs pass-through
    gatherers: list[dict[str, Any]] = []
    pass_through: list[dict[str, Any]] = []

    for config in configs:
        if _is_gatherer(config):
            gatherers.append(config)
        else:
            pass_through.append(config)

    # If there are no filterable gatherers, return all configs unchanged.
    if not gatherers:
        return configs

    # If the query is empty (no tokens), trigger safe-all fallback.
    if not tokens:
        return configs

    # Filter gatherers by capability match.
    matched_gatherers = [g for g in gatherers if _gatherer_matches(g, tokens)]

    # Safe-all fallback: no gatherers matched → return ALL original configs.
    if not matched_gatherers:
        return configs

    # ── Effectiveness filtering (R-EFF-05) ────────────────────
    if effectiveness_service is not None:
        matched_gatherers = _filter_skip_gatherers(matched_gatherers, effectiveness_service)
        # If effectiveness removed ALL matched gatherers, fall back to all configs.
        if not matched_gatherers:
            return configs

    # Return matched gatherers + all pass-through configs (preserve original order).
    # Use id() for both sets to avoid O(N²) equality checks.
    matched_set = {id(g) for g in matched_gatherers} | {id(c) for c in pass_through}
    return [c for c in configs if id(c) in matched_set]


def _filter_skip_gatherers(
    gatherers: list[dict[str, Any]],
    effectiveness_service: ToolEffectivenessService,
) -> list[dict[str, Any]]:
    """Remove gatherers whose declared tools are ALL in SKIP tier.

    A gatherer's tools are inferred from its ``capabilities`` list.
    Gatherers with at least one non-SKIP tool are kept.
    """
    from vaig.core.effectiveness import EffectivenessTier

    kept: list[dict[str, Any]] = []
    for gatherer in gatherers:
        capabilities: list[str] = [
            c for c in gatherer.get("capabilities", []) if isinstance(c, str)
        ]
        if not capabilities:
            # No declared capabilities — keep (can't evaluate)
            kept.append(gatherer)
            continue

        # TODO: capabilities are query keywords, not tool names — effectiveness
        # scoring by capability is an approximation. A future refactor should
        # map gatherer capabilities to actual tool names for accurate filtering.
        all_skip = all(
            effectiveness_service.get_tool_score(cap).tier == EffectivenessTier.SKIP
            for cap in capabilities
        )
        if all_skip:
            logger.info(
                "Filtering gatherer %s — all tools (%s) are SKIP tier",
                gatherer.get("name", "?"),
                ", ".join(capabilities),
            )
        else:
            kept.append(gatherer)

    return kept
