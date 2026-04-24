from __future__ import annotations

"""Tests: attachment_context parameter for all 7 prompt builders (G2)."""

import pytest

from vaig.skills.service_health.prompts._gatherer import build_gatherer_prompt
from vaig.skills.service_health.prompts._reporter import build_reporter_prompt
from vaig.skills.service_health.prompts._sub_gatherers import (
    build_datadog_gatherer_prompt,
    build_event_gatherer_prompt,
    build_logging_gatherer_prompt,
    build_node_gatherer_prompt,
    build_workload_gatherer_prompt,
)

# ---------------------------------------------------------------------------
# Builder callables with minimal required args.
# NOTE: build_datadog_gatherer_prompt returns "" when datadog_api_enabled=False,
# so we pass datadog_api_enabled=True to get a real prompt to test the prefix.
# ---------------------------------------------------------------------------

_BUILDERS = [
    ("build_gatherer_prompt", lambda **kw: build_gatherer_prompt(**kw)),
    ("build_reporter_prompt", lambda **kw: build_reporter_prompt(**kw)),
    ("build_node_gatherer_prompt", lambda **kw: build_node_gatherer_prompt(**kw)),
    ("build_workload_gatherer_prompt", lambda **kw: build_workload_gatherer_prompt(**kw)),
    (
        "build_datadog_gatherer_prompt",
        lambda **kw: build_datadog_gatherer_prompt(datadog_api_enabled=True, **kw),
    ),
    ("build_event_gatherer_prompt", lambda **kw: build_event_gatherer_prompt(**kw)),
    ("build_logging_gatherer_prompt", lambda **kw: build_logging_gatherer_prompt(**kw)),
]

_BUILDER_IDS = [name for name, _ in _BUILDERS]
_BUILDER_FNS = [fn for _, fn in _BUILDERS]


@pytest.mark.parametrize("builder", _BUILDER_FNS, ids=_BUILDER_IDS)
def test_builder_identity_when_no_context(builder) -> None:  # type: ignore[type-arg]
    """attachment_context=None → output equals output with no kwarg at all."""
    without = builder()
    with_none = builder(attachment_context=None)
    assert without == with_none


@pytest.mark.parametrize("builder", _BUILDER_FNS, ids=_BUILDER_IDS)
def test_builder_prefixes_when_context_provided(builder) -> None:  # type: ignore[type-arg]
    """attachment_context='UNIQUE_MARKER_42' → marker and header appear in result."""
    marker = "UNIQUE_MARKER_42"
    result = builder(attachment_context=marker)
    assert marker in result
    assert "## Attached Context" in result
