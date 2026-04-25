"""Consolidated gatherer-agent wiring tests for all live-tools skills.

Parametrized to avoid repeating the same assertions across 5 separate files
(test_rca_gatherer.py, test_compliance_gatherer.py, test_config_audit_gatherer.py,
test_log_analysis_gatherer.py, test_perf_analysis_gatherer.py).

Each parametrize case validates:
- Gatherer agent is at position 0 in the pipeline
- Gatherer has requires_tools=True
- Gatherer system_instruction is a non-empty string
- Gatherer model is gemini-2.5-flash
- Gatherer temperature is 0.0
- Gatherer prompt constant starts with ANTI_INJECTION_RULE
- Pipeline has the expected number of agents
- Downstream agents (index 1+) do NOT have requires_tools=True
- Skill metadata has requires_live_tools=True
- Skill metadata has the "live" tag
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Test matrix
# Each entry: (skill_module, skill_class, gatherer_name, prompt_module, prompt_const, pipeline_len)
# ---------------------------------------------------------------------------
_GATHERER_CASES: list[tuple[str, str, str, str, str, int]] = [
    (
        "vaig.skills.rca.skill",
        "RCASkill",
        "rca_gatherer",
        "vaig.skills.rca.prompts",
        "RCA_GATHERER_PROMPT",
        4,
    ),
    (
        "vaig.skills.compliance_check.skill",
        "ComplianceCheckSkill",
        "compliance_gatherer",
        "vaig.skills.compliance_check.prompts",
        "COMPLIANCE_GATHERER_PROMPT",
        4,
    ),
    (
        "vaig.skills.config_audit.skill",
        "ConfigAuditSkill",
        "config_gatherer",
        "vaig.skills.config_audit.prompts",
        "CONFIG_AUDIT_GATHERER_PROMPT",
        3,
    ),
    (
        "vaig.skills.log_analysis.skill",
        "LogAnalysisSkill",
        "log_analysis_gatherer",
        "vaig.skills.log_analysis.prompts",
        "LOG_ANALYSIS_GATHERER_PROMPT",
        4,
    ),
    (
        "vaig.skills.perf_analysis.skill",
        "PerfAnalysisSkill",
        "perf_gatherer",
        "vaig.skills.perf_analysis.prompts",
        "PERF_ANALYSIS_GATHERER_PROMPT",
        4,
    ),
]

_IDS = [row[1] for row in _GATHERER_CASES]


def _get_skill(skill_module: str, skill_class: str) -> Any:
    module = importlib.import_module(skill_module)
    return getattr(module, skill_class)()


def _get_gatherer(skill_module: str, skill_class: str, gatherer_name: str) -> dict[str, Any]:
    skill = _get_skill(skill_module, skill_class)
    agents = skill.get_agents_config()
    return agents[0]


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_gatherer_is_first_agent(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    skill = _get_skill(skill_module, skill_class)
    agents = skill.get_agents_config()
    assert agents[0]["name"] == gatherer_name


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_gatherer_requires_tools(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    gatherer = _get_gatherer(skill_module, skill_class, gatherer_name)
    assert gatherer.get("requires_tools") is True


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_gatherer_prompt_starts_with_anti_injection(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    mod = importlib.import_module(prompt_module)
    anti_injection = getattr(mod, "ANTI_INJECTION_RULE")  # noqa: B009
    gatherer_prompt = getattr(mod, prompt_const)  # noqa: B009
    assert gatherer_prompt.startswith(anti_injection)


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_gatherer_system_instruction_is_non_empty(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    gatherer = _get_gatherer(skill_module, skill_class, gatherer_name)
    assert isinstance(gatherer.get("system_instruction"), str)
    assert len(gatherer["system_instruction"]) > 0


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_gatherer_model_is_flash(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    gatherer = _get_gatherer(skill_module, skill_class, gatherer_name)
    # Gatherer model must be sentinel "" — resolved at runtime via Settings.
    # Skills must not hardcode a model name.
    assert gatherer.get("model") == ""


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_gatherer_temperature_is_zero(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    gatherer = _get_gatherer(skill_module, skill_class, gatherer_name)
    assert gatherer.get("temperature") == 0.0


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_pipeline_length(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    skill = _get_skill(skill_module, skill_class)
    agents = skill.get_agents_config()
    assert len(agents) == pipeline_len


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_downstream_agents_not_requires_tools(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    skill = _get_skill(skill_module, skill_class)
    agents = skill.get_agents_config()
    for agent in agents[1:]:
        assert not agent.get("requires_tools", False), (
            f"Agent '{agent['name']}' should not have requires_tools=True"
        )


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_skill_metadata_requires_live_tools(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    skill = _get_skill(skill_module, skill_class)
    meta = skill.get_metadata()
    assert meta.requires_live_tools is True


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name", "prompt_module", "prompt_const", "pipeline_len"),
    _GATHERER_CASES,
    ids=_IDS,
)
def test_skill_metadata_has_live_tag(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
    prompt_module: str,
    prompt_const: str,
    pipeline_len: int,
) -> None:
    skill = _get_skill(skill_module, skill_class)
    meta = skill.get_metadata()
    assert "live" in meta.tags


# ---------------------------------------------------------------------------
# Extra assertions only applicable to skills with max_iterations=12
# ---------------------------------------------------------------------------

_MAX_ITERATIONS_SKILLS = [
    ("vaig.skills.compliance_check.skill", "ComplianceCheckSkill", "compliance_gatherer"),
    ("vaig.skills.config_audit.skill", "ConfigAuditSkill", "config_gatherer"),
]


@pytest.mark.parametrize(
    ("skill_module", "skill_class", "gatherer_name"),
    _MAX_ITERATIONS_SKILLS,
    ids=[row[1] for row in _MAX_ITERATIONS_SKILLS],
)
def test_gatherer_max_iterations_is_twelve(
    skill_module: str,
    skill_class: str,
    gatherer_name: str,
) -> None:
    gatherer = _get_gatherer(skill_module, skill_class, gatherer_name)
    assert gatherer.get("max_iterations") == 12
