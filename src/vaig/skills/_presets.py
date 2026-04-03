"""Skill preset definitions — map preset names to phase lists, agent topologies, and file flags."""

from __future__ import annotations

from dataclasses import dataclass, field

from vaig.skills.base import SkillPhase


@dataclass(frozen=True)
class SkillPreset:
    """Immutable preset describing a skill archetype.

    Each preset defines which phases the skill supports, how many agents
    it spawns, what roles those agents play, and which extra files the
    scaffolder should generate.
    """

    name: str
    phases: list[SkillPhase] = field(default_factory=list)
    agent_count: int = 1
    agent_roles: list[str] = field(default_factory=list)
    generate_schema: bool = False
    requires_live_tools: bool = False


PRESETS: dict[str, SkillPreset] = {
    "analysis": SkillPreset(
        name="analysis",
        phases=[SkillPhase.ANALYZE, SkillPhase.EXECUTE, SkillPhase.REPORT],
        agent_count=1,
        agent_roles=["analyst"],
        generate_schema=False,
        requires_live_tools=False,
    ),
    "live-tools": SkillPreset(
        name="live-tools",
        phases=[SkillPhase.ANALYZE, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT],
        agent_count=3,
        agent_roles=["gatherer", "analyzer", "reporter"],
        generate_schema=True,
        requires_live_tools=True,
    ),
    "coding": SkillPreset(
        name="coding",
        phases=[SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT],
        agent_count=2,
        agent_roles=["planner", "executor"],
        generate_schema=True,
        requires_live_tools=False,
    ),
}


def get_preset(name: str) -> SkillPreset:
    """Return a preset by name.

    Raises:
        ValueError: If *name* is not a known preset, listing valid options.
    """
    try:
        return PRESETS[name]
    except KeyError:
        valid = ", ".join(sorted(PRESETS))
        msg = f"Unknown preset {name!r}. Valid presets: {valid}"
        raise ValueError(msg) from None
