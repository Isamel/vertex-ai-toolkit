"""Skill scaffolding — generate boilerplate for new custom skills."""

from __future__ import annotations

import re
from pathlib import Path


def _to_snake_case(name: str) -> str:
    """Convert a kebab-case or mixed name to snake_case."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip("-_")).strip("_")


def _to_class_name(name: str) -> str:
    """Convert a kebab-case or mixed name to PascalCaseSkill."""
    parts = re.split(r"[^a-zA-Z0-9]", name)
    pascal = "".join(p.capitalize() for p in parts if p)
    return f"{pascal}Skill"


def _to_kebab_case(name: str) -> str:
    """Convert a snake_case or mixed name to kebab-case."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip("-_")).strip("-")


_INIT_TEMPLATE = '''"""{display_name} Skill — {description}."""
'''

_PROMPTS_TEMPLATE = '''"""{display_name} Skill — prompts."""

from vaig.core.prompt_defense import (
    ANTI_INJECTION_RULE,
    DELIMITER_DATA_END,
    DELIMITER_DATA_START,
)

SYSTEM_INSTRUCTION = f"""{{ANTI_INJECTION_RULE}}

You are a specialist in {display_name}.

## Your Expertise
- TODO: List your areas of expertise

## Analysis Framework
- TODO: Describe your analysis methodology

## Output Standards
- Provide clear, actionable recommendations
- Include confidence levels (High/Medium/Low) for findings
- Distinguish between CONFIRMED facts and HYPOTHESIZED causes
"""

PHASE_PROMPTS = {{
    "analyze": f"""## Phase: Initial Analysis

Analyze the following input and identify key areas of concern.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Review the provided context
2. Identify patterns, issues, or opportunities
3. Prioritize findings by severity/impact
""",
    "execute": f"""## Phase: Deep Analysis

Perform a detailed analysis based on the initial findings.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Deep-dive into each identified area
2. Provide specific, evidence-based findings
3. Suggest concrete remediation steps
""",
    "report": f"""## Phase: Final Report

Generate a comprehensive report with findings and recommendations.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Summarize all findings
2. Prioritize recommendations (P0/P1/P2)
3. Include estimated effort for each recommendation
""",
}}
'''

_SKILL_TEMPLATE = '''"""{display_name} Skill — {description}."""

from __future__ import annotations

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from {prompts_import} import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class {class_name}(BaseSkill):
    """{display_name} skill.

    TODO: Add a detailed description of what this skill does,
    what kind of input it expects, and what output it produces.
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="{skill_name}",
            display_name="{display_name}",
            description="{description}",
            version="1.0.0",
            tags={tags},
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.EXECUTE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)
'''


def scaffold_skill(
    name: str,
    target_dir: Path,
    *,
    description: str = "A custom skill",
    tags: list[str] | None = None,
) -> Path:
    """Create the boilerplate files for a new skill.

    Args:
        name: Skill name (kebab-case, snake_case, or mixed).
        target_dir: Parent directory where the skill subdirectory will be created.
        description: One-line description for the skill.
        tags: Optional list of tags for skill routing.

    Returns:
        Path to the created skill directory.

    Raises:
        FileExistsError: If the skill directory already exists.
    """
    snake = _to_snake_case(name)
    kebab = _to_kebab_case(name)
    class_name = _to_class_name(name)
    display_name = " ".join(w.capitalize() for w in kebab.split("-"))

    skill_dir = target_dir / snake
    if skill_dir.exists():
        msg = f"Skill directory already exists: {skill_dir}"
        raise FileExistsError(msg)

    skill_dir.mkdir(parents=True)

    tag_list = tags or [kebab]
    tags_repr = repr(tag_list)

    # __init__.py
    (skill_dir / "__init__.py").write_text(
        _INIT_TEMPLATE.format(display_name=display_name, description=description),
        encoding="utf-8",
    )

    # prompts.py
    (skill_dir / "prompts.py").write_text(
        _PROMPTS_TEMPLATE.format(display_name=display_name),
        encoding="utf-8",
    )

    # skill.py — determine the import path
    # If inside vaig package, use dotted import; otherwise use relative
    prompts_import = f"vaig.skills.{snake}.prompts"
    try:
        # Check if we're inside the vaig package tree
        skill_dir.relative_to(Path(__file__).parent)
    except ValueError:
        # Custom skill outside vaig package — use relative import
        prompts_import = ".prompts"

    (skill_dir / "skill.py").write_text(
        _SKILL_TEMPLATE.format(
            display_name=display_name,
            description=description,
            class_name=class_name,
            skill_name=kebab,
            tags=tags_repr,
            prompts_import=prompts_import,
        ),
        encoding="utf-8",
    )

    return skill_dir
