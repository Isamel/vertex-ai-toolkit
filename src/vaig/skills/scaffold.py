"""Skill scaffolding — generate boilerplate for new custom skills."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.skills._presets import SkillPreset


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

_MULTI_AGENT_PROMPTS_TEMPLATE = '''"""{display_name} Skill — prompts."""

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
    "validate": f"""## Phase: Validation

Validate findings against the original context and user request.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Cross-check findings for accuracy
2. Verify recommendations are actionable
3. Flag any inconsistencies or gaps
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

_MULTI_AGENT_SKILL_TEMPLATE = '''"""{display_name} Skill — {description}."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from {prompts_import} import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class {class_name}(BaseSkill):
    """{display_name} skill (multi-agent).

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
                {supported_phases}
            ],
            recommended_model="gemini-2.5-pro",
            requires_live_tools={requires_live_tools},
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return agent configurations for multi-agent execution."""
        return [
            {agent_configs}
        ]
'''

_CODING_PROMPTS_TEMPLATE = '''"""{display_name} Skill — prompts."""

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
    "analyze": f"""## Phase: Analyze

Understand the problem and gather relevant information.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Identify the core problem or requirement
2. List relevant files, modules, or components
3. Note any constraints or dependencies
""",
    "plan": f"""## Phase: Plan

Create a detailed implementation plan.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Break down the task into concrete steps
2. Identify risks and edge cases
3. Define acceptance criteria
""",
    "execute": f"""## Phase: Execute

Implement the planned changes.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Implement each step from the plan
2. Follow project coding conventions
3. Add appropriate error handling
""",
    "validate": f"""## Phase: Validate

Verify the implementation is correct.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Review the implementation against the plan
2. Check for edge cases and error handling
3. Verify test coverage
""",
    "report": f"""## Phase: Report

Summarize the implementation and results.

### Context
{{DELIMITER_DATA_START}}
{{{{context}}}}
{{DELIMITER_DATA_END}}

### User Request
{{{{user_input}}}}

### Instructions
1. Summarize changes made
2. List files modified
3. Note any remaining tasks or follow-ups
""",
}}
'''

_CODING_SKILL_TEMPLATE = '''"""{display_name} Skill — {description}."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from {prompts_import} import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class {class_name}(BaseSkill):
    """{display_name} skill (planner + executor).

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
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.VALIDATE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return planner + executor agent configs."""
        meta = self.get_metadata()
        return [
            {{
                "name": f"{{meta.name}}-planner",
                "role": "Planner",
                "system_instruction": self.get_system_instruction(),
                "model": meta.recommended_model,
            }},
            {{
                "name": f"{{meta.name}}-executor",
                "role": "Executor",
                "system_instruction": self.get_system_instruction(),
                "model": meta.recommended_model,
            }},
        ]
'''

_TEST_TEMPLATE = '''"""Tests for {display_name} skill."""

from __future__ import annotations

from {skill_import} import {class_name}


class Test{class_name}:
    """Basic tests for {class_name}."""

    def test_metadata(self) -> None:
        skill = {class_name}()
        meta = skill.get_metadata()
        assert meta.name == "{skill_name}"
        assert meta.display_name == "{display_name}"
        assert len(meta.supported_phases) > 0

    def test_system_instruction(self) -> None:
        skill = {class_name}()
        instruction = skill.get_system_instruction()
        assert isinstance(instruction, str)
        assert len(instruction) > 0
'''

_README_TEMPLATE = '''# {display_name}

{description}

## Usage

```python
from vaig.skills.registry import SkillRegistry

registry = SkillRegistry(settings)
skill = registry.get("{skill_name}")
```

## Phases

{phases_list}

## Customization

Edit `skill.py` and `prompts.py` to implement your skill logic.
'''

_SCHEMA_TEMPLATE = '''"""{display_name} Skill — input/output schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class {class_name_no_suffix}Input(BaseModel):
    """Input schema for {display_name}."""

    query: str = Field(..., description="The user query or request")
    # TODO: Add additional input fields as needed


class {class_name_no_suffix}Output(BaseModel):
    """Output schema for {display_name}."""

    result: str = Field(..., description="The analysis result")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    # TODO: Add additional output fields as needed
'''


def scaffold_skill(
    name: str,
    target_dir: Path,
    *,
    description: str = "A custom skill",
    tags: list[str] | None = None,
    preset: SkillPreset | None = None,
) -> Path:
    """Create the boilerplate files for a new skill.

    Args:
        name: Skill name (kebab-case, snake_case, or mixed).
        target_dir: Parent directory where the skill subdirectory will be created.
        description: One-line description for the skill.
        tags: Optional list of tags for skill routing.
        preset: Optional preset determining template and file generation.
            When ``None``, the default analysis template is used (backward compat).

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

    # Determine the import path for prompts
    prompts_import = f"vaig.skills.{snake}.prompts"
    try:
        skill_dir.relative_to(Path(__file__).parent)
    except ValueError:
        prompts_import = ".prompts"

    # Determine the import path for skill (used by test template)
    skill_import = f"vaig.skills.{snake}.skill"
    try:
        skill_dir.relative_to(Path(__file__).parent)
    except ValueError:
        skill_import = ".skill"

    # --- When no preset is given, produce the exact same output as before ---
    if preset is None:
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

        # skill.py — original single-agent template
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

    # --- Preset-based scaffolding ---

    # __init__.py (same for all presets)
    (skill_dir / "__init__.py").write_text(
        _INIT_TEMPLATE.format(display_name=display_name, description=description),
        encoding="utf-8",
    )

    # Select and write prompts.py + skill.py based on preset
    if preset.name == "coding":
        # Coding: 5-phase planner+executor dual-agent
        (skill_dir / "prompts.py").write_text(
            _CODING_PROMPTS_TEMPLATE.format(display_name=display_name),
            encoding="utf-8",
        )
        (skill_dir / "skill.py").write_text(
            _CODING_SKILL_TEMPLATE.format(
                display_name=display_name,
                description=description,
                class_name=class_name,
                skill_name=kebab,
                tags=tags_repr,
                prompts_import=prompts_import,
            ),
            encoding="utf-8",
        )
    elif preset.agent_count >= 2:  # noqa: PLR2004
        # Multi-agent (e.g. live-tools)
        (skill_dir / "prompts.py").write_text(
            _MULTI_AGENT_PROMPTS_TEMPLATE.format(display_name=display_name),
            encoding="utf-8",
        )

        # Build agent config lines
        agent_config_lines = []
        for role in preset.agent_roles:
            agent_config_lines.append(
                "{"
                f'"name": "{kebab}-{role}", '
                f'"role": "{role.capitalize()}", '
                '"system_instruction": self.get_system_instruction(), '
                '"model": self.get_metadata().recommended_model'
                "},"
            )
        agent_configs = "\n            ".join(agent_config_lines)

        # Build supported_phases lines
        phase_lines = [f"SkillPhase.{p.name}," for p in preset.phases]
        supported_phases = "\n                ".join(phase_lines)

        (skill_dir / "skill.py").write_text(
            _MULTI_AGENT_SKILL_TEMPLATE.format(
                display_name=display_name,
                description=description,
                class_name=class_name,
                skill_name=kebab,
                tags=tags_repr,
                supported_phases=supported_phases,
                requires_live_tools=str(preset.requires_live_tools),
                prompts_import=prompts_import,
                agent_configs=agent_configs,
            ),
            encoding="utf-8",
        )
    else:
        # Single-agent preset (e.g. analysis)
        (skill_dir / "prompts.py").write_text(
            _PROMPTS_TEMPLATE.format(display_name=display_name),
            encoding="utf-8",
        )
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

    # --- Additional files (always generated with presets) ---

    # test_<snake>.py
    (skill_dir / f"test_{snake}.py").write_text(
        _TEST_TEMPLATE.format(
            display_name=display_name,
            class_name=class_name,
            skill_name=kebab,
            skill_import=skill_import,
        ),
        encoding="utf-8",
    )

    # Remove the class suffix for schema naming (e.g. "MyToolSkill" → "MyTool")
    class_name_no_suffix = class_name.removesuffix("Skill")

    # README.md
    phases_list = "\n".join(f"- {p.value}" for p in preset.phases)
    (skill_dir / "README.md").write_text(
        _README_TEMPLATE.format(
            display_name=display_name,
            description=description,
            skill_name=kebab,
            phases_list=phases_list,
        ),
        encoding="utf-8",
    )

    # schema.py (only if preset requires it)
    if preset.generate_schema:
        (skill_dir / "schema.py").write_text(
            _SCHEMA_TEMPLATE.format(
                display_name=display_name,
                class_name_no_suffix=class_name_no_suffix,
            ),
            encoding="utf-8",
        )

    return skill_dir
