"""Code Migration Skill implementation."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.migration.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class MigrationSkill(BaseSkill):
    """Code Migration skill for migrating ETL pipelines and applications between platforms.

    Specialized in Pentaho → AWS Glue migrations but supports any source→target migration.
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="migration",
            display_name="Code Migration",
            description="Migrate code and ETL pipelines between platforms (Pentaho → AWS Glue, etc.)",
            version="1.0.0",
            tags=["migration", "etl", "pentaho", "aws-glue", "pyspark", "code-generation"],
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

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "code_analyzer",
                "role": "Source Code Analyzer",
                "system_instruction": (
                    "You are a code analysis specialist. Your job is to deeply understand "
                    "source code, ETL definitions (Pentaho KTR/KJB XML), and data pipelines. "
                    "Extract: data flows, transformations, business logic, dependencies, "
                    "database connections, and variables. Produce a structured analysis."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "code_generator",
                "role": "Target Code Generator",
                "system_instruction": (
                    "You are a code generation specialist for cloud data platforms. "
                    "Generate production-ready PySpark/AWS Glue code from source analysis. "
                    "Follow AWS best practices: DynamicFrame usage, bookmark management, "
                    "pushdown predicates, error handling, CloudWatch logging. "
                    "Code must be clean, documented, and testable."
                ),
                "model": "gemini-2.5-pro",
            },
            {
                "name": "migration_validator",
                "role": "Migration Validator",
                "system_instruction": (
                    "You are a code migration validation specialist. Compare source and "
                    "target code for functional equivalence. Identify gaps, behavioral "
                    "differences, and edge cases. Generate test cases and validation queries."
                ),
                "model": "gemini-2.5-flash",
            },
        ]
