"""Database Review Skill — schema, query, and operational database analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.db_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class DbReviewSkill(BaseSkill):
    """Database Review skill for schema, query, and operational database analysis.

    Supports multi-agent execution with specialized agents:
    - Query Analyzer: Execution plans, N+1 detection, index utilization
    - Schema Reviewer: Normalization, constraints, data types, partitioning
    - Database Lead: Synthesizes into operational review report
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="db-review",
            display_name="Database Review",
            description=(
                "Review database schemas, queries, and execution plans for performance "
                "issues, design problems, and operational risks"
            ),
            version="1.0.0",
            tags=[
                "database",
                "sql",
                "performance",
                "schema",
                "queries",
                "optimization",
            ],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.VALIDATE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "query_analyzer",
                "role": "Query Performance Analyzer",
                "system_instruction": (
                    "You are a database query performance specialist. Your job is to evaluate "
                    "query execution plans (EXPLAIN / EXPLAIN ANALYZE output) to identify full "
                    "table scans on large tables, missing index usage, suboptimal join strategies "
                    "(nested loop where hash join would be better), unnecessary sort operations, "
                    "hash aggregate spills to disk, and sequential scans that should be index "
                    "scans. Detect N+1 query patterns by analyzing query frequency and similarity. "
                    "Identify lock contention risks from long-running queries, DDL operations on "
                    "hot tables, and missing row-level locking hints. Quantify estimated vs actual "
                    "row counts to detect stale statistics causing plan regression."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "schema_reviewer",
                "role": "Schema Design Reviewer",
                "system_instruction": (
                    "You are a database schema design specialist. Your job is to review schema "
                    "definitions for normalization issues (violations of 1NF through BCNF), "
                    "missing NOT NULL constraints on columns that should never be null, "
                    "inappropriate data types (VARCHAR(255) for everything, INT for boolean, "
                    "TEXT for enum-like columns), missing foreign key constraints allowing "
                    "orphaned records, missing indexes on foreign key columns (critical for "
                    "join performance and CASCADE operations), and partitioning opportunities "
                    "for large tables. Evaluate migration safety: online schema change "
                    "requirements, lock duration estimation, backward compatibility of pending "
                    "changes, and data migration correctness."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "database_lead",
                "role": "Database Review Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]
