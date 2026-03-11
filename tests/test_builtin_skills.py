"""Tests for the built-in RCA, Anomaly, and Migration skills."""

from __future__ import annotations

from vaig.skills.base import SkillPhase


class TestRCASkill:
    def test_metadata(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        meta = skill.get_metadata()
        assert meta.name == "rca"
        assert meta.display_name == "Root Cause Analysis"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert "incident" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompt(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.ANALYZE,
            context="error logs here",
            user_input="Why did the service crash?",
        )
        assert "error logs here" in prompt
        assert "Why did the service crash?" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "log_analyzer" in names
        assert "metric_correlator" in names
        assert "rca_lead" in names


class TestAnomalySkill:
    def test_metadata(self) -> None:
        from vaig.skills.anomaly.skill import AnomalySkill

        skill = AnomalySkill()
        meta = skill.get_metadata()
        assert meta.name == "anomaly"
        assert SkillPhase.ANALYZE in meta.supported_phases

    def test_agents_config(self) -> None:
        from vaig.skills.anomaly.skill import AnomalySkill

        skill = AnomalySkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "pattern_analyzer" in names
        assert "anomaly_detector" in names


class TestMigrationSkill:
    def test_metadata(self) -> None:
        from vaig.skills.migration.skill import MigrationSkill

        skill = MigrationSkill()
        meta = skill.get_metadata()
        assert meta.name == "migration"
        assert "pentaho" in [t.lower() for t in meta.tags] or len(meta.tags) > 0

    def test_agents_config(self) -> None:
        from vaig.skills.migration.skill import MigrationSkill

        skill = MigrationSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "code_analyzer" in names
        assert "code_generator" in names
        assert "migration_validator" in names
