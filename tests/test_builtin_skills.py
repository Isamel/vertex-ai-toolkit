"""Tests for the built-in RCA, Anomaly, Migration, Log Analysis, Error Triage, Config Audit, SLO Review, and Postmortem skills."""

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


class TestLogAnalysisSkill:
    def test_metadata(self) -> None:
        from vaig.skills.log_analysis.skill import LogAnalysisSkill

        skill = LogAnalysisSkill()
        meta = skill.get_metadata()
        assert meta.name == "log-analysis"
        assert meta.display_name == "Log Analysis"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "logs" in meta.tags
        assert "sre" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.log_analysis.skill import LogAnalysisSkill

        skill = LogAnalysisSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.log_analysis.skill import LogAnalysisSkill

        skill = LogAnalysisSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="sample log entries",
                user_input="Why are we seeing 500 errors?",
            )
            assert "sample log entries" in prompt
            assert "Why are we seeing 500 errors?" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.log_analysis.skill import LogAnalysisSkill

        skill = LogAnalysisSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "pattern_detector" in names
        assert "context_analyzer" in names
        assert "diagnostic_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestErrorTriageSkill:
    def test_metadata(self) -> None:
        from vaig.skills.error_triage.skill import ErrorTriageSkill

        skill = ErrorTriageSkill()
        meta = skill.get_metadata()
        assert meta.name == "error-triage"
        assert meta.display_name == "Error Triage"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "errors" in meta.tags
        assert "sre" in meta.tags
        assert "triage" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.error_triage.skill import ErrorTriageSkill

        skill = ErrorTriageSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.error_triage.skill import ErrorTriageSkill

        skill = ErrorTriageSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="500 Internal Server Error in payments service",
                user_input="Users cannot complete checkout",
            )
            assert "500 Internal Server Error in payments service" in prompt
            assert "Users cannot complete checkout" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.error_triage.skill import ErrorTriageSkill

        skill = ErrorTriageSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "error_classifier" in names
        assert "triage_coordinator" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestConfigAuditSkill:
    def test_metadata(self) -> None:
        from vaig.skills.config_audit.skill import ConfigAuditSkill

        skill = ConfigAuditSkill()
        meta = skill.get_metadata()
        assert meta.name == "config-audit"
        assert meta.display_name == "Config Audit"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "config" in meta.tags
        assert "sre" in meta.tags
        assert "security" in meta.tags
        assert "audit" in meta.tags
        assert "compliance" in meta.tags
        assert "infrastructure" in meta.tags
        assert "reliability" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.config_audit.skill import ConfigAuditSkill

        skill = ConfigAuditSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.config_audit.skill import ConfigAuditSkill

        skill = ConfigAuditSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="apiVersion: apps/v1\nkind: Deployment",
                user_input="Check this Kubernetes deployment for security issues",
            )
            assert "apiVersion: apps/v1" in prompt
            assert "Check this Kubernetes deployment for security issues" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.config_audit.skill import ConfigAuditSkill

        skill = ConfigAuditSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "security_scanner" in names
        assert "reliability_auditor" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestSloReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.slo_review.skill import SloReviewSkill

        skill = SloReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "slo-review"
        assert meta.display_name == "SLO Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "slo" in meta.tags
        assert "sli" in meta.tags
        assert "sre" in meta.tags
        assert "reliability" in meta.tags
        assert "error-budget" in meta.tags
        assert "observability" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.slo_review.skill import SloReviewSkill

        skill = SloReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.slo_review.skill import SloReviewSkill

        skill = SloReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="SLO: 99.9% availability, current: 99.7%",
                user_input="Review our SLOs for the payments service",
            )
            assert "SLO: 99.9% availability, current: 99.7%" in prompt
            assert "Review our SLOs for the payments service" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.slo_review.skill import SloReviewSkill

        skill = SloReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "sli_analyzer" in names
        assert "budget_strategist" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestPostmortemSkill:
    def test_metadata(self) -> None:
        from vaig.skills.postmortem.skill import PostmortemSkill

        skill = PostmortemSkill()
        meta = skill.get_metadata()
        assert meta.name == "postmortem"
        assert meta.display_name == "Postmortem"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "postmortem" in meta.tags
        assert "sre" in meta.tags
        assert "blameless" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.postmortem.skill import PostmortemSkill

        skill = PostmortemSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.postmortem.skill import PostmortemSkill

        skill = PostmortemSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="database failover at 03:00 UTC caused 45min outage",
                user_input="Payment service was down for 45 minutes",
            )
            assert "database failover at 03:00 UTC caused 45min outage" in prompt
            assert "Payment service was down for 45 minutes" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.postmortem.skill import PostmortemSkill

        skill = PostmortemSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "timeline_builder" in names
        assert "impact_assessor" in names
        assert "postmortem_author" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0
