"""Tests for all 28 built-in skills: RCA, Anomaly, Migration, Log Analysis, Error Triage, Config Audit, SLO Review, Postmortem, Code Review, IaC Review, Cost Analysis, Capacity Planning, Test Generation, Compliance Check, API Design, Runbook Generator, Dependency Audit, Database Review, Pipeline Review, Performance Analysis, Threat Model, Change Risk, Alert Tuning, Resilience Review, Incident Comms, Toil Analysis, Network Review, and ADR Generator."""

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


class TestCodeReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.code_review.skill import CodeReviewSkill

        skill = CodeReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "code-review"
        assert meta.display_name == "Code Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "code-review" in meta.tags
        assert "quality" in meta.tags
        assert "security" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.code_review.skill import CodeReviewSkill

        skill = CodeReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.code_review.skill import CodeReviewSkill

        skill = CodeReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="def process_payment(card_number, amount):",
                user_input="Review this payment processing code",
            )
            assert "def process_payment(card_number, amount):" in prompt
            assert "Review this payment processing code" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.code_review.skill import CodeReviewSkill

        skill = CodeReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "code_reviewer" in names
        assert "security_auditor" in names
        assert "review_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestIacReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.iac_review.skill import IacReviewSkill

        skill = IacReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "iac-review"
        assert meta.display_name == "IaC Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "iac" in meta.tags
        assert "terraform" in meta.tags
        assert "security" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.iac_review.skill import IacReviewSkill

        skill = IacReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.iac_review.skill import IacReviewSkill

        skill = IacReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context='resource "aws_s3_bucket" "data" { acl = "public-read" }',
                user_input="Review this Terraform config for security issues",
            )
            assert "aws_s3_bucket" in prompt
            assert "Review this Terraform config for security issues" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.iac_review.skill import IacReviewSkill

        skill = IacReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "plan_analyzer" in names
        assert "drift_detector" in names
        assert "iac_reviewer" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestCostAnalysisSkill:
    def test_metadata(self) -> None:
        from vaig.skills.cost_analysis.skill import CostAnalysisSkill

        skill = CostAnalysisSkill()
        meta = skill.get_metadata()
        assert meta.name == "cost-analysis"
        assert meta.display_name == "Cost Analysis"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "cost" in meta.tags
        assert "finops" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.cost_analysis.skill import CostAnalysisSkill

        skill = CostAnalysisSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.cost_analysis.skill import CostAnalysisSkill

        skill = CostAnalysisSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="Monthly AWS bill: $45,000. EC2: $28,000, RDS: $12,000",
                user_input="Analyze our cloud costs and find savings",
            )
            assert "Monthly AWS bill: $45,000" in prompt
            assert "Analyze our cloud costs and find savings" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.cost_analysis.skill import CostAnalysisSkill

        skill = CostAnalysisSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "resource_scanner" in names
        assert "cost_optimizer" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestCapacityPlanningSkill:
    def test_metadata(self) -> None:
        from vaig.skills.capacity_planning.skill import CapacityPlanningSkill

        skill = CapacityPlanningSkill()
        meta = skill.get_metadata()
        assert meta.name == "capacity-planning"
        assert meta.display_name == "Capacity Planning"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "capacity" in meta.tags
        assert "scaling" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.capacity_planning.skill import CapacityPlanningSkill

        skill = CapacityPlanningSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.capacity_planning.skill import CapacityPlanningSkill

        skill = CapacityPlanningSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="CPU: 78% avg, Memory: 85% peak, Disk: 92% used",
                user_input="Plan capacity for Black Friday traffic spike",
            )
            assert "CPU: 78% avg" in prompt
            assert "Plan capacity for Black Friday traffic spike" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.capacity_planning.skill import CapacityPlanningSkill

        skill = CapacityPlanningSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "trend_analyzer" in names
        assert "capacity_modeler" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestTestGenerationSkill:
    def test_metadata(self) -> None:
        from vaig.skills.test_generation.skill import TestGenerationSkill

        skill = TestGenerationSkill()
        meta = skill.get_metadata()
        assert meta.name == "test-generation"
        assert meta.display_name == "Test Generation"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "testing" in meta.tags
        assert "tdd" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.test_generation.skill import TestGenerationSkill

        skill = TestGenerationSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.test_generation.skill import TestGenerationSkill

        skill = TestGenerationSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="class UserService:\n    def create_user(self, email, name):",
                user_input="Generate tests for the UserService class",
            )
            assert "class UserService:" in prompt
            assert "Generate tests for the UserService class" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.test_generation.skill import TestGenerationSkill

        skill = TestGenerationSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "test_planner" in names
        assert "test_writer" in names
        assert "coverage_analyzer" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestComplianceCheckSkill:
    def test_metadata(self) -> None:
        from vaig.skills.compliance_check.skill import ComplianceCheckSkill

        skill = ComplianceCheckSkill()
        meta = skill.get_metadata()
        assert meta.name == "compliance-check"
        assert meta.display_name == "Compliance Check"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "compliance" in meta.tags
        assert "audit" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.compliance_check.skill import ComplianceCheckSkill

        skill = ComplianceCheckSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.compliance_check.skill import ComplianceCheckSkill

        skill = ComplianceCheckSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="AWS account with S3 buckets and RDS instances",
                user_input="Check SOC 2 compliance for our cloud infrastructure",
            )
            assert "AWS account with S3 buckets and RDS instances" in prompt
            assert "Check SOC 2 compliance for our cloud infrastructure" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.compliance_check.skill import ComplianceCheckSkill

        skill = ComplianceCheckSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "regulation_mapper" in names
        assert "gap_auditor" in names
        assert "compliance_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestAPIDesignSkill:
    def test_metadata(self) -> None:
        from vaig.skills.api_design.skill import APIDesignSkill

        skill = APIDesignSkill()
        meta = skill.get_metadata()
        assert meta.name == "api-design"
        assert meta.display_name == "API Design Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "api" in meta.tags
        assert "rest" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.api_design.skill import APIDesignSkill

        skill = APIDesignSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.api_design.skill import APIDesignSkill

        skill = APIDesignSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="GET /api/users/{id}\nPOST /api/users\nDELETE /api/users/{id}",
                user_input="Review this REST API for best practices",
            )
            assert "GET /api/users/{id}" in prompt
            assert "Review this REST API for best practices" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.api_design.skill import APIDesignSkill

        skill = APIDesignSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "contract_analyzer" in names
        assert "security_reviewer" in names
        assert "api_design_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestRunbookGeneratorSkill:
    def test_metadata(self) -> None:
        from vaig.skills.runbook_generator.skill import RunbookGeneratorSkill

        skill = RunbookGeneratorSkill()
        meta = skill.get_metadata()
        assert meta.name == "runbook-generator"
        assert meta.display_name == "Runbook Generator"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "runbook" in meta.tags
        assert "sre" in meta.tags

    def test_system_instruction(self) -> None:
        from vaig.skills.runbook_generator.skill import RunbookGeneratorSkill

        skill = RunbookGeneratorSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.runbook_generator.skill import RunbookGeneratorSkill

        skill = RunbookGeneratorSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="Production Kubernetes cluster with 50 nodes and PostgreSQL database",
                user_input="Create a runbook for database failover procedure",
            )
            assert "Production Kubernetes cluster with 50 nodes and PostgreSQL database" in prompt
            assert "Create a runbook for database failover procedure" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.runbook_generator.skill import RunbookGeneratorSkill

        skill = RunbookGeneratorSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "procedure_analyst" in names
        assert "step_writer" in names
        assert "runbook_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestDependencyAuditSkill:
    def test_metadata(self) -> None:
        from vaig.skills.dependency_audit.skill import DependencyAuditSkill

        skill = DependencyAuditSkill()
        meta = skill.get_metadata()
        assert meta.name == "dependency-audit"
        assert meta.display_name == "Dependency Audit"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "security" in meta.tags
        assert "supply-chain" in meta.tags
        assert "dependencies" in meta.tags
        assert "cve" in meta.tags
        assert "vulnerability" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.dependency_audit.skill import DependencyAuditSkill

        skill = DependencyAuditSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.dependency_audit.skill import DependencyAuditSkill

        skill = DependencyAuditSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="requirements.txt with outdated packages",
                user_input="Audit dependencies for vulnerabilities",
            )
            assert "requirements.txt with outdated packages" in prompt
            assert "Audit dependencies for vulnerabilities" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.dependency_audit.skill import DependencyAuditSkill

        skill = DependencyAuditSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "vulnerability_scanner" in names
        assert "license_analyst" in names
        assert "dependency_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestDbReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.db_review.skill import DbReviewSkill

        skill = DbReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "db-review"
        assert meta.display_name == "Database Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "database" in meta.tags
        assert "sql" in meta.tags
        assert "performance" in meta.tags
        assert "schema" in meta.tags
        assert "queries" in meta.tags
        assert "optimization" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.db_review.skill import DbReviewSkill

        skill = DbReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.db_review.skill import DbReviewSkill

        skill = DbReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="SELECT * FROM users WHERE email LIKE '%@example.com'",
                user_input="Review this query for performance issues",
            )
            assert "SELECT * FROM users WHERE email LIKE '%@example.com'" in prompt
            assert "Review this query for performance issues" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.db_review.skill import DbReviewSkill

        skill = DbReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "query_analyzer" in names
        assert "schema_reviewer" in names
        assert "database_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestPipelineReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.pipeline_review.skill import PipelineReviewSkill

        skill = PipelineReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "pipeline-review"
        assert meta.display_name == "Pipeline Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "cicd" in meta.tags
        assert "pipeline" in meta.tags
        assert "github-actions" in meta.tags
        assert "gitlab-ci" in meta.tags
        assert "devops" in meta.tags
        assert "deployment" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.pipeline_review.skill import PipelineReviewSkill

        skill = PipelineReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.pipeline_review.skill import PipelineReviewSkill

        skill = PipelineReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest",
                user_input="Review this GitHub Actions workflow",
            )
            assert "name: CI" in prompt
            assert "Review this GitHub Actions workflow" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.pipeline_review.skill import PipelineReviewSkill

        skill = PipelineReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "security_auditor" in names
        assert "efficiency_analyzer" in names
        assert "pipeline_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestPerfAnalysisSkill:
    def test_metadata(self) -> None:
        from vaig.skills.perf_analysis.skill import PerfAnalysisSkill

        skill = PerfAnalysisSkill()
        meta = skill.get_metadata()
        assert meta.name == "perf-analysis"
        assert meta.display_name == "Performance Analysis"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "performance" in meta.tags
        assert "latency" in meta.tags
        assert "profiling" in meta.tags
        assert "tracing" in meta.tags
        assert "optimization" in meta.tags
        assert "bottleneck" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.perf_analysis.skill import PerfAnalysisSkill

        skill = PerfAnalysisSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.perf_analysis.skill import PerfAnalysisSkill

        skill = PerfAnalysisSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="P99 latency: 2.5s, P50: 200ms, CPU: 85%",
                user_input="Analyze performance bottlenecks in the API",
            )
            assert "P99 latency: 2.5s, P50: 200ms, CPU: 85%" in prompt
            assert "Analyze performance bottlenecks in the API" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.perf_analysis.skill import PerfAnalysisSkill

        skill = PerfAnalysisSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "trace_analyzer" in names
        assert "resource_profiler" in names
        assert "performance_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestThreatModelSkill:
    def test_metadata(self) -> None:
        from vaig.skills.threat_model.skill import ThreatModelSkill

        skill = ThreatModelSkill()
        meta = skill.get_metadata()
        assert meta.name == "threat-model"
        assert meta.display_name == "Threat Modeling"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "security" in meta.tags
        assert "threat-modeling" in meta.tags
        assert "stride" in meta.tags
        assert "attack-surface" in meta.tags
        assert "risk" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.threat_model.skill import ThreatModelSkill

        skill = ThreatModelSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.threat_model.skill import ThreatModelSkill

        skill = ThreatModelSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="REST API with OAuth2, PostgreSQL, Redis cache",
                user_input="Conduct threat modeling for the payments service",
            )
            assert "REST API with OAuth2, PostgreSQL, Redis cache" in prompt
            assert "Conduct threat modeling for the payments service" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.threat_model.skill import ThreatModelSkill

        skill = ThreatModelSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "attack_surface_mapper" in names
        assert "threat_enumerator" in names
        assert "threat_model_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestChangeRiskSkill:
    def test_metadata(self) -> None:
        from vaig.skills.change_risk.skill import ChangeRiskSkill

        skill = ChangeRiskSkill()
        meta = skill.get_metadata()
        assert meta.name == "change-risk"
        assert meta.display_name == "Change Risk Assessment"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "change-management" in meta.tags
        assert "risk" in meta.tags
        assert "deployment" in meta.tags
        assert "rollback" in meta.tags
        assert "blast-radius" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.change_risk.skill import ChangeRiskSkill

        skill = ChangeRiskSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.change_risk.skill import ChangeRiskSkill

        skill = ChangeRiskSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="PR #456: database schema migration adding new columns",
                user_input="Assess risk for this database migration deployment",
            )
            assert "PR #456: database schema migration adding new columns" in prompt
            assert "Assess risk for this database migration deployment" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.change_risk.skill import ChangeRiskSkill

        skill = ChangeRiskSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "change_analyzer" in names
        assert "risk_scorer" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestAlertTuningSkill:
    def test_metadata(self) -> None:
        from vaig.skills.alert_tuning.skill import AlertTuningSkill

        skill = AlertTuningSkill()
        meta = skill.get_metadata()
        assert meta.name == "alert-tuning"
        assert meta.display_name == "Alert & Monitoring Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "observability" in meta.tags
        assert "alerting" in meta.tags
        assert "monitoring" in meta.tags
        assert "noise-reduction" in meta.tags
        assert "on-call" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.alert_tuning.skill import AlertTuningSkill

        skill = AlertTuningSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.alert_tuning.skill import AlertTuningSkill

        skill = AlertTuningSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="Alert: HighCPU fires 50 times/day, action rate 5%",
                user_input="Review our alerting rules for noise reduction",
            )
            assert "Alert: HighCPU fires 50 times/day, action rate 5%" in prompt
            assert "Review our alerting rules for noise reduction" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.alert_tuning.skill import AlertTuningSkill

        skill = AlertTuningSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "noise_analyzer" in names
        assert "coverage_assessor" in names
        assert "observability_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestResilienceReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.resilience_review.skill import ResilienceReviewSkill

        skill = ResilienceReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "resilience-review"
        assert meta.display_name == "Resilience Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "reliability" in meta.tags
        assert "chaos-engineering" in meta.tags
        assert "resilience" in meta.tags
        assert "failure-modes" in meta.tags
        assert "fault-tolerance" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.resilience_review.skill import ResilienceReviewSkill

        skill = ResilienceReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.resilience_review.skill import ResilienceReviewSkill

        skill = ResilienceReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="Microservices with circuit breakers and retry logic",
                user_input="Review resilience patterns for the order service",
            )
            assert "Microservices with circuit breakers and retry logic" in prompt
            assert "Review resilience patterns for the order service" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.resilience_review.skill import ResilienceReviewSkill

        skill = ResilienceReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "failure_mode_analyzer" in names
        assert "experiment_designer" in names
        assert "resilience_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestIncidentCommsSkill:
    def test_metadata(self) -> None:
        from vaig.skills.incident_comms.skill import IncidentCommsSkill

        skill = IncidentCommsSkill()
        meta = skill.get_metadata()
        assert meta.name == "incident-comms"
        assert meta.display_name == "Incident Communications"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "incident-response" in meta.tags
        assert "communication" in meta.tags
        assert "status-page" in meta.tags
        assert "stakeholder" in meta.tags
        assert "crisis" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.incident_comms.skill import IncidentCommsSkill

        skill = IncidentCommsSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.incident_comms.skill import IncidentCommsSkill

        skill = IncidentCommsSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="SEV1: Payment processing down for 30 minutes",
                user_input="Draft incident communications for stakeholders",
            )
            assert "SEV1: Payment processing down for 30 minutes" in prompt
            assert "Draft incident communications for stakeholders" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.incident_comms.skill import IncidentCommsSkill

        skill = IncidentCommsSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "status_writer" in names
        assert "comms_coordinator" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestToilAnalysisSkill:
    def test_metadata(self) -> None:
        from vaig.skills.toil_analysis.skill import ToilAnalysisSkill

        skill = ToilAnalysisSkill()
        meta = skill.get_metadata()
        assert meta.name == "toil-analysis"
        assert meta.display_name == "Toil Analysis"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "sre" in meta.tags
        assert "toil" in meta.tags
        assert "automation" in meta.tags
        assert "operational-efficiency" in meta.tags
        assert "on-call" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.toil_analysis.skill import ToilAnalysisSkill

        skill = ToilAnalysisSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.toil_analysis.skill import ToilAnalysisSkill

        skill = ToilAnalysisSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="On-call tickets: 120/month, 60% manual certificate rotation",
                user_input="Analyze operational toil and propose automation",
            )
            assert "On-call tickets: 120/month, 60% manual certificate rotation" in prompt
            assert "Analyze operational toil and propose automation" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.toil_analysis.skill import ToilAnalysisSkill

        skill = ToilAnalysisSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "toil_detector" in names
        assert "automation_planner" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestNetworkReviewSkill:
    def test_metadata(self) -> None:
        from vaig.skills.network_review.skill import NetworkReviewSkill

        skill = NetworkReviewSkill()
        meta = skill.get_metadata()
        assert meta.name == "network-review"
        assert meta.display_name == "Network Architecture Review"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "networking" in meta.tags
        assert "firewall" in meta.tags
        assert "dns" in meta.tags
        assert "load-balancer" in meta.tags
        assert "service-mesh" in meta.tags
        assert "security" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.network_review.skill import NetworkReviewSkill

        skill = NetworkReviewSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.network_review.skill import NetworkReviewSkill

        skill = NetworkReviewSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="VPC with public and private subnets, ALB, NAT gateway",
                user_input="Review network architecture for security issues",
            )
            assert "VPC with public and private subnets, ALB, NAT gateway" in prompt
            assert "Review network architecture for security issues" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.network_review.skill import NetworkReviewSkill

        skill = NetworkReviewSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3
        names = {a["name"] for a in agents}
        assert "security_reviewer" in names
        assert "topology_analyzer" in names
        assert "network_lead" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0


class TestAdrGeneratorSkill:
    def test_metadata(self) -> None:
        from vaig.skills.adr_generator.skill import AdrGeneratorSkill

        skill = AdrGeneratorSkill()
        meta = skill.get_metadata()
        assert meta.name == "adr-generator"
        assert meta.display_name == "ADR Generator"
        assert meta.version == "1.0.0"
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases
        assert "documentation" in meta.tags
        assert "architecture" in meta.tags
        assert "decision-record" in meta.tags
        assert "adr" in meta.tags
        assert "technical-writing" in meta.tags
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_system_instruction(self) -> None:
        from vaig.skills.adr_generator.skill import AdrGeneratorSkill

        skill = AdrGeneratorSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_phase_prompts(self) -> None:
        from vaig.skills.adr_generator.skill import AdrGeneratorSkill

        skill = AdrGeneratorSkill()
        for phase in [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE, SkillPhase.VALIDATE, SkillPhase.REPORT]:
            prompt = skill.get_phase_prompt(
                phase,
                context="Choosing between PostgreSQL and MongoDB for user data",
                user_input="Generate an ADR for database selection decision",
            )
            assert "Choosing between PostgreSQL and MongoDB for user data" in prompt
            assert "Generate an ADR for database selection decision" in prompt

    def test_agents_config(self) -> None:
        from vaig.skills.adr_generator.skill import AdrGeneratorSkill

        skill = AdrGeneratorSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "context_researcher" in names
        assert "adr_author" in names
        for agent in agents:
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0
