"""Tests for anti-hallucination rules in analysis skill prompts.

These tests ensure that prompts for analysis-related skills contain
explicit anti-hallucination/anti-fabrication rules so the model does
NOT invent data when analyzing results for anomalies or diagnostics.

Regression protection — if someone removes these constraints, tests fail.
"""

from __future__ import annotations


class TestAnomalyAntiHallucination:
    """Validate anti-hallucination rules in the Anomaly Detection skill."""

    def test_system_instruction_has_anti_hallucination_rules(self) -> None:
        """SYSTEM_INSTRUCTION must contain anti-hallucination rules."""
        from vaig.skills.anomaly.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "fabricate" in SYSTEM_INSTRUCTION
        assert "No placeholder names" in SYSTEM_INSTRUCTION
        assert "Insufficient data" in SYSTEM_INSTRUCTION

    def test_system_instruction_requires_evidence(self) -> None:
        """SYSTEM_INSTRUCTION must require evidence for every claim."""
        from vaig.skills.anomaly.prompts import SYSTEM_INSTRUCTION

        assert "evidence" in SYSTEM_INSTRUCTION.lower()
        assert "OBSERVED" in SYSTEM_INSTRUCTION
        assert "INFERRED" in SYSTEM_INSTRUCTION

    def test_system_instruction_forbids_extrapolation(self) -> None:
        """SYSTEM_INSTRUCTION must forbid data extrapolation."""
        from vaig.skills.anomaly.prompts import SYSTEM_INSTRUCTION

        assert "NEVER extrapolate" in SYSTEM_INSTRUCTION

    def test_analyze_phase_has_anti_fabrication_rules(self) -> None:
        """Analyze phase prompt must prevent data fabrication."""
        from vaig.skills.anomaly.prompts import PHASE_PROMPTS

        analyze = PHASE_PROMPTS["analyze"]
        assert "NEVER invent" in analyze
        assert "OBSERVED" in analyze
        assert "INFERRED" in analyze

    def test_execute_phase_has_anti_fabrication_rules(self) -> None:
        """Execute phase prompt must prevent fabricated evidence."""
        from vaig.skills.anomaly.prompts import PHASE_PROMPTS

        execute = PHASE_PROMPTS["execute"]
        assert "NEVER fabricate" in execute
        assert "evidence" in execute.lower()

    def test_report_phase_has_anti_fabrication_rules(self) -> None:
        """Report phase prompt must prevent invented data in reports."""
        from vaig.skills.anomaly.prompts import PHASE_PROMPTS

        report = PHASE_PROMPTS["report"]
        assert "NEVER invent" in report
        assert "Insufficient data" in report
        assert "NEVER fabricate" in report

    def test_pattern_analyzer_agent_has_anti_fabrication(self) -> None:
        """Pattern analyzer sub-agent must have anti-fabrication rules."""
        from vaig.skills.anomaly.skill import AnomalySkill

        skill = AnomalySkill()
        agents = skill.get_agents_config()
        pattern_agent = next(a for a in agents if a["name"] == "pattern_analyzer")
        instruction = pattern_agent["system_instruction"]
        assert "NEVER invent" in instruction
        assert "fabricate" in instruction


class TestLogAnalysisAntiHallucination:
    """Validate anti-hallucination rules in the Log Analysis skill."""

    def test_system_instruction_has_anti_hallucination_rules(self) -> None:
        """SYSTEM_INSTRUCTION must contain anti-hallucination rules."""
        from vaig.skills.log_analysis.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "fabricate" in SYSTEM_INSTRUCTION
        assert "Insufficient log data" in SYSTEM_INSTRUCTION

    def test_system_instruction_requires_evidence(self) -> None:
        """SYSTEM_INSTRUCTION must require evidence for every claim."""
        from vaig.skills.log_analysis.prompts import SYSTEM_INSTRUCTION

        assert "evidence" in SYSTEM_INSTRUCTION.lower()
        assert "cite" in SYSTEM_INSTRUCTION.lower()

    def test_system_instruction_forbids_extrapolation(self) -> None:
        """SYSTEM_INSTRUCTION must forbid trend extrapolation."""
        from vaig.skills.log_analysis.prompts import SYSTEM_INSTRUCTION

        assert "NEVER extrapolate" in SYSTEM_INSTRUCTION


class TestErrorTriageAntiHallucination:
    """Validate anti-hallucination rules in the Error Triage skill."""

    def test_system_instruction_has_anti_hallucination_rules(self) -> None:
        """SYSTEM_INSTRUCTION must contain anti-hallucination rules."""
        from vaig.skills.error_triage.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "fabricate" in SYSTEM_INSTRUCTION
        assert "Insufficient data" in SYSTEM_INSTRUCTION

    def test_system_instruction_requires_evidence(self) -> None:
        """SYSTEM_INSTRUCTION must require evidence for every finding."""
        from vaig.skills.error_triage.prompts import SYSTEM_INSTRUCTION

        assert "evidence" in SYSTEM_INSTRUCTION.lower()

    def test_system_instruction_forbids_extrapolation(self) -> None:
        """SYSTEM_INSTRUCTION must forbid impact extrapolation."""
        from vaig.skills.error_triage.prompts import SYSTEM_INSTRUCTION

        assert "NEVER extrapolate" in SYSTEM_INSTRUCTION


class TestPerfAnalysisAntiHallucination:
    """Validate anti-hallucination rules in the Performance Analysis skill."""

    def test_system_instruction_has_anti_hallucination_rules(self) -> None:
        """SYSTEM_INSTRUCTION must contain anti-hallucination rules."""
        from vaig.skills.perf_analysis.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "fabricate" in SYSTEM_INSTRUCTION
        assert "Insufficient data" in SYSTEM_INSTRUCTION

    def test_system_instruction_requires_evidence(self) -> None:
        """SYSTEM_INSTRUCTION must require evidence for every finding."""
        from vaig.skills.perf_analysis.prompts import SYSTEM_INSTRUCTION

        assert "evidence" in SYSTEM_INSTRUCTION.lower()

    def test_system_instruction_forbids_extrapolation(self) -> None:
        """SYSTEM_INSTRUCTION must forbid trend extrapolation."""
        from vaig.skills.perf_analysis.prompts import SYSTEM_INSTRUCTION

        assert "NEVER extrapolate" in SYSTEM_INSTRUCTION

    def test_system_instruction_measured_vs_estimated(self) -> None:
        """SYSTEM_INSTRUCTION must distinguish MEASURED from ESTIMATED."""
        from vaig.skills.perf_analysis.prompts import SYSTEM_INSTRUCTION

        assert "MEASURED" in SYSTEM_INSTRUCTION
        assert "ESTIMATED" in SYSTEM_INSTRUCTION


class TestAlertTuningAntiHallucination:
    """Validate anti-hallucination rules in the Alert Tuning skill."""

    def test_system_instruction_has_anti_hallucination_rules(self) -> None:
        """SYSTEM_INSTRUCTION must contain anti-hallucination rules."""
        from vaig.skills.alert_tuning.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "fabricate" in SYSTEM_INSTRUCTION
        assert "Insufficient data" in SYSTEM_INSTRUCTION

    def test_system_instruction_requires_evidence(self) -> None:
        """SYSTEM_INSTRUCTION must require evidence for every finding."""
        from vaig.skills.alert_tuning.prompts import SYSTEM_INSTRUCTION

        assert "evidence" in SYSTEM_INSTRUCTION.lower()

    def test_system_instruction_forbids_extrapolation(self) -> None:
        """SYSTEM_INSTRUCTION must forbid trend extrapolation."""
        from vaig.skills.alert_tuning.prompts import SYSTEM_INSTRUCTION

        assert "NEVER extrapolate" in SYSTEM_INSTRUCTION

    def test_system_instruction_data_not_available(self) -> None:
        """SYSTEM_INSTRUCTION must handle missing metrics gracefully."""
        from vaig.skills.alert_tuning.prompts import SYSTEM_INSTRUCTION

        assert "Data not available" in SYSTEM_INSTRUCTION
