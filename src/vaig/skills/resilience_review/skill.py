"""Resilience Review Skill — failure mode analysis and chaos engineering planning."""

from __future__ import annotations

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.resilience_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ResilienceReviewSkill(BaseSkill):
    """Resilience Review skill for failure mode analysis and chaos engineering planning.

    Supports multi-agent execution with specialized agents:
    - Failure Mode Analyzer: Enumerates failure modes and assesses mitigations
    - Experiment Designer: Designs chaos experiments for unvalidated claims
    - Resilience Lead: Synthesizes scorecard with gaps, experiments, and gameday plan
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="resilience-review",
            display_name="Resilience Review",
            description=(
                "Analyze system resilience by enumerating failure modes, assessing "
                "mitigations, and designing chaos experiments to validate claims"
            ),
            version="1.0.0",
            tags=[
                "reliability",
                "chaos-engineering",
                "resilience",
                "failure-modes",
                "fault-tolerance",
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

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "failure_mode_analyzer",
                "role": "Failure Mode Analyzer",
                "system_instruction": (
                    "You are a failure mode analysis specialist. Your job is to systematically "
                    "enumerate failure modes for each component in the system — process "
                    "crash/hang, memory leak, thread deadlock, GC pressure, network partition, "
                    "latency injection, packet loss, DNS failure, dependency unavailability, "
                    "slow dependency response, resource exhaustion (CPU, memory, disk, file "
                    "descriptors, connection pools, thread pools), data corruption, replication "
                    "lag, configuration drift, secret expiration, and certificate expiration. "
                    "For each failure mode, assess existing mitigations: circuit breakers "
                    "(thresholds, half-open behavior), retries (backoff strategy, idempotency), "
                    "timeouts (connect vs read, cascade safety), fallbacks (degraded experience "
                    "quality), bulkheads (resource isolation), health checks (shallow vs deep), "
                    "redundancy (replicas, multi-zone), and auto-scaling responsiveness. "
                    "Rate mitigation status as: Absent, Partial, Adequate, or Verified."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "experiment_designer",
                "role": "Chaos Experiment Designer",
                "system_instruction": (
                    "You are a chaos engineering experiment design specialist. Your job is "
                    "to design chaos experiments for each unvalidated resilience claim in the "
                    "system. For each experiment, define: a clear hypothesis stating expected "
                    "behavior under failure, the specific failure to inject (network partition, "
                    "latency injection, process kill, CPU stress, disk fill, DNS failure, "
                    "dependency error injection), steady state metrics to measure (error rate, "
                    "latency percentiles, throughput, user-facing error rate), success criteria "
                    "that confirm the hypothesis, abort conditions that trigger immediate "
                    "experiment termination (error rate exceeding threshold, latency exceeding "
                    "SLO, user-visible impact detected), blast radius containment strategy "
                    "(single instance, single zone, percentage of traffic, feature-flagged "
                    "cohort), and recommended tooling (Litmus Chaos, Gremlin, Toxiproxy, "
                    "AWS FIS, tc/iptables). Design experiments from simplest to most complex, "
                    "starting in staging and graduating to production with containment."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "resilience_lead",
                "role": "Resilience Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]
