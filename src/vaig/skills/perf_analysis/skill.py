"""Performance Analysis Skill — distributed tracing, profiling, and optimization analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.perf_analysis.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class PerfAnalysisSkill(BaseSkill):
    """Performance Analysis skill for tracing, profiling, and optimization.

    Supports multi-agent execution with specialized agents:
    - Trace Analyzer: Distributed trace parsing, critical path, fan-out detection
    - Resource Profiler: CPU profiles, memory allocations, GC, thread contention
    - Performance Lead: Correlates findings into prioritized optimization plan
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="perf-analysis",
            display_name="Performance Analysis",
            description=(
                "Analyze distributed traces, CPU/memory profiles, and performance "
                "metrics to identify bottlenecks and optimization opportunities"
            ),
            version="1.0.0",
            tags=[
                "performance",
                "latency",
                "profiling",
                "tracing",
                "optimization",
                "bottleneck",
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

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "trace_analyzer",
                "role": "Distributed Trace Analyzer",
                "system_instruction": (
                    "You are a distributed tracing specialist. Your job is to parse traces "
                    "from OpenTelemetry, Jaeger, Zipkin, or AWS X-Ray formats to reconstruct "
                    "request flows across services. Identify the critical path — the longest "
                    "sequential chain of operations that determines end-to-end latency. Detect "
                    "fan-out amplification where a single request spawns N downstream calls, "
                    "especially when N grows with data size. Quantify cross-service network "
                    "latency vs in-service processing time for each hop. Identify retries, "
                    "timeouts, and circuit breaker activations. Analyze long-tail P99 latency "
                    "contributors and tail latency amplification in fan-out architectures. "
                    "Detect coordinated omission in benchmark traces."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "resource_profiler",
                "role": "Resource Profiler",
                "system_instruction": (
                    "You are a system resource profiling specialist. Your job is to analyze "
                    "CPU profiles (flame graphs, hot function reports) to identify functions "
                    "consuming disproportionate CPU time, unexpected system call overhead, and "
                    "compiler optimization opportunities. Evaluate memory allocation patterns — "
                    "allocation rate in hot loops, GC frequency and pause times, heap growth "
                    "trends, large object allocations in request-scoped code, and memory leak "
                    "indicators. Profile thread and goroutine contention — lock hold times, "
                    "lock wait times, thread pool saturation, and deadlock risk patterns. "
                    "Assess I/O wait patterns — synchronous I/O in async contexts, excessive "
                    "disk access, network round-trip overhead, and connection pool exhaustion."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "performance_lead",
                "role": "Performance Analysis Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]
