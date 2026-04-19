# Skills Guide

VAIG ships with 31 built-in skills organized by domain. Each skill provides a specialized system instruction, multi-agent pipeline, and phase-based workflow.

## How Skills Work

Each skill defines:
- **System instruction** — Domain expertise injected into the AI
- **Agent pipeline** — 2-3 specialist agents that run sequentially
- **Supported phases** — Workflow stages the skill can execute
- **Tags** — Categories for discovery and auto-detection

### Skill Phases

Skills follow a phase-based workflow using `SkillPhase`:

| Phase | Description |
|-------|-------------|
| `ANALYZE` | Examine the input, identify patterns, classify issues |
| `PLAN` | Create an action plan or remediation strategy |
| `EXECUTE` | Generate artifacts (code, configs, reports) |
| `VALIDATE` | Verify outputs against requirements |
| `REPORT` | Produce a final structured report |

Not all skills support all phases. Use `vaig skills info <name>` to see supported phases.

### Using Skills

```bash
# CLI
vaig ask "Review this code" -f app.py -s code-review
vaig ask "Analyze these logs" -f error.log -s log-analysis --auto-skill

# REPL
/skill rca
/phase analyze
```

## Skills by Category

### SRE & Incident Response

| Skill | Name | Description | Phases | Live |
|-------|------|-------------|--------|------|
| `rca` | Root Cause Analysis | Investigate production incidents using 5 Whys + Fishbone methodology | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT | No |
| `postmortem` | Postmortem | Generate comprehensive, blameless incident postmortems following Google SRE best practices | ANALYZE, PLAN, EXECUTE, REPORT | No |
| `error-triage` | Error Triage | Rapid error classification and prioritization during incidents | ANALYZE, REPORT | No |
| `incident-comms` | Incident Communications | Generate coordinated incident communications — status page updates, executive briefs, customer emails, and regulatory notifications | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT | No |
| `service-health` | Service Health Assessment | Live Kubernetes service health check with tool-backed data collection | ANALYZE, EXECUTE, REPORT | **Yes** |
| `runbook-generator` | Runbook Generator | Generate operational runbooks with step-by-step procedures, decision trees, rollback plans, and escalation paths | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT | No |

### Observability & Monitoring

| Skill | Name | Description | Phases |
|-------|------|-------------|--------|
| `log-analysis` | Log Analysis | Analyze production logs to identify error patterns, timing anomalies, and root cause hypotheses | ANALYZE, PLAN, REPORT |
| `anomaly` | Anomaly Detection | Detect anomalies in logs, metrics, data, and system behavior | ANALYZE, EXECUTE, REPORT |
| `alert-tuning` | Alert & Monitoring Review | Review alerting rules for noise reduction, coverage gaps, and monitoring quality using USE/RED methodology | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
| `slo-review` | SLO Review | Analyze SLO/SLI definitions, error budget consumption, and reliability targets based on Google SRE principles | ANALYZE, REPORT |

### Infrastructure & Operations

| Skill | Name | Description | Phases |
|-------|------|-------------|--------|
| `iac-review` | IaC Review | Review Infrastructure-as-Code for security misconfigurations, cost optimization, reliability gaps, and IaC best practices | ANALYZE, REPORT |
| `config-audit` | Config Audit | Audit infrastructure and application configs for security issues, misconfigurations, and reliability risks | ANALYZE, REPORT |
| `capacity-planning` | Capacity Planning | Forecast resource capacity needs, identify scaling bottlenecks, and recommend infrastructure scaling strategies | ANALYZE, PLAN, EXECUTE, REPORT |
| `network-review` | Network Architecture Review | Review network architecture for security vulnerabilities, topology weaknesses, DNS misconfigurations, and service mesh policy issues | ANALYZE, PLAN, REPORT |
| `cost-analysis` | Cost Analysis | Analyze cloud infrastructure costs, identify optimization opportunities, and provide FinOps recommendations | ANALYZE, PLAN, REPORT |
| `resilience-review` | Resilience Review | Analyze system resilience by enumerating failure modes, assessing mitigations, and designing chaos experiments to validate claims | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
| `toil-analysis` | Toil Analysis | Analyze operational work to identify toil, quantify its cost, and prioritize automation opportunities by ROI | ANALYZE, PLAN, EXECUTE, REPORT |

### Code Quality & Development

| Skill | Name | Description | Phases |
|-------|------|-------------|--------|
| `code-review` | Code Review | Automated code review analyzing architecture patterns, security vulnerabilities, performance issues, and best practices | ANALYZE, REPORT |
| `test-generation` | Test Generation | Generate comprehensive test suites from source code with unit, integration, and edge case coverage | ANALYZE, PLAN, EXECUTE, REPORT |
| `api-design` | API Design Review | Review API design for REST/GraphQL/gRPC best practices, consistency, security, and developer experience | ANALYZE, PLAN, REPORT |
| `db-review` | Database Review | Review database schemas, queries, and execution plans for performance issues, design problems, and operational risks | ANALYZE, PLAN, REPORT |
| `migration` | ETL Migration | Migrate ETL pipelines between platforms (Pentaho KTR/KJB to AWS Glue PySpark, etc.) | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
| `code-migration` | Code Migration | Migrate source code between programming languages (e.g., Python → Go) using a 6-phase state machine with YAML-driven idiom and dependency mappings; LLM fallback fills gaps automatically. **Note:** For ETL pipeline migration (Pentaho → AWS Glue), use the `migration` skill instead. | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
| `greenfield` | Greenfield Project | Scaffold new projects from scratch using a 6-stage pipeline: Requirements, Architecture Decision, Project Spec, Scaffold, Implement, and Verify | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
| `perf-analysis` | Performance Analysis | Analyze distributed traces, CPU/memory profiles, and performance metrics to identify bottlenecks and optimization opportunities | ANALYZE, PLAN, REPORT |

### Security & Compliance

| Skill | Name | Description | Phases |
|-------|------|-------------|--------|
| `threat-model` | Threat Modeling | Conduct STRIDE-based threat modeling to identify attack surfaces, enumerate threats, and recommend prioritized countermeasures | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
| `compliance-check` | Compliance Check | Audit systems for regulatory compliance (SOC 2, ISO 27001, HIPAA, PCI-DSS, GDPR) and generate remediation plans | ANALYZE, PLAN, EXECUTE, REPORT |
| `dependency-audit` | Dependency Audit | Audit software dependencies for known vulnerabilities, license compliance issues, supply-chain risks, and dependency hygiene | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |

### DevOps & CI/CD

| Skill | Name | Description | Phases |
|-------|------|-------------|--------|
| `pipeline-review` | Pipeline Review | Review CI/CD pipeline configurations for security risks, build efficiency, deployment safety, and pipeline-as-code hygiene | ANALYZE, PLAN, REPORT |
| `change-risk` | Change Risk Assessment | Assess deployment risk by analyzing change scope, blast radius, reversibility, and generating pre-deployment checklists and CAB summaries | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |

### Documentation & Governance

| Skill | Name | Description | Phases |
|-------|------|-------------|--------|
| `adr-generator` | ADR Generator | Generate architecture decision records (ADRs) from context, conversations, and requirements using MADR format | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |

## Structured Output (JSON Schema)

Some skills use Gemini's `response_schema` parameter to constrain the model's output to a strict JSON structure defined by a Pydantic v2 model. Instead of producing free-form Markdown, the model returns validated JSON that is then converted to a display format by the skill's `post_process_report()` method.

### How It Works

1. **Skill defines a Pydantic schema** — A `BaseModel` subclass describes every field, type, and constraint the report must contain.
2. **Agent config includes the schema** — The reporter agent's config sets `response_schema` (the model class) and `response_mime_type` (`"application/json"`).
3. **Gemini returns schema-constrained JSON** — The model produces output guided by the schema. The skill then validates it client-side.
4. **Post-processing converts to Markdown** — `post_process_report()` calls `Model.model_validate_json()` and renders the validated object via a `to_markdown()` method.

### Which Skills Use It

| Skill | Schema | Purpose |
|-------|--------|---------|
| `service-health` | `HealthReport` | Structured health reports with findings, recommendations, and timeline |

### HealthReport Schema Overview

The `HealthReport` root model (`src/vaig/skills/service_health/schema.py`) contains these top-level sections:

| Field | Type | Description |
|-------|------|-------------|
| `executive_summary` | `ExecutiveSummary` | Overall status, scope, issue counts |
| `cluster_overview` | `list[ClusterMetric]` | Key cluster metrics table |
| `service_statuses` | `list[ServiceStatus]` | Per-service health with pod/CPU/memory data |
| `findings` | `list[Finding]` | Issues grouped by severity (CRITICAL to INFO) |
| `root_cause_hypotheses` | `list[RootCauseHypothesis]` | Causal mechanism and evidence per finding |
| `causal_graph` | `CausalGraph` | Structured cause-effect relationships for evidence tracing (v0.17+) |
| `recommendations` | `list[RecommendedAction]` | Prioritized remediation with urgency and effort |
| `timeline` | `list[TimelineEvent]` | Chronological event sequence |
| `metadata` | `ReportMetadata` | Generation timestamp, cluster, model used |

### Benefits

- **Deterministic structure** — Every report has the same sections in the same order, regardless of model temperature or prompt variation.
- **Type safety** — Pydantic validation catches malformed output before it reaches the user.
- **Machine-parseable** — Downstream tools can consume the raw JSON (via `report.to_dict()`) for dashboards, alerting, or further automation.
- **Graceful degradation** — If JSON parsing fails, `post_process_report()` falls back to raw content with a warning log.

## Auto-Skill Detection

VAIG can automatically select the best skill for your query:

```bash
# The --auto-skill flag matches your query to skill tags
vaig ask "Why are my pods crashing?" -f logs.txt --auto-skill
# → Automatically selects: error-triage or rca

vaig ask "Is this Terraform safe?" -f main.tf --auto-skill
# → Automatically selects: iac-review
```

The auto-detection uses the `suggest_skill()` method in `SkillRegistry`, which matches query keywords against skill tags and descriptions.

## Composite Skills

You can combine multiple skills for comprehensive analysis:

```python
# In code — the CompositeSkill merges system instructions,
# agent pipelines, tags, and phases from multiple skills
from vaig.skills.base import CompositeSkill

composite = CompositeSkill(
    name="full-audit",
    skills=[code_review_skill, threat_model_skill, compliance_skill],
)
```

## Custom Skills

### YAML-Based Skill Definitions (v0.16+, Skill Plugin System)

The easiest way to create a skill is with a YAML definition file — no Python required:

```yaml
# ~/.vaig/skills/my-skill/skill.yaml
name: my-skill
display_name: My Custom Skill
description: Does something cool
version: 1.0.0
tags: [custom, analysis]
phases: [ANALYZE, REPORT]

system_instruction: |
  You are a specialized assistant for analyzing X.
  Always provide structured output with findings and recommendations.

agents:
  - role: analyzer
    model: gemini-2.5-flash
    instruction: Analyze the input for issues.
  - role: reporter
    model: gemini-2.5-pro
    instruction: Produce a structured report based on the analysis.

phase_prompts:
  ANALYZE: "Analyze the following input:\n\n{context}\n\nUser request: {user_input}"
  REPORT: "Generate a final structured report from the analysis."
```

### Python-Based Skill Definitions

For advanced control, subclass `BaseSkill` directly:

```bash
vaig skills create my-skill -d "My custom analysis" -t "custom,analysis"
```

This generates a skill directory with:
- `skill.py` — Skill class with metadata, system instruction, phases, and agent config
- `prompts.py` — Phase-specific prompt templates
- `__init__.py` — Package init

```python
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase, SkillResult


class MySkill(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="my-skill",
            display_name="My Custom Skill",
            description="Does something cool",
            tags=["custom"],
        )

    def get_system_instruction(self) -> str:
        return "You are a specialized assistant for..."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Context:\n{context}\n\nTask: {user_input}"
```

### Installing Custom Skills

Place skills in one or more directories and configure:

```yaml
# ~/.vaig/config.yaml
skills:
  external_dirs:           # Preferred (v0.16+) — list of directories
    - ~/.vaig/skills
    - ./team-skills
  # custom_dir: ...        # Deprecated — use external_dirs instead
```

> **Note:** Custom skills are auto-discovered at startup. YAML-based skills can be hot-reloaded without restarting VAIG.

## Skill Details

Use the CLI to inspect any skill:

```bash
$ vaig skills info rca

Name:           rca
Display Name:   Root Cause Analysis
Description:    Investigate production incidents using 5 Whys + Fishbone methodology
Version:        1.0.0
Tags:           incident, sre, debugging, post-mortem, logs, metrics
Phases:         ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT
Model:          gemini-2.5-pro
Live Tools:     No

Agents:
  1. evidence_collector — Evidence Collector
  2. hypothesis_builder — Hypothesis Builder
  3. rca_synthesizer  — RCA Synthesizer
```

---

[Back to index](README.md)
