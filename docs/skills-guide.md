# Skills Guide

VAIG ships with 29 built-in skills organized by domain. Each skill provides a specialized system instruction, multi-agent pipeline, and phase-based workflow.

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
| `migration` | Code Migration | Migrate code and ETL pipelines between platforms (Pentaho to AWS Glue, etc.) | ANALYZE, PLAN, EXECUTE, VALIDATE, REPORT |
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

Create your own skills using the scaffold command:

```bash
vaig skills create my-skill -d "My custom analysis" -t "custom,analysis"
```

This generates a skill directory with:
- `skill.py` — Skill class with metadata, system instruction, phases, and agent config
- `prompts.py` — Phase-specific prompt templates
- `__init__.py` — Package init

Place custom skills in `~/.vaig/skills/` or configure a custom directory:

```yaml
# vaig.yaml
skills:
  custom_dir: ./my-skills
```

> **Note:** Custom skills are auto-discovered at startup. They must follow the same `BaseSkill` interface as built-in skills.

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
