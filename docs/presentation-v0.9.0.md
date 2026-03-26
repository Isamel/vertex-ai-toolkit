---
marp: true
theme: default
paginate: true
header: "Vertex AI Toolkit (vaig) — v0.9.0"
footer: "Confidential — Internal Use Only"
---

<!-- _class: lead -->

# Vertex AI Toolkit
## vaig — v0.9.0

**Multi-agent AI operations platform for GKE and GCP**

*March 26, 2026*

<!-- Speaker notes:
Welcome. This deck covers two things: what the team delivered in the v0.9.0 cycle, and why vaig should be adopted across the organization. The core thesis is simple — GCP operations are expensive and manual; vaig automates them with AI. Let the numbers speak.
-->

---

## The Problem

GKE and GCP operations today are:

- **Time-intensive** — diagnosing a service incident requires navigating kubectl, Cloud Logging, Cloud Monitoring, and Helm simultaneously
- **Error-prone** — manual cost analysis via spreadsheet misses container-level waste and idle namespaces
- **Expertise-gated** — effective cluster troubleshooting requires deep GKE, Istio, and ArgoCD knowledge
- **Slow to scale** — onboarding engineers to GCP operations takes weeks, not days
- **Inconsistent** — every engineer follows a different diagnostic runbook, yielding inconsistent outputs

<!-- Speaker notes:
This isn't theoretical. Every time an on-call engineer gets paged at 2am, they're doing this manually. Every sprint planning involving cost review requires a senior engineer spending half a day in spreadsheets. vaig addresses all of this.
-->

---

## The Solution

**vaig** turns natural language into GCP operations.

```
vaig ask "Why is the payment-service degraded in prod?"
vaig ask "Show me namespace cost waste for the last 30 days" --skill cost-analysis
vaig ask "Migrate this Python module to Go" --code --pipeline
vaig ask "Scaffold a new microservice for order processing" --code
```

- Single CLI entry point for all GKE, Vertex AI, and GCP workflows
- Extensible skill system — 31 specialized multi-agent workflows built in
- Pluggable architecture — add custom tools or MCP servers without code changes
- Runs locally or in CI/CD pipelines

<!-- Speaker notes:
The interface is deliberately simple. Any engineer who can write a sentence can use vaig. The complexity lives in the agents and tools underneath.
-->

---

<!-- _class: lead -->

# What vaig Does

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                        CLI (Typer)                        │
│              ask  |  chat  |  live  |  code               │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                  Skill Orchestrator                        │
│      Routes request → selects skill → initializes        │
│      agents → merges results → formats output             │
└──────┬──────────────┬───────────────┬────────────────────┘
       │              │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
│ InfraAgent  │ │ CodingAgent │ │ SpecialistAgent │
│ GKE diag.   │ │ Code tasks  │ │ Skill-specific  │
└──────┬──────┘ └──────┬──────┘ └─────┬───────────┘
       │               │              │
┌──────▼───────────────▼──────────────▼───────────┐
│                  Tool Registry                    │
│  kubectl · gcloud · monitoring · file · shell    │
│  helm · argocd · argo rollouts · datadog · mcp   │
└──────────────────────────────────────────────────┘
```

<!-- Speaker notes:
Three layers: CLI, orchestration, execution. The skill orchestrator is the key — it routes intent to the right combination of agents and tools, then synthesizes results into a structured report.
-->

---

## Skills Ecosystem — 31 Specialized Workflows

**Infrastructure & Reliability**
`service-health` · `rca` · `anomaly` · `postmortem` · `slo-review` · `resilience-review` · `alert-tuning` · `toil-analysis`

**Cost & Capacity**
`cost-analysis` · `capacity-planning` · `change-risk`

**Code & Architecture**
`code-migration` · `greenfield` · `code-review` · `iac-review` · `api-design` · `adr-generator` · `test-generation`

**Security & Compliance**
`compliance-check` · `threat-model` · `dependency-audit`

**Operations**
`log-analysis` · `error-triage` · `config-audit` · `runbook-generator` · `incident-comms` · `network-review` · `pipeline-review` · `db-review` · `perf-analysis` · `migration`

<!-- Speaker notes:
31 skills ship out of the box. Each skill is a multi-agent workflow — not a single prompt. They enforce structured reasoning, anti-hallucination rules, and output validation.
-->

---

## GCP Services Supported

| Domain | Services |
|--------|----------|
| **Compute** | GKE Standard, GKE Autopilot (32 regions) |
| **Observability** | Cloud Monitoring, Cloud Logging |
| **Deployment** | ArgoCD, Argo Rollouts, Helm |
| **Mesh** | Anthos Service Mesh (Istio) |
| **Platform** | gcloud CLI, kubectl, Cloud Container API |
| **Extensions** | Datadog, MCP servers, custom Python plugins |

**Tool surface:** 28 tool modules — diagnostics, discovery, scaling, security, mutations, cost estimation, mesh introspection

<!-- Speaker notes:
vaig is not a toy. It connects to the real GCP control plane via the same APIs your SREs use manually. The difference is that the process is AI-driven, structured, and repeatable.
-->

---

<!-- _class: lead -->

# v0.9.0 — What We Built

### 8 PRs · 4 capability blocks · 10-day delivery cycle

---

## Initiative Overview — v0.9.0

| PR | Block | Capability |
|----|-------|-----------|
| #105 | Foundation | Refactor monolithic prompts into 9-file package |
| #106 | Cost | GKE Autopilot workload cost estimation (basic) |
| #107 | Cost | 32-region Autopilot pricing table |
| #108 | Cost | Cost estimation v2 — usage metrics + namespace summaries |
| #109 | Coding | CoT enforcement, text-delimited prompt defense, SPEC phase, `verify_completeness` |
| #110 | Coding | `CodeMigrationSkill` — 6-phase language migration state machine |
| #111 | Coding | `CodingSkillOrchestrator` (3-agent pipeline) + `GreenfieldSkill` |
| #112 | Docs | Full documentation sync covering all v0.9.0 changes |

**26 commits** merged. All changes covered by the existing test suite (5,938 tests).

<!-- Speaker notes:
Four distinct blocks of work, each building on the previous. Cost estimation, coding skill evolution, quality hardening, and documentation. Shipped in 10 days.
-->

---

## Cost Estimation v2

**Problem:** Teams had no visibility into Autopilot workload cost at the container level. Namespace-level waste was invisible.

**What was built:**
- Per-container CPU and memory cost breakdown for every GKE Autopilot workload
- Namespace-level cost summaries with waste detection and efficiency metrics
- Cloud Monitoring API integration — actual usage vs. requested resources
- 32 GCP regions covered in the Autopilot pricing table

**Before:** Manual `kubectl` + spreadsheet — hours per review cycle
**After:** `vaig ask "Show cost waste in the payments namespace"` — seconds

<!-- Speaker notes:
This is the most immediate ROI story. If your organization runs Autopilot clusters, you are almost certainly paying for idle or oversized containers. vaig surfaces that automatically.
-->

---

## Coding Skill Evolution — 3 Phases

**Phase 1 — Quality Hardening (#109)**
- Chain-of-thought (CoT) enforcement in coding agent system prompts
- Text-delimited boundaries (`DELIMITER_DATA_START`/`DELIMITER_DATA_END`) for prompt injection defense
- SPEC phase — added as a preliminary specification step before implementation, within Phase 1
- `verify_completeness` tool — scans output for TODO, FIXME, `pass`, `...`, `NotImplementedError`

**Phase 2 — CodeMigrationSkill (#110)**
- 6-phase state machine: Inventory → Semantic Map → Spec → Implement → Verify → Report
- YAML-driven idiom mappings: 12 Python→Go idiom translations + 17 dependency mappings
- Distinct from ETL migration — purpose-built for source code language conversion

**Phase 3 — 3-Agent Pipeline (#111)**
- `CodingSkillOrchestrator`: Planner → Implementer → Verifier
- Activate with `vaig ask --code --pipeline "..."`

<!-- Speaker notes:
This is the engineering story. Three successive PRs transformed the coding workflow from a single-agent prompt into a hardened, multi-agent pipeline with anti-hallucination guards and structured verification.
-->

---

## GreenfieldSkill — New Project Scaffolding

**Problem:** Starting a new service from scratch requires architectural decisions, project structure, boilerplate, and working code — all consistent with team standards.

**What was built:**
- 6-stage pipeline: Requirements → Architecture Decision → Project Spec → Scaffold → Implement → Verify
- Driven entirely from a natural language description
- Produces a working, runnable project skeleton with tests and configuration

**Example:**
```bash
vaig ask --code "Create a new Go microservice for order processing \
  with REST API, PostgreSQL, and Kubernetes manifests"
```

The output is a structured project directory — not a skeleton with TODO placeholders.

<!-- Speaker notes:
GreenfieldSkill is the developer productivity multiplier. New projects that used to take a day to scaffold correctly now take minutes. Standards and structure are baked into the pipeline.
-->

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Test functions | **5,938** |
| Test files | **135** |
| Source files (`src/vaig/`) | **207** |
| Lines of code | **~60,200** |
| Total commits | **424** |
| v0.9.0 commits | **26** |
| Skills | **31** |
| Agent modules | **7** |
| Tool modules | **28** |

CI pipeline enforces: ruff (lint), mypy (strict type checking), pytest (all tests)

<!-- Speaker notes:
Nearly 6,000 tests. Strict mypy type checking. Every PR is blocked by CI. The codebase is not a prototype — it is production-grade Python with the discipline of a typed, tested, linted codebase.
-->

---

<!-- _class: lead -->

# Impact & ROI

---

## Time Savings — Operations

**GKE Service Incident Diagnosis**
- Before: Navigate kubectl, Cloud Logging, Monitoring, and Helm manually — 30–60 min per incident
- After: `vaig ask "Why is {service} degraded?"` — structured report in under 2 minutes
- The service-health skill runs a 4-agent two-pass investigation pipeline automatically

**Namespace Cost Review**
- Before: Export kubectl resource data → Cloud Monitoring queries → spreadsheet analysis — half-day effort
- After: `vaig ask --skill cost-analysis "Show namespace waste"` — seconds
- Per-container breakdown, usage vs. request delta, efficiency score — all automated

**Operational overhead reduction estimate:** 80%+ on recurring cost review tasks

<!-- Speaker notes:
These are conservative estimates. The real number depends on how frequently teams run these analyses. Teams that do weekly cost reviews will see the biggest gains.
-->

---

## Developer Productivity

**Code Migration (`CodeMigrationSkill`)**
- Structured 6-phase migration from Python to Go (or other target languages)
- YAML-driven idiom and dependency mapping — not ad-hoc prompt engineering
- Produces verifiable, runnable output — not boilerplate with gaps

**Greenfield Project Scaffolding (`GreenfieldSkill`)**
- Architecture decision → spec → scaffold → working code in one command
- Enforces standards from the first line of code

**Code Review and Test Generation**
- `code-review` skill: structured review with findings, severity, and remediation
- `test-generation` skill: generates test suites from source code

**`verify_completeness`** — automated detection of incomplete implementations before delivery

<!-- Speaker notes:
The coding skills are not replacing engineers — they are removing the mechanical, low-creativity tasks that consume engineering time without producing value. Engineers should be solving problems, not writing boilerplate.
-->

---

## Risk Reduction

**Prompt Injection Defense**
- `wrap_untrusted_content()` wraps all external data in text-delimited boundaries (`DELIMITER_DATA_START`/`DELIMITER_DATA_END`)
- Prevents untrusted content (logs, code, user input) from hijacking agent instructions

**Anti-Hallucination**
- CoT enforcement in system prompts requires agents to reason step-by-step before concluding
- Anti-hallucination rules block fabricated tool calls and invented resource names
- `verify_completeness` prevents shipping code with unimplemented stubs

**Type Safety and CI Enforcement**
- 137 mypy strict type errors resolved; type checking is now a blocking CI gate
- Every merge to `main` requires passing lint, types, and all 5,938 tests

**These are not optional safeguards — they are architectural properties of the system.**

<!-- Speaker notes:
When AI agents interact with real infrastructure, hallucination is not an inconvenience — it is a risk. vaig was designed from the start with defense-in-depth: multiple layers of prompt hardening, structured output enforcement, and completeness verification.
-->

---

<!-- _class: lead -->

# Adoption Path

---

## Getting Started

**Installation**

```bash
# Core (ask, chat, code commands)
pip install vertex-ai-toolkit

# With GKE live infrastructure support
pip install "vertex-ai-toolkit[live]"
```

**First commands**

```bash
# Interactive chat
vaig chat

# Single question
vaig ask "What are the top 3 cost drivers in my cluster?"

# Use a specific skill
vaig ask "Run a cost analysis for the staging namespace" --skill cost-analysis

# Code with 3-agent pipeline
vaig ask --code --pipeline "Refactor this module to use async/await" -f main.py
```

**Requirements:** Python 3.11+, Google Cloud project, Application Default Credentials

<!-- Speaker notes:
Installation is one command. Authentication reuses existing ADC credentials — no new secrets to manage. If your engineers already have gcloud configured, they can run vaig immediately.
-->

---

## Integration Points

**CLI — Drop-in for ad-hoc operations**
```bash
vaig ask "Show pod restart counts in production" --output report.md
```

**Python SDK — Embed in automation scripts**
```python
# Skills are invoked via the CLI or the Skill Orchestrator; there is no standalone execute() API.
# Use the CLI for scripted integration:
#   vaig ask "Assess service health in namespace payments" --skill service-health
```

**CI/CD pipelines — Automated cost and security review**
```yaml
- name: Cost review
  run: vaig ask --skill cost-analysis "Flag namespaces over budget" --output cost.md
```

**MCP / Plugin extension — Add custom tools**
```yaml
# config/default.yaml
mcp_servers:
  - name: internal-tools
    command: ["python", "-m", "my_tools_server"]
```

<!-- Speaker notes:
vaig is not just a CLI tool — it is a platform. Teams can embed it in their pipelines, extend it with their own tools, and integrate it into existing automation workflows.
-->

---

## What Comes Next

The v0.9.0 cycle established the foundation. The trajectory is clear:

- **Deeper cost intelligence** — cross-namespace budget alerts, trend analysis, and anomaly detection on spend
- **Migration coverage expansion** — additional language pairs beyond Python→Go
- **Agentic CI integration** — automated PR review, test generation, and greenfield standards enforcement
- **Platform observability** — unified GKE health dashboard generated from vaig reports
- **Broader GCP surface** — Cloud Run, Cloud SQL, Pub/Sub, and Artifact Registry integration

The skill framework is designed for extension. Adding a new workflow requires implementing a single skill class — the orchestration, tooling, and output formatting are inherited.

<!-- Speaker notes:
We are not committing to a roadmap here. We are communicating architectural readiness. The platform can grow in any of these directions without rearchitecting the core.
-->

---

<!-- _class: lead -->

# Call to Action

---

## For Management — What the Team Delivered

**In 10 days (v0.8.0 → v0.9.0):**

- 8 pull requests merged across 4 capability blocks
- 26 commits, zero regression (5,938 tests passing)
- 3 new major capabilities: cost estimation v2, CodeMigrationSkill, GreenfieldSkill
- 4 quality improvements: CoT enforcement, text-delimited prompt defense, SPEC phase, verify_completeness
- Full documentation coverage synchronized with implementation

**The codebase:** 207 source files, ~60,200 lines, 31 skills, 7 agent modules, 28 tool modules

This was delivered with discipline: strict type checking, comprehensive testing, and structured prompt engineering. The work is production-ready.

<!-- Speaker notes:
This is the accountability slide. The team shipped real, tested, documented capabilities in under two weeks. Not prototypes. Not demos. Production code with CI gates.
-->

---

## For C-Suite — Adopt vaig

**The operational problem is real:**
GKE cost overruns, slow incident resolution, and high engineering overhead in GCP operations are measurable costs — in engineering hours and infrastructure spend.

**vaig is the solution, already built:**
- 31 specialized AI workflows ready to deploy today
- Cost estimation with per-container waste detection
- Incident diagnosis in seconds instead of hours
- Code migration and scaffolding at engineering scale

**The ask:**
1. Approve internal adoption as the standard GCP operations tool
2. Integrate vaig into the CI/CD pipeline for automated cost and security review
3. Socialize the coding skills to engineering teams for migration and greenfield projects

**This tool was built by your team. It runs on your infrastructure. The cost of adoption is near zero.**

<!-- Speaker notes:
This is the close. The tool exists. The tests pass. The documentation is written. The only remaining variable is organizational will to adopt it. The ROI case is strong: automation of recurring manual tasks, risk reduction, and accelerated delivery.
-->

---

<!-- _class: lead -->

# Thank You

**Vertex AI Toolkit — v0.9.0**

Repository: `vertex-ai-toolkit`
Documentation: `docs/`
Getting Started: `docs/getting-started.md`

*Questions?*

<!-- Speaker notes:
Open for questions. The full documentation is in the repository. For a live demo, we can run any of the commands shown in this deck against a real cluster.
-->
