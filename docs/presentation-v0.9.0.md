---
marp: true
theme: default
paginate: true
header: "Vertex AI Toolkit (vaig) — Post-v0.11.0"
footer: "Confidential — Internal Use Only"
---

<!-- _class: lead -->

# Vertex AI Toolkit
## vaig — Post-v0.11.0

**Multi-agent AI operations platform for GKE and GCP**

*March 29, 2026*

<!-- Speaker notes:
Welcome. This deck covers the full platform as of post-v0.11.0 — what the team has delivered across 7 releases, and why vaig should be adopted across the organization. The core thesis is simple — GCP operations are expensive and manual; vaig automates them with AI. Let the numbers speak.
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
- Extensible skill system — 33 specialized multi-agent workflows built in
- 62 tool functions across diagnostics, cost, security, and mutations
- Pluggable architecture — add custom tools or MCP servers without code changes
- 13 CLI commands: `ask`, `chat`, `live`, `discover`, `doctor`, `feedback`, `optimize`, `export` + `sessions`, `models`, `skills`, `stats`, `mcp`
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

## Skills Ecosystem — 33 Specialized Workflows

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

**New in v0.10–v0.11**
`discover` · `optimize`

<!-- Speaker notes:
33 skills ship out of the box. Each skill is a multi-agent workflow — not a single prompt. They enforce structured reasoning, anti-hallucination rules, and output validation.
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

**Tool surface:** 62 tool functions — diagnostics, discovery, scaling, security, mutations, cost estimation, mesh introspection

<!-- Speaker notes:
vaig is not a toy. It connects to the real GCP control plane via the same APIs your SREs use manually. The difference is that the process is AI-driven, structured, and repeatable.
-->

---

<!-- _class: lead -->

# v0.9.0 → v0.11.0 — What We Built

### 161 PRs · 291 commits · 7 releases (v0.1.0 through v0.11.0)

---

## Release Highlights — v0.9.0 through v0.11.0

| Release | Key Capabilities |
|---------|-----------------|
| v0.9.0 | Cost Estimation v2, CodeMigrationSkill, GreenfieldSkill, CodingSkillOrchestrator |
| v0.10.0 | `vaig discover` (autonomous cluster scanning), `vaig doctor` (environment healthcheck), `vaig feedback` (bug reports & feature requests) |
| v0.11.0 | `vaig optimize` (tool call analysis + report quality analysis with `--reports`), Watch mode diff + HTML export, RAG Engine integration, Shared Pipeline State, Adaptive Prompt Tuning, `/live` REPL command |

**291 commits** merged across **161 PRs**. All changes covered by the test suite (**6,631 tests passing**).

<!-- Speaker notes:
Seven releases from v0.1.0 through v0.11.0. The platform has grown from a prototype to a production-grade operations tool with 33 skills, 62 tool functions, and 13 CLI commands. The delivery cadence has been consistent and disciplined.
-->

---

## New Commands — v0.10.0 & v0.11.0

**`vaig discover`** — Autonomous Cluster Health Scanning
- Scans entire cluster topology without a predefined question
- Maps workloads, namespaces, resource utilization, and dependency graphs
- Surfaces anomalies proactively — before they become incidents

**`vaig doctor`** — Environment Healthcheck
- Validates configuration, credentials, cluster connectivity, and tool availability
- Pre-flight check before running live investigations

**`vaig feedback`** — Bug Reports & Feature Requests
- Submit structured feedback directly from the CLI
- Streamlines issue creation for the development team

**`vaig optimize`** — Tool Call Analysis & Report Quality
- Analyzes tool call patterns to identify inefficiencies in agent execution
- `--reports` flag: analyzes past report quality and suggests prompt improvements
- Drives the Adaptive Prompt Tuning system

<!-- Speaker notes:
Four new commands since v0.9.0. Discover and doctor address the proactive monitoring gap — teams no longer wait for alerts. Feedback closes the loop with end users. Optimize turns the platform's own telemetry into self-improvement.
-->

---

## Platform Intelligence — v0.11.0

**RAG Engine Integration with Vertex AI**
- Retrieval-Augmented Generation powered by Vertex AI embeddings
- Skills can query indexed documentation, runbooks, and past incident reports
- Reduces hallucination by grounding agent responses in real organizational data

**Shared Pipeline State Across Agents**
- Agents within a multi-agent pipeline share state — findings from one agent inform the next
- Eliminates redundant tool calls and improves cross-agent coherence

**Adaptive Prompt Tuning**
- System learns from past report quality and user feedback
- Automatically adjusts agent prompts based on analysis patterns
- Driven by `vaig optimize --reports` telemetry

**`/live` REPL Command**
- Interactive Read-Eval-Print Loop for iterative investigations
- Run follow-up queries without restarting the agent pipeline
- Maintains conversation context across commands

<!-- Speaker notes:
These are architectural capabilities, not just features. RAG grounding, shared state, and adaptive tuning represent the platform evolving from a stateless tool into an intelligent system that gets better with use.
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
| Test functions | **6,631** |
| Test files | **151** |
| Source files (`src/vaig/`) | **219** |
| Lines of code | **66,732** |
| Total commits | **291** |
| Merged PRs | **161** |
| Releases | **7** (v0.1.0–v0.11.0) |
| Skills | **33** |
| Tool functions | **62** |
| Config classes | **29** |
| CLI commands | **13** |

CI pipeline enforces: ruff (lint), mypy (strict type checking), pytest (all tests)

<!-- Speaker notes:
Over 6,631 tests across 151 test files. Strict mypy type checking. Every PR is blocked by CI. The codebase is not a prototype — it is production-grade Python with the discipline of a typed, tested, linted codebase.
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
- Every merge to `main` requires passing lint, types, and all 6,631 tests

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

The v0.11.0 cycle solidified the platform's intelligence layer. The trajectory is clear:

- **Deeper cost intelligence** — cross-namespace budget alerts, trend analysis, and anomaly detection on spend
- **Migration coverage expansion** — additional language pairs beyond Python→Go
- **Agentic CI integration** — automated PR review, test generation, and greenfield standards enforcement
- **Platform observability** — unified GKE health dashboard generated from vaig reports
- **Broader GCP surface** — Cloud Run, Cloud SQL, Pub/Sub, and Artifact Registry integration
- **Enhanced RAG** — deeper integration with organizational knowledge bases and incident history

The skill framework is designed for extension. Adding a new workflow requires implementing a single skill class — the orchestration, tooling, and output formatting are inherited.

<!-- Speaker notes:
We are not committing to a roadmap here. We are communicating architectural readiness. The platform can grow in any of these directions without rearchitecting the core.
-->

---

<!-- _class: lead -->

# Call to Action

---

## For Management — What the Team Delivered

**Across 7 releases (v0.1.0 → v0.11.0):**

- 161 pull requests merged across 291 commits
- 6,631 tests passing — zero regressions
- 33 specialized multi-agent skills, 62 tool functions, 13 CLI commands
- Key capabilities: cost estimation, cluster discovery, environment healthcheck, report optimization, RAG engine, adaptive prompt tuning
- Full documentation coverage synchronized with implementation

**The codebase:** 219 source files, 66,732 lines, 33 skills, 62 tool functions, 29 config classes

This was delivered with discipline: strict type checking, comprehensive testing, and structured prompt engineering. The work is production-ready.

<!-- Speaker notes:
This is the accountability slide. The team shipped real, tested, documented capabilities across seven releases. Not prototypes. Not demos. Production code with CI gates.
-->

---

## For C-Suite — Adopt vaig

**The operational problem is real:**
GKE cost overruns, slow incident resolution, and high engineering overhead in GCP operations are measurable costs — in engineering hours and infrastructure spend.

**vaig is the solution, already built:**
- 33 specialized AI workflows ready to deploy today
- Cost estimation with per-container waste detection
- Autonomous cluster discovery and environment healthcheck
- Incident diagnosis in seconds instead of hours
- Code migration and scaffolding at engineering scale
- RAG-grounded responses and adaptive prompt tuning

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

**Vertex AI Toolkit — Post-v0.11.0**

Repository: `vertex-ai-toolkit`
Documentation: `docs/`
Getting Started: `docs/getting-started.md`

*Questions?*

<!-- Speaker notes:
Open for questions. The full documentation is in the repository. For a live demo, we can run any of the commands shown in this deck against a real cluster.
-->
