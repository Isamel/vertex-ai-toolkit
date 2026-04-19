# Architecture

This document describes the system architecture of VAIG using ASCII diagrams.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLI LAYER (Typer)                               │
│                                                                              │
│  vaig chat  vaig ask  vaig live  vaig code  vaig discover  vaig fleet  ...   │
│  vaig doctor  vaig stats  vaig sessions  vaig skills  vaig models  vaig web  │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SKILLS LAYER (35+ built-in)                        │
│                                                                              │
│  service_health    rca             anomaly         code_migration            │
│  greenfield        migration       code_review     log_analysis              │
│  incident_comms    postmortem      runbook_gen      discovery                │
│  threat_model      perf_analysis   slo_review       ... 20 more              │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                             AGENTS LAYER                                     │
│                                                                              │
│  Orchestrator          SpecialistAgent         ToolAwareAgent               │
│  CodingAgent           CodingSkillOrchestrator InfraAgent                   │
│  InvestigationAgent    ChunkedProcessor                                      │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              TOOLS LAYER                                     │
│                                                                              │
│  gke/       gcloud_tools    file_tools    shell_tools    mcp_bridge          │
│  integrations/  knowledge/   repo/         test_runner   plugin_loader       │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                      │
│                                                                              │
│  GeminiClient  Settings     Auth           SessionManager  CostTracker       │
│  WorkspaceRAG  GitManager   MemoryPipeline EventBus        TelemetryCollector│
│  Investigation EvidenceLedger GlobalBudget  WorkspaceJail   PromptDefense    │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                   │
│                                                                              │
│  Vertex AI (Gemini)   GKE / Kubernetes   Cloud Logging   Cloud Monitoring    │
│  ChromaDB (local)     SQLite (local)     GitHub (gh CLI)  Datadog API        │
│  ArgoCD               Helm               MCP Servers      PagerDuty / Slack  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Map

### `cli/` — Command Layer

Typer-based CLI entry points. Each command in `cli/commands/` maps to a
top-level `vaig <verb>` subcommand. The REPL (`cli/repl.py`) handles the
interactive `vaig chat` session loop.

| Module | Responsibility |
|--------|----------------|
| `app.py` | Root Typer app, registers all command groups |
| `commands/ask.py` | Single-shot query → agent execution |
| `commands/chat.py` | Interactive REPL session |
| `commands/live.py` | Infrastructure health → service_health skill |
| `commands/_code.py` | `vaig code` subcommands (migrate, review, etc.) |
| `commands/discover.py` | Cluster/workload discovery |
| `commands/fleet.py` | Multi-cluster fleet operations |
| `commands/incident_cmd.py` | Incident workflow automation |
| `commands/remediate.py` | Fix application with skill execution |
| `commands/schedule.py` | Cron / scheduled skill runs |
| `commands/web.py` | Embedded FastAPI web UI |
| `repl.py` | Chat REPL loop with slash-command handling |
| `display.py` | Rich console formatting helpers |
| `export.py` | Report export to file |

---

### `agents/` — Multi-Agent Pipelines

Agents are stateless execution units. Each receives a system prompt, a tool
registry, and a user prompt; they loop until they produce a final response.

| Module | Responsibility |
|--------|----------------|
| `base.py` | `BaseAgent`, `AgentResult` dataclass |
| `specialist.py` | `SpecialistAgent` — text-only, no tool use |
| `tool_aware.py` | `ToolAwareAgent` — tool-use loop (up to N iterations) |
| `coding.py` | `CodingAgent` — single-agent file I/O + shell (simple tasks) |
| `coding_pipeline.py` | `CodingSkillOrchestrator` — 3-agent pipeline (see Phase 1 below) |
| `infra_agent.py` | `InfraAgent` — GKE + gcloud tool specialisation |
| `investigation_agent.py` | `InvestigationAgent` — hypothesis–test–refine loop |
| `orchestrator.py` | `Orchestrator` — multi-phase skill coordination |
| `chunked.py` | `ChunkedProcessor` — map-reduce over large inputs |
| `registry.py` | Agent registry / factory |
| `mixins.py` | Shared agent behaviour mixins |
| `utils.py` | Agent utility helpers |

---

### `skills/` — Domain-Specific Workflows

Skills are named workflows that compose agents into multi-step pipelines.
They are discovered and loaded by `skills/loader.py` and `skills/registry.py`.
Each skill directory contains `skill.py` (logic) and `prompts.py` (prompt templates).

| Skill | Description |
|-------|-------------|
| `service_health/` | 4-agent pipeline: Gatherer → Analyzer → Verifier → Reporter |
| `rca/` | Root Cause Analysis with 5-Whys causal reasoning |
| `anomaly/` | TrendConfig-based threshold and anomaly detection |
| `code_migration/` | Language migration with IdiomGenerator + CodingSkillOrchestrator |
| `greenfield/` | 6-stage scaffold: spec → design → impl → test → docs → review |
| `migration/` | ETL / data migration planning and validation |
| `code_review/` | Automated PR and file code review |
| `test_generation/` | Test suite generation for existing code |
| `log_analysis/` | Structured log pattern extraction |
| `incident_comms/` | Incident communication drafts (status pages, Slack, PagerDuty) |
| `postmortem/` | Blameless postmortem generation |
| `runbook_generator/` | Operational runbook authoring |
| `threat_model/` | STRIDE threat modelling |
| `perf_analysis/` | Performance bottleneck identification |
| `capacity_planning/` | Capacity and scaling recommendations |
| `slo_review/` | SLO burn-rate and error-budget analysis |
| `discovery/` | Workload and dependency discovery |
| `adr_generator/` | Architecture Decision Record authoring |
| `change_risk/` | Change risk assessment |
| `compliance_check/` | Security and compliance policy validation |
| `config_audit/` | Configuration drift and audit |
| `cost_analysis/` | Cloud cost analysis |
| `db_review/` | Database schema and query review |
| `dependency_audit/` | Dependency vulnerability audit |
| `error_triage/` | Error classification and triage |
| `iac_review/` | IaC (Terraform, Helm) review |
| `network_review/` | Network topology and policy review |
| `pipeline_review/` | CI/CD pipeline review |
| `resilience_review/` | Resilience and SRE review |
| `alert_tuning/` | Alert threshold tuning |
| `api_design/` | REST / gRPC API design review |
| `toil_analysis/` | Toil quantification and reduction |

---

### `tools/` — Atomic Tool Functions

Tools are the lowest-level execution units. They make real side-effect calls
(kubectl, gcloud, file I/O, etc.) and are registered in a `ToolRegistry` that
agents query.

```
tools/
├── gke/                   GKE + Kubernetes tool suite
│   ├── _registry.py       create_gke_tools() factory
│   ├── _clients.py        K8s client cache, Autopilot detection, ArgoCD client
│   ├── _resources.py      Resource maps, aliases, gap detection
│   ├── _formatters.py     Table formatters for kubectl output
│   ├── _cache.py          TTL cache for repeated queries
│   ├── kubectl.py         get, describe, logs, top, get_labels
│   ├── diagnostics.py     events, rollout, container status, node conditions
│   ├── discovery.py       workloads, mesh, network topology
│   ├── mesh.py            Istio/ASM config, security, sidecars
│   ├── mutations.py       scale, restart, annotate, label
│   ├── security.py        RBAC check, exec_command
│   ├── helm.py            releases, status, history, values
│   ├── argocd.py          applications, diff, sync, history
│   ├── datadog.py         Datadog monitor / metric queries
│   ├── monitoring.py      Cloud Monitoring / metrics API
│   ├── trend_analysis.py  Time-series trend + threshold checks
│   ├── scaling.py         HPA / VPA scaling queries
│   └── billing.py         Cost estimation from resource usage
│
├── integrations/          External notification + ticketing tools
│   ├── slack.py           Post to Slack channels
│   ├── pagerduty.py       Incident trigger / resolve
│   ├── opsgenie.py        Alert management
│   └── github.py          PR creation, issue management
│
├── knowledge/             Knowledge and memory retrieval tools
│   ├── search_rag.py      Query the RAG memory store
│   ├── recall_similar_cases.py  Historical case lookup
│   ├── query_pattern_history.py Pattern recurrence queries
│   └── fetch_doc.py       Fetch external documentation
│
├── repo/                  Remote repository tools
│   ├── shallow_clone.py   Shallow git clone for code review
│   ├── knowledge.py       Extract repo knowledge
│   └── batch.py           Batch repo operations
│
├── file_tools.py          read, write, edit, patch, list, search, verify_completeness
├── shell_tools.py         run_command (sandboxed subprocess)
├── gcloud_tools.py        Cloud Logging queries, Cloud Monitoring
├── test_runner.py         TestRunnerTool — detect and run test suites
├── mcp_bridge.py          MCP (Model Context Protocol) tool bridge
└── plugin_loader.py       Dynamic Python plugin tool loader
```

---

### `core/` — Config, Memory, RAG, Git, Shared Utilities

Foundation layer consumed by all other packages. No business logic here —
only infrastructure and cross-cutting concerns.

| Module | Responsibility |
|--------|----------------|
| `client.py` | `GeminiClient` — sync + async Vertex AI calls, retry, streaming |
| `config.py` | `Settings` — Pydantic v2 hierarchical config (env > YAML > defaults) |
| `auth.py` | ADC, Service Account impersonation, gcloud token refresh |
| `workspace_rag.py` | `WorkspaceRAG` — ChromaDB local vector index (see Phase 2 below) |
| `git_integration.py` | `GitManager` — git/gh lifecycle automation (see Phase 3 below) |
| `memory/` | Finding persistence pipeline (see Phase 5 below) |
| `investigation.py` | `InvestigationPlan` execution primitives |
| `evidence_ledger.py` | `EvidenceLedger` — append-only per-run evidence store |
| `self_correction.py` | `SelfCorrectionController` — detect circles/contradictions |
| `global_budget.py` | `GlobalBudgetManager` — cross-session token budget enforcement |
| `context_budget.py` | Per-call context window budget tracking |
| `event_bus.py` | `EventBus` — in-process event publish/subscribe |
| `events.py` | Event type definitions (`LoopStepEvent`, etc.) |
| `subscribers/` | Event subscribers: audit, memory, telemetry, fix_outcome |
| `session.py` (session/) | `SessionManager` — SQLite session + conversation store |
| `cost_tracker.py` | Per-session cost accumulation and reporting |
| `telemetry.py` | `TelemetryCollector` — buffered SQLite telemetry |
| `cache.py` | `ResponseCache` — LRU + TTL in-memory response cache |
| `prompt_defense.py` | `wrap_untrusted_content()`, anti-hallucination rules |
| `language.py` | 9-language detection for multilingual CLI output |
| `workspace_jail.py` | `WorkspaceJail` — copy-on-write workspace isolation |
| `circuit_breaker.py` | Circuit breaker for external service calls |
| `schemas.py` | Shared Pydantic schemas (`VerificationReport`, etc.) |
| `exceptions.py` | Custom exceptions (`BudgetExhaustedError`, etc.) |
| `report_store.py` | Persistent report storage |
| `rag.py` | Base RAG abstractions |

---

### `integrations/` — Notification Dispatch

Outbound integrations for reports and alerts.

| Module | Responsibility |
|--------|----------------|
| `dispatcher.py` | Route findings to configured channels |
| `slack.py` | Slack message formatting and posting |
| `pagerduty.py` | PagerDuty incident lifecycle |
| `jira.py` | Jira ticket creation |
| `google_chat.py` | Google Chat notifications |
| `email_sender.py` | SMTP email dispatch |
| `finding_exporter.py` | Export structured findings to integrations |
| `webhook_server.py` | Inbound webhook receiver |

---

### `platform/` — Multi-Tenant API (optional)

FastAPI application for hosted / team deployments with JWT auth and Firestore
persistence. Only loaded when running `vaig web` or in server mode.

---

### `web/` — Embedded Web UI (optional)

FastAPI + SSE real-time web interface. Routes mirror the CLI commands.
Loaded on `vaig web` or `--web` flag.

---

## New Components (Phase 1–8)

### Phase 1 — CodingSkillOrchestrator (3-Agent Pipeline)

`src/vaig/agents/coding_pipeline.py`

Replaces the single `CodingAgent` for complex coding tasks. Three specialised
agents execute sequentially with strict role separation:

```
User task
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  CodingSkillOrchestrator                              │
│                                                       │
│  1. Planner (temperature=0.4)                         │
│     • Reads codebase with read_file / list_files      │
│     • Writes PLAN.md (task, files, interfaces, tests) │
│     │                                                 │
│     ▼                                                 │
│  2. Implementer (temperature=0.1)  ◄──────────────┐  │
│     • Reads PLAN.md                               │  │
│     • Writes ALL files (zero placeholders)        │  │
│     • Runs verify_completeness                    │  │
│     │                                             │  │
│     ▼                                             │  │
│  3. Verifier (temperature=0.1)                    │  │
│     • Runs verify_completeness on all files       │  │
│     • Syntax checks via run_command               │  │
│     • Emits structured VerificationReport         │  │
│     │                                             │  │
│     ├─── PASS ──────────────────────────────────►│  │
│     │                                             │  │
│     └─── FAIL ──► build feedback (top 5 issues) ─┘  │
│                   (up to max_fix_iterations)          │
└───────────────────────────────────────────────────────┘
    │
    ▼
CodingPipelineResult
  .plan                  ← PLAN.md content
  .implementation_summary
  .verification_report
  .success               ← bool
  .usage                 ← aggregated token usage
```

Key design decisions:
- Each agent is a `ToolAwareAgent` — reuses existing infrastructure
- Fix-forward loop caps feedback at 5 issues to prevent prompt bloat
- `WorkspaceJail` can isolate changes to a temp directory (`workspace_isolation=True`)
- `WorkspaceRAG` is optionally registered as `search_workspace_knowledge` tool

---

### Phase 2 — WorkspaceRAG (ChromaDB Local Vector Index)

`src/vaig/core/workspace_rag.py`

Provides semantic code search over the local workspace using ChromaDB.
Registered as the `search_workspace_knowledge` tool in `CodingSkillOrchestrator`.

```
workspace/
    │
    ▼
WorkspaceRAG.build_index()
    │  Walk files matching configured extensions
    │  Skip: .git, .venv, node_modules, __pycache__, etc.
    │  Skip: files > 1 MB
    │  Chunk: 200 lines with 20-line overlap
    │
    ▼
ChromaDB PersistentClient
  path: workspace/.vaig/workspace-index/
  collection: "workspace"
  documents: text chunks
  metadatas: {file: str, chunk_index: int}
    │
    ▼
WorkspaceRAG.search(query, k=5)
    │  Auto-rebuild if stale (mtime check)
    │  ChromaDB query_texts (built-in embedding)
    │  Convert L2 distance → similarity score (0–1)
    │
    ▼
[{file, chunk, score}, ...]
```

---

### Phase 3 — GitManager (Git Lifecycle Automation)

`src/vaig/core/git_integration.py`

Subprocess-based wrapper around `git` and `gh` CLI. Gated by `GitConfig.enabled`
(disabled by default — all methods are no-ops when `enabled=False`).

```
CodingSkillOrchestrator pipeline
    │
    ├── GitManager.create_branch("vaig/add-retry-logic")
    │     git checkout -b <branch>
    │     Guard: refuses to operate if already on main/master
    │
    ├── [pipeline writes files]
    │
    ├── GitManager.commit_all("feat(retry): add exponential back-off")
    │     git add -A
    │     git commit -m <message>
    │     Guard: raises GitSafetyError on protected branches
    │
    ├── GitManager.push(set_upstream=True)
    │     git push -u origin <branch>
    │
    └── GitManager.create_pr(title, body, base="main")
          gh pr create --title ... --body ... --base main
          Returns: PR URL string

Safety exceptions:
  GitSafetyError  — operation on main/master
  GitDirtyError   — uncommitted changes when clean tree required
```

---

### Phase 4 — IdiomGenerator (LLM-Powered Idiom Map)

`src/vaig/skills/code_migration/idiom_generator.py`

Generates language-pair idiom maps (e.g., Python → Go) using an LLM, then
caches them to `~/.vaig/idioms/<source>_to_<target>.yaml` for reuse.

```
code_migration skill
    │
    ▼
IdiomGenerator.generate(source_lang, target_lang)
    │
    ├── Check cache: ~/.vaig/idioms/python_to_go.yaml
    │   └── If exists → load and return
    │
    └── LLM call (GeminiClient)
        Prompt: structured YAML schema for idiom map
        Response: YAML with idioms[] + dependencies{}
        │
        └── Parse YAML → validate schema
            Save to cache → return IdiomMap
```

---

### Phase 5 — Memory Pipeline (Finding Persistence + Recurrence)

`src/vaig/core/memory/`

Persists investigation findings across runs and tracks recurrence patterns
to surface chronic issues.

```
Investigation / service_health run
    │
    ▼
memory/fingerprint.py — ObservationFingerprint
    Deterministic hash of (category, service, title, description)

memory/pattern_store.py — PatternMemoryStore (SQLite)
    Table: pattern_entries
    Fields: fingerprint, category, service, title, first_seen, last_seen, count

memory/recurrence.py — RecurrenceAnalyzer
    For each finding:
        Look up fingerprint in PatternMemoryStore
        If found and count >= chronic_threshold → CHRONIC
        If found and count >= recurrence_threshold → RECURRING
        If new → FIRST_OCCURRENCE
    Returns: {fingerprint: RecurrenceSignal}

memory/memory_rag.py — MemoryRAG
    Vector search over past findings for similar case recall

memory/outcome_store.py — OutcomeStore
    Records resolution outcomes per fingerprint

memory/memory_correction.py — MemoryCorrection
    Feeds prior failure patterns into InvestigationAgent prompt
    to avoid repeating unsuccessful tool calls (MEM-05)
```

---

### Phase 6 — Investigation Pipeline (Budget-Enforced Hypothesis Loop)

`src/vaig/agents/investigation_agent.py`

Autonomous investigation with hypothesis → tool call → evidence loop,
bounded by a global token budget.

```
service_health skill (Orchestrator)
    │
    ▼
InvestigationAgent (extends ToolAwareAgent)
    │
    ├── Receives InvestigationPlan with ordered InvestigationStep[]
    │
    └── For each step:
        1. Check memory_correction — skip if prior failure exists
        2. Check EvidenceLedger cache — skip if already answered
        3. Execute tool call → append EvidenceEntry to ledger
        4. SelfCorrectionController.decide():
           │  CONTINUE  → proceed to next step
           │  CIRCLE    → skip repeated tool pattern
           │  ESCALATE  → break loop, return partial results
        5. GlobalBudgetManager.check() → raise BudgetExhaustedError if exhausted

Loop terminates on:
  - All steps complete / skipped
  - BudgetExhaustedError
  - max_iterations reached
  - SelfCorrectionController returns ESCALATE
```

---

### Phase 7 — Anomaly Detection (TrendConfig Threshold Analysis)

`src/vaig/skills/anomaly/`
`src/vaig/tools/gke/trend_analysis.py`

Threshold-based anomaly detection using configurable `TrendConfig` policies.

```
vaig live (or scheduled run)
    │
    ▼
anomaly skill → TrendAnalysisTool
    │
    ├── Fetch time-series metrics from Cloud Monitoring / Datadog
    │
    ├── TrendConfig per metric:
    │   { metric, window_minutes, p95_threshold, spike_multiplier, min_samples }
    │
    └── For each metric window:
        Compute p50, p95, trend slope
        Compare against thresholds
        Emit AnomalyEvent if threshold exceeded
        Return: [{metric, value, threshold, severity, trend}]
```

---

## Data Flow Diagrams

### `vaig live` — Infrastructure Health Analysis

```
User: vaig live "Why are pods crashing?"
    │
    ▼
cli/commands/live.py
    Build Settings + tool registry (GKE tools, Cloud Logging, Cloud Monitoring)
    │
    ▼
service_health skill (Orchestrator)
    │
    ├── Language detection + Autopilot injection
    │
    ├── Pass 1 — health_gatherer (ToolAwareAgent)
    │   Loop (max 25 iterations):
    │     Gemini → function_call → K8s / Cloud Logging tools
    │     kubectl_get("pods") → pod list
    │     get_events(type=Warning) → events
    │     gcloud_logging_query(severity>=ERROR) → log entries
    │   Output: gathered data + Investigation Checklist
    │
    ├── Validate gatherer output
    │   If incomplete → Pass 2 (deepening, same history, no reset)
    │
    ├── health_analyzer (SpecialistAgent)
    │   Causal reasoning (5 Whys), management context detection
    │   Verification Gap annotation per finding
    │   Output: structured findings with confidence levels
    │
    ├── health_verifier (ToolAwareAgent)
    │   For each finding with Verification Gap:
    │     Run targeted tool call
    │     Upgrade/downgrade confidence
    │   Output: verified findings
    │
    ├── health_reporter (SpecialistAgent, JSON Schema mode)
    │   Gemini → response_schema=HealthReport, mime=application/json
    │   post_process_report() → validate + to_markdown()
    │   Output: final report
    │
    └── RecurrenceAnalyzer — annotate findings with CHRONIC/RECURRING signals
        Memory subscribers → persist findings to PatternMemoryStore
    │
    ▼
cli/display.py → Rich console output
User: formatted report + cost summary
```

---

### `vaig code migrate` — Language Migration Pipeline

```
User: vaig code migrate --from python --to go ./src
    │
    ▼
cli/commands/_code.py
    Build Settings + CodingSkillOrchestrator
    │
    ▼
code_migration skill
    │
    ├── Phase 1 — IdiomGenerator
    │   Check ~/.vaig/idioms/python_to_go.yaml
    │   If missing → LLM call → cache YAML idiom map
    │
    ├── Phase 2 — File Discovery
    │   Walk source directory, collect .py files
    │   Optionally build WorkspaceRAG index
    │
    ├── Phase 3 — CodingSkillOrchestrator.run() per file batch
    │   Planner: read source + idiom map → write PLAN.md
    │   Implementer: translate files → write .go output
    │   Verifier: compile check, interface match, placeholder scan
    │   Fix-forward loop if verification fails
    │
    ├── Phase 4 — GitManager (if enabled)
    │   create_branch("vaig/migrate-python-to-go")
    │   commit_all("feat(migration): translate src/ from Python to Go")
    │   push() → create_pr()
    │
    └── Phase 5 — Report
        Files written, verification status, PR URL (if created)
    │
    ▼
User: migration report
```

---

### `vaig ask` — Coding Pipeline

```
User: vaig ask "Add retry logic to the GCS upload function"
    │
    ▼
cli/commands/ask.py
    Detect task type → route to CodingSkillOrchestrator or CodingAgent
    Build Settings + GeminiClient
    │
    ▼
CodingSkillOrchestrator.run(task)
    │
    ├── WorkspaceJail — optionally copy workspace to temp dir
    │
    ├── Planner (gemini-2.5-pro, temp=0.4)
    │   search_workspace_knowledge("GCS upload retry") → relevant chunks
    │   read_file("gcs_client.py") → current implementation
    │   write_file("PLAN.md") → implementation plan
    │
    ├── Implementer (gemini-2.5-pro, temp=0.1)
    │   read_file("PLAN.md")
    │   patch_file("gcs_client.py", diff) → add retry with exponential back-off
    │   run_command("python -m pytest tests/test_gcs.py -x")
    │   verify_completeness(["gcs_client.py"])
    │
    └── Verifier (gemini-2.5-flash, temp=0.1)
        verify_completeness(["gcs_client.py"])
        run_command("python -m py_compile gcs_client.py")
        Output: VerificationReport{success: true}
    │
    ▼
CodingPipelineResult
    │
    ├── GitManager.commit_all (if enabled)
    │
    ▼
cli/display.py → rich diff + verification report
User: files modified, test results, cost summary
```

---

## Service Health Pipeline (4-Agent Sequential)

Detailed sequence for `vaig live`:

```
User
 │  "Why are pods crashing?"
 ▼
vaig live (CLI)
 │  execute_with_tools(query, skill="service_health", registry)
 ▼
Orchestrator
 │  Language detection + Autopilot cluster injection
 │
 │  ┌──────────────────────────────────────────────────────┐
 │  │ PASS 1: Data Collection                              │
 │  │ health_gatherer (ToolAwareAgent, max_iter=25)        │
 │  │   Gemini ↔ K8s: kubectl_get(pods)                   │
 │  │   Gemini ↔ K8s: get_events(Warning)                 │
 │  │   Gemini ↔ K8s: kubectl_describe(replicasets)       │
 │  │   Gemini ↔ Cloud Logging: severity>=ERROR           │
 │  │ → gathered data + Investigation Checklist            │
 │  └──────────────────────────────────────────────────────┘
 │
 │  validate_gatherer_output() → if incomplete:
 │  ┌──────────────────────────────────────────────────────┐
 │  │ PASS 2: Incremental Deepening                        │
 │  │ health_gatherer (same history, no reset)             │
 │  │   Additional targeted tool calls only                │
 │  │ → merge Pass 1 + Pass 2                              │
 │  └──────────────────────────────────────────────────────┘
 │
 │  ┌──────────────────────────────────────────────────────┐
 │  │ ANALYSIS                                             │
 │  │ health_analyzer (SpecialistAgent)                    │
 │  │   Causal reasoning, 5-Whys, correlation              │
 │  │   Verification Gap annotation per finding            │
 │  │ → structured findings with confidence levels         │
 │  └──────────────────────────────────────────────────────┘
 │
 │  ┌──────────────────────────────────────────────────────┐
 │  │ VERIFICATION                                         │
 │  │ health_verifier (ToolAwareAgent)                     │
 │  │   For each finding with Verification Gap:            │
 │  │     → targeted tool call → upgrade/downgrade conf.   │
 │  │ → verified findings                                  │
 │  └──────────────────────────────────────────────────────┘
 │
 │  ┌──────────────────────────────────────────────────────┐
 │  │ REPORT GENERATION                                    │
 │  │ health_reporter (SpecialistAgent, JSON Schema mode)  │
 │  │   Gemini → response_schema=HealthReport              │
 │  │            mime_type=application/json                │
 │  │   post_process_report() → validate → to_markdown()  │
 │  │ → final report                                       │
 │  └──────────────────────────────────────────────────────┘
 │
 ▼
OrchestratorResult → CLI → Rich console + cost summary
```

---

## Tool Layer — GKE Package Structure

```
src/vaig/tools/gke/
├── __init__.py          re-exports
├── _registry.py         create_gke_tools() factory
├── _clients.py          K8s client cache, Autopilot detection, ArgoCD client
├── _resources.py        Resource maps, aliases, gap detection
├── _formatters.py       Table formatters for kubectl output
├── _cache.py            TTL cache for repeated queries
├── kubectl.py           get, describe, logs, top, get_labels
├── diagnostics.py       events, rollout, container status, node conditions
├── discovery.py         workloads, mesh, network topology
├── mesh.py              Istio/ASM config, security, sidecars
├── mutations.py         scale, restart, annotate, label
├── security.py          RBAC check, exec_command
├── helm.py              releases, status, history, values
├── argocd.py            applications, diff, sync, history
├── argo_rollouts.py     Argo Rollouts progressive delivery
├── datadog.py           Datadog monitor / metric queries (tool wrapper)
├── datadog_api.py       Datadog API client
├── monitoring.py        Cloud Monitoring metrics API
├── metrics_api.py       Metrics API abstractions
├── trend_analysis.py    TrendConfig-based threshold and anomaly analysis
├── scaling.py           HPA / VPA scaling queries
├── billing.py           Resource cost estimation
└── cost_estimation.py   Cost modelling helpers
```

---

## ArgoCD Connection Topologies

```
Topology A — Same Cluster
─────────────────────────
  VAIG ──► Cluster A (workloads + ArgoCD)


Topology B — Management Cluster (same project)
───────────────────────────────────────────────
  VAIG ──[GKE tools, default kubeconfig]──► Cluster A (workloads)
  VAIG ──[ArgoCD tools, argocd.context]──► Cluster M (ArgoCD)


Topology C — Different Project
───────────────────────────────
  VAIG ──[GKE tools]────────────────────► Cluster A (Project Y)
  VAIG ──[ArgoCD tools, argocd.context]──► Cluster M (Project X)


Topology D — ArgoCD REST API (SaaS / any topology)
───────────────────────────────────────────────────
  VAIG ──[GKE tools]─────────────────────► Any Cluster
  VAIG ──[ArgoCD REST API, server+token]──► ArgoCD Server (any location)
                                              │
                                              └──manages──► Any Cluster
```

---

## Configuration Layering

```
Environment Variables         highest priority
  VAIG_GCP__PROJECT_ID=...
        │
        ▼
YAML Config                   medium priority
  config/default.yaml
        │
        ▼
Code Defaults                 lowest priority
  Pydantic Field defaults
        │
        ▼
  Merged Settings (Pydantic v2)
        │
   ┌────┼────────┬──────────────┬──────────────┐
   ▼    ▼        ▼              ▼              ▼
GeminiClient  ToolRegistry  SessionManager  TelemetryCollector
```

---

## Cost Tracking Flow

```
User sends message
    │
    ▼
REPL._check_budget()
  CostTracker → OK / WARNING / EXCEEDED
    │
    ▼
API call (GeminiClient)
    │
    ▼
CostTracker.record(model, tokens_in, tokens_out, thinking)
  → calculate_cost() → accumulate session total
    │
User: /cost
    ▼
CostTracker.summary()
  → total cost, per-model breakdown
    │
User: /quit
    ▼
SessionManager.save_cost_data(tracker.to_dict())
  → SQLite: UPDATE sessions SET metadata = ...
    │
User: vaig chat --resume
    ▼
SessionManager.load_cost_data(session_id)
  → SQLite: SELECT metadata FROM sessions
  → restored CostTracker
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| CLI framework | Typer + Rich |
| Data validation | Pydantic v2 |
| LLM API | google-genai (Vertex AI / Gemini) |
| Vector search | ChromaDB (optional, local) |
| Persistence | SQLite (sessions, telemetry, memory) |
| Git automation | subprocess → git + gh CLI |
| Web UI | FastAPI + SSE + Jinja2 |
| Platform API | FastAPI + Firestore + JWT |
| Infra access | kubernetes Python client + kubectl |
| Config | YAML + env vars (Pydantic Settings) |
| Packaging | pyproject.toml (hatchling) |
