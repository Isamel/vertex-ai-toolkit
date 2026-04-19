# Agents Guide

VAIG uses a multi-agent architecture where specialized agents collaborate to solve complex problems. Each skill defines its own agent pipeline, and the Orchestrator coordinates their execution.

## Agent Hierarchy

```
BaseAgent (ABC)
├── SpecialistAgent          — Direct model calls (no tools)
├── ToolAwareAgent           — Generic agent with configurable tools
│   ├── CodingAgent          — File I/O + shell commands (single-agent mode)
│   ├── InfraAgent           — GKE + GCloud tools for SRE
│   └── InvestigationAgent   — Deterministic step-runner with budget + memory
├── Orchestrator             — Coordinates multi-agent execution
├── CodingSkillOrchestrator  — 3-agent coding pipeline (Planner→Implementer→Verifier)
└── ChunkedProcessor         — Map-Reduce for large files
```

---

## Agent Types

### SpecialistAgent

The simplest agent type. Wraps a `GeminiClient` for direct model calls without any tool access. Used for analysis, planning, and report generation where tool access is not needed.

```python
{
    "name": "health_analyzer",
    "role": "Health Pattern Analyzer",
    "requires_tools": False,   # → instantiated as SpecialistAgent
    "system_instruction": "You are an SRE analysis specialist...",
    "model": "gemini-2.5-flash",
    "temperature": 0.2,
}
```

### ToolAwareAgent

A generic agent with a configurable tool registry and an autonomous tool-use loop. It calls tools, observes results, and continues reasoning until the task is complete or `max_iterations` is reached.

Skills mark agents that need tools with `requires_tools: True`:

```python
{
    "name": "health_verifier",
    "role": "Health Finding Verifier",
    "requires_tools": True,    # → instantiated as ToolAwareAgent
    "tool_categories": ["kubernetes", "scaling", "mesh"],
    "max_iterations": 15,
    "temperature": 0.2,
}
```

### CodingAgent

Single-agent mode for software engineering tasks. Uses `ToolLoopMixin` with `temperature=0.2`, `frequency_penalty=0.15`, and `max_iterations` from `coding.max_tool_iterations` (default: 25).

**Available tools:**
- **File tools**: `read_file`, `write_file`, `edit_file`, `list_files`, `search_files`, `verify_completeness`
- **Shell tools**: `run_command` (with allowlist/denylist from config)
- **Plugin tools**: MCP and Python module plugins (when configured)

Destructive tools (`write_file`, `edit_file`, `run_command`) can require explicit confirmation via an injectable `confirm_fn` callback when `coding.confirm_actions: true`.

Activated via `vaig ask --code` or `/code` in the REPL.

```bash
# Single-agent coding mode
vaig ask "Refactor auth module to use dependency injection" -f auth.py --code

# In the REPL
/code
> Add type hints to all functions in utils.py
```

### CodingSkillOrchestrator

A 3-agent pipeline for complex coding tasks that benefit from a planning phase. Activated with `--pipeline` on `vaig ask --code`, or by setting `coding.pipeline_mode: true` in config.

**Pipeline flow:**

```
[User Request]
      │
      ▼
 ┌──────────┐  temperature=0.4    Reads codebase, writes PLAN.md
 │  Planner │  orchestrator_model Produces architecture decisions, file
 └──────────┘  (gemini-2.5-pro)   breakdown, edge cases
      │
      │  PLAN.md as context
      ▼
 ┌─────────────┐  temperature=0.1  Full file tool access (read/write/edit/
 │ Implementer │  implementer_model shell/verify_completeness). Prefers
 └─────────────┘  (gemini-2.5-pro) patch_file for existing files,
      │                             write_file for new files.
      │  Implementation as context
      ▼
 ┌──────────┐  temperature=0.1    Runs verify_completeness + syntax checks.
 │ Verifier │  verifier_model     Produces structured VerificationReport
 └──────────┘  (gemini-2.5-flash) (JSON) or regex-matched PASS/FAIL.
      │
      ├─ PASS → [Final Result]
      │
      └─ FAIL → structured XML feedback (top 5 issues)
                     │
                     └─→ re-runs Implementer (up to max_fix_iterations)
                               └─→ Verifier again
                                     ├─ PASS → [Final Result]
                                     └─ FAIL → [Error with verifier findings]
```

**Model assignments:**

| Agent | Config key | Default |
|-------|-----------|---------|
| Planner | `coding.orchestrator_model` | client's current model (`gemini-2.5-pro`) |
| Implementer | `coding.implementer_model` | client's current model (`gemini-2.5-pro`) |
| Verifier | `coding.verifier_model` | `settings.models.fallback` (`gemini-2.5-flash`) |

**Configuration:**

```yaml
coding:
  pipeline_mode: true           # Enable 3-agent pipeline (default: false)
  max_fix_iterations: 1         # Verifier→Implementer retry loops (default: 1)
  workspace_isolation: false    # Copy workspace to temp dir; sync back on success
  confirm_actions: false        # Pipeline mode: must be false (non-interactive)
  max_tool_iterations: 25       # Tool loop cap per agent
```

```bash
# Pipeline mode via flag
vaig ask "Implement a rate limiter with Redis" --code --pipeline

# Pipeline mode always-on
# coding:
#   pipeline_mode: true
vaig ask "Implement a rate limiter with Redis" --code
```

> **Note:** `confirm_actions` is not supported in pipeline mode. Set to `false` when using `--pipeline`.

### InfraAgent

SRE-focused agent for live Kubernetes investigation. Uses `ToolLoopMixin` with `temperature=0.2`, `frequency_penalty=0.15`, and `max_tool_iterations=25`.

**Available tools (17 total):**

| Category | Tools |
|----------|-------|
| GKE read (13) | `kubectl_get`, `kubectl_describe`, `kubectl_logs`, `kubectl_top`, `get_events`, `get_rollout_status`, `get_rollout_history`, `get_node_conditions`, `get_container_status`, `check_rbac`, `helm_*`, `argocd_*` |
| GKE write (4) | `kubectl_scale`, `kubectl_restart`, `kubectl_label`, `kubectl_annotate` |
| GCloud (2) | `gcloud_logging_query`, `gcloud_monitoring_query` |
| GKE exec (1) | `exec_command` — **disabled by default**, requires `gke.exec_enabled: true` |

Activated via `vaig live` or `vaig ask --live`.

```bash
vaig live "Why is payment-service returning 503s?" \
  --cluster prod --namespace payments

vaig ask --live "Show me nodes with memory pressure"
```

### InvestigationAgent

Extends `ToolAwareAgent` with a **deterministic step execution** model. Instead of a free-form tool-use loop, it drives `InvestigationPlan` steps one by one.

**Per-step execution order:**

```
For each step in InvestigationPlan:
  1. Cache check      — skip if identical tool call was already made
  2. Budget check     — abort if GlobalBudgetManager.is_exhausted()
  3. MEM-05 hook      — check PatternMemoryStore for known patterns
  4. SH-06 hook       — SelfCorrectionController check (CONTINUE / FORCE_DIFFERENT / ESCALATE)
  5. Tool call        — execute the planned tool
  6. Evidence ledger  — append result to evidence list
```

**Termination conditions:**
- All steps complete or skipped
- `BudgetExhaustedError` raised
- `max_iterations` reached (default: 10)
- `SelfCorrectionController` returns `ESCALATE`

Activated via `investigation.enabled: true` in config (see [Investigation Pipeline](#investigation-pipeline) below).

---

## Orchestrator

Coordinates multi-agent execution. Takes a list of agents defined by a skill and runs them according to a strategy.

**Execution strategies:**

| Strategy | Description | How triggered |
|----------|-------------|---------------|
| `sequential` | Agent N's output becomes context for Agent N+1 | Default |
| `fanout` | All agents run in parallel (ThreadPoolExecutor), results merged | Skill declares no `parallel_group` but sets strategy |
| `parallel_sequential` | Parallel group runs concurrently, then sequential tail follows | Any agent config has a `parallel_group` key |
| `lead-delegate` | Lead agent invokes sub-agents as tools via `injectable_agents` | Skill configures `injectable_agents` key |

The Orchestrator auto-detects `parallel_group` keys in the returned configs and upgrades from `"sequential"` to `"parallel_sequential"` automatically.

```python
@dataclass
class OrchestratorResult:
    final_response: str
    agent_results: list[AgentResult]
    strategy: str
    total_tokens: int
```

---

## ChunkedProcessor

Handles files that exceed the model's context window using a Map-Reduce approach:

1. **Chunk** — Split the input into overlapping chunks
2. **Map** — Process each chunk independently with a specialist agent
3. **Reduce** — Merge all chunk results into a final analysis

```yaml
chunking:
  chunk_overlap_ratio: 0.1      # 10% overlap between chunks
  token_safety_margin: 0.1      # Reserve 10% of context for prompt
  chars_per_token: 2.0          # Character-to-token ratio estimate
  inter_chunk_delay: 2.0        # Seconds between chunk API calls
```

---

## Skill Pipelines

### Service Health Pipeline

**Default (parallel-then-sequential):**

```
[vaig check / vaig ask --live "health of X"]
            │
            ▼
    ┌───────────────────────────────────────────────────┐
    │              Parallel Group: "gather"             │
    │                                                   │
    │  node_gatherer    ── GKE node/cluster health      │
    │  workload_gatherer── pods, deployments, HPA       │
    │  event_gatherer   ── events, networking, storage  │
    │  logging_gatherer ── Cloud Logging errors/warns   │
    │  datadog_gatherer*── APM/metrics correlation      │
    └───────────────────────────────────────────────────┘
            │  merged output
            ▼
     health_analyzer   (gemini-2.5-flash, temp=0.2)
            │  pattern analysis + Verification Gaps
            ▼
    [optional: investigation phase — see below]
            │
            ▼
     health_verifier   (gemini-2.5-flash, temp=0.2, max_iter=15)
            │  confirmed/upgraded findings
            ▼
     health_reporter   (gemini-2.5-flash, temp=0.3, structured JSON)
            │
            ▼
      HealthReport → Markdown
```

`* datadog_gatherer` only appears when `datadog.enabled: true`.

**Sub-gatherer model assignments:**

| Agent | Model | max_iterations | Scope |
|-------|-------|---------------|-------|
| `node_gatherer` | `gemini-2.5-pro` | 15 (4 on Autopilot) | Step 1: cluster & nodes |
| `workload_gatherer` | `gemini-2.5-pro` | 20 | Steps 2,4,5,6: pods/deployments/HPA |
| `event_gatherer` | `gemini-2.5-pro` | 10 | Steps 3,8,9,10: events/networking/storage/GitOps |
| `logging_gatherer` | `gemini-2.5-pro` | 8 | Steps 7a,7b: Cloud Logging |
| `datadog_gatherer` | `gemini-2.5-flash` | 8 | Datadog APM/metrics |
| `health_analyzer` | `gemini-2.5-flash` | — | Text-only pattern analysis |
| `health_verifier` | `gemini-2.5-flash` | 15 | Targeted verification calls |
| `health_reporter` | `gemini-2.5-flash` | — | Structured JSON report |

**Legacy sequential pipeline** (single monolithic gatherer — available for backward compat):

```
health_gatherer (gemini-2.5-pro, temp=0.0, max_iter=25)
      │
      ▼
health_analyzer → health_verifier → health_reporter
```

Use `ServiceHealthSkill.get_sequential_agents_config()` to access this path.

---

### Investigation Pipeline

Activated by setting `investigation.enabled: true`. Inserts two agents between `health_analyzer` and `health_verifier`:

```
health_analyzer
      │  analysis + hypotheses
      ▼
health_planner      (SpecialistAgent, gemini-2.5-flash, temp=0.1)
      │  InvestigationPlan (structured step list)
      ▼
health_investigator (InvestigationAgent, gemini-2.5-flash, temp=0.1)
      │  executed evidence per step
      ▼
health_verifier
```

**health_planner** converts the analyzer's hypotheses into a concrete `InvestigationPlan` — a deterministic list of tool calls and expected evidence.

**health_investigator** drives each step of the plan using the deterministic execution model described in [InvestigationAgent](#investigationagent).

#### Autonomous mode

When `investigation.autonomous_mode: true`, the investigator gets three additional controllers:

| Controller | Purpose |
|-----------|---------|
| `SelfCorrectionController` | Detects tool call circles and stale iteration loops; can issue `FORCE_DIFFERENT` or `ESCALATE` |
| `GlobalBudgetManager` | Enforces `budget_per_run_usd` — aborts when cost limit is reached |
| `PatternMemoryStore` | MEM-05 memory: checks `.vaig/memory/patterns/` for known patterns before making a tool call |

```yaml
investigation:
  enabled: true
  autonomous_mode: true         # Activates budget + memory + self-correction
  budget_per_run_usd: 0.50      # 0.0 = no cap
  max_iterations: 10            # Steps before force-stop
  max_steps_per_plan: 15
  circle_threshold: 2           # Same (tool, args) calls before flagging a circle
  memory_correction: true       # MEM-05 pre-action hook
```

**`SelfCorrectionConfig`** (applies globally when not overridden by `investigation.circle_threshold`):

```yaml
self_correction:
  enabled: false                # Auto-enables when non-default values set
  max_repeated_calls: 3         # Same (tool, args_hash) before flagging circle
  max_stale_iterations: 5       # Consecutive no-progress iterations → FORCE_DIFFERENT
  contradiction_sensitivity: 0.8
  max_budget_per_step_usd: 0.10
```

```bash
# Run investigation with budget cap
vaig check "Is payment-service healthy?" --namespace payments
# (investigation.enabled: true in vaig.yaml)
```

---

## Agent Roles

Agents are tagged with a role via `AgentRole`:

| Role | Description |
|------|-------------|
| `orchestrator` | Coordinates other agents |
| `specialist` | Domain expert (analysis, planning, reporting) |
| `assistant` | General-purpose helper |
| `coder` | File and code operations |
| `sre` | Infrastructure and operations |

---

## Configuration Reference

### `agents:` block

```yaml
agents:
  orchestrator_model: gemini-2.5-pro    # Planning / orchestration roles
  specialist_model: gemini-2.5-flash    # Analysis / reporting specialists
  max_concurrent: 3                     # Fan-out parallelism cap
  max_iterations_retry: 15
  parallel_tool_calls: true             # Async path; semaphore = max_concurrent_tool_calls
  max_failures_before_fallback: 2       # Rate-limit errors → switch to models.fallback
  min_inter_call_delay: 0.0             # RPM throttle; set >0 to avoid 429s
```

### `coding:` block

```yaml
coding:
  pipeline_mode: false          # 3-agent pipeline (Planner→Implementer→Verifier)
  workspace_root: "."
  workspace_isolation: false    # Copy workspace to temp dir; sync back on success
  max_tool_iterations: 25       # Tool loop cap per agent
  max_fix_iterations: 1         # Verifier→Implementer retry loops
  confirm_actions: false        # Require confirmation for destructive tools
  allowed_commands: []          # Shell allowlist (empty = all allowed)
  denied_commands: []           # Shell denylist
```

### `investigation:` block

```yaml
investigation:
  enabled: false                # Gates health_planner + health_investigator
  autonomous_mode: false        # Adds budget + memory + self-correction
  budget_per_run_usd: 0.0       # 0.0 = no cap
  max_iterations: 10
  max_steps_per_plan: 15
  circle_threshold: 2           # Overrides SelfCorrectionConfig when set
  memory_correction: true       # MEM-05 pre-action hook
```

---

## AgentConfig Reference

Every agent is instantiated from an `AgentConfig` dataclass (`src/vaig/agents/base.py`). Skills populate these fields via `get_agents_config()` dicts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Unique agent identifier |
| `role` | `str` | *required* | Human-readable role description |
| `system_instruction` | `str` | *required* | System prompt defining agent expertise |
| `model` | `str` | `"gemini-2.5-pro"` | Gemini model ID |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_output_tokens` | `int` | `16384` | Maximum response length |
| `frequency_penalty` | `float \| None` | `None` | Penalty for repeated tokens |
| `response_schema` | `type[BaseModel] \| None` | `None` | Pydantic model for structured JSON output |
| `response_mime_type` | `str \| None` | `None` | Set to `"application/json"` with `response_schema` |
| `requires_tools` | `bool` | `False` | `True` → ToolAwareAgent; `False` → SpecialistAgent |
| `tool_categories` | `list[str]` | `[]` | Tool categories to register (kubernetes, scaling, logging, …) |
| `max_iterations` | `int` | `10` | Tool-use loop cap |
| `parallel_group` | `str \| None` | `None` | Group key that triggers `parallel_sequential` strategy |
| `agent_class` | `str \| None` | `None` | Override agent class (e.g. `"InvestigationAgent"`) |
| `injectable_agents` | `list[str]` | `[]` | Sub-agents available as tools (lead-delegate strategy) |

When `response_schema` is set, `SpecialistAgent` passes it to `GeminiClient.generate()` via `types.GenerateContentConfig`. Gemini constrains its output to match the schema and returns a JSON string. The skill's `post_process_report()` method is responsible for converting the JSON to a display format.

---

## Viewing Agent Config

In the REPL, use `/agents` to see the current skill's agent pipeline:

```
/agents

Agents for skill: service-health (parallel_sequential)
  Parallel group: gather
    1. node_gatherer      (tool_aware) — Cluster & Node Health Gatherer
    2. workload_gatherer  (tool_aware) — Workload Health Gatherer
    3. event_gatherer     (tool_aware) — Events & Infrastructure Gatherer
    4. logging_gatherer   (tool_aware) — Cloud Logging Gatherer
  Sequential:
    5. health_analyzer    (specialist) — Health Pattern Analyzer
    6. health_verifier    (tool_aware) — Health Finding Verifier
    7. health_reporter    (specialist) — Health Report Generator
```

---

[Back to index](README.md)
