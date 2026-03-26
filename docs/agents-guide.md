# Agents Guide

VAIG uses a multi-agent architecture where specialized agents collaborate to solve complex problems. Each skill defines its own agent pipeline, and the Orchestrator coordinates their execution.

## Agent Hierarchy

```
BaseAgent (ABC)
├── SpecialistAgent          — Direct model calls (no tools)
├── ToolAwareAgent           — Generic agent with configurable tools
│   ├── CodingAgent          — File I/O + shell commands
│   └── InfraAgent           — GKE + GCloud tools for SRE
├── Orchestrator             — Coordinates multi-agent execution
├── CodingSkillOrchestrator  — 3-agent coding orchestrator (Planner→Implementer→Verifier)
└── ChunkedProcessor         — Map-Reduce for large files
```

## Agent Types

### SpecialistAgent

The simplest agent type. Wraps a `GeminiClient` for direct model calls without any tool access. Used for analysis, planning, and report generation where tool access is not needed.

```python
# Skills define specialist agents for phases like analysis and reporting
{
    "name": "evidence_collector",
    "role": "Evidence Collector",
    "system_instruction": "You are an evidence analysis specialist...",
    "model": "gemini-2.5-flash",
}
```

### ToolAwareAgent

A generic agent with a configurable tool registry and an autonomous tool-use loop. It can call tools, observe results, and continue reasoning until the task is complete.

Skills mark agents that need tools with `requires_tools: True`:

```python
{
    "name": "health_gatherer",
    "role": "Health Data Gatherer",
    "requires_tools": True,  # Gets instantiated as ToolAwareAgent
    "system_instruction": "You gather live health data from Kubernetes...",
}
```

### CodingAgent

Specialized agent for software engineering tasks. Has access to:
- **File tools**: `read_file`, `write_file`, `edit_file`, `list_files`, `search_files`, `verify_completeness`
- **Shell tools**: `run_command` (with allowlist)

Activated via `vaig ask --code` or `/code` in the REPL.

The CodingAgent uses `ToolLoopMixin` for its tool-use loop, with a configurable maximum of iterations (`coding.max_tool_iterations`, default: 25).

```bash
# Enable coding agent mode
vaig ask "Refactor this to use async/await" -f server.py --code

# In REPL
/code
> Refactor the auth module to use dependency injection
```

### CodingSkillOrchestrator

A 3-agent orchestrator for complex coding tasks that require a structured planning phase before writing code. Activate with `--pipeline` on `vaig ask --code` or by setting `coding.pipeline_mode: true` in config.

**Agent flow:**

```
[User Input]
    └─→ Planner
          └─→ Implementer (has full file tools access)
                └─→ Verifier
                      └─→ [Final Result]
```

- **Planner** — Generates a specification document: architecture decisions, file breakdown, edge cases
- **Implementer** — Executes the plan using the full `CodingAgent` tool set (read, write, edit, shell, `verify_completeness`)
- **Verifier** — Reviews the implementation against the plan and reports pass/fail. On failure, the pipeline short-circuits and returns the verifier's findings

The Verifier uses a `_parse_success` function that is robust to freeform model output — it checks for explicit failure indicators before assuming success, which prevents false positives from ambiguous responses.

```bash
# Pipeline mode — structured Planner → Implementer → Verifier
vaig ask "Implement a rate limiter with Redis" --code --pipeline

# Also configurable in vaig.yaml
# coding:
#   pipeline_mode: true
```

> **Note:** Pipeline mode does not support interactive `confirm_actions`. Use in non-interactive contexts or set `coding.confirm_actions: false` when using `--pipeline`.

### InfraAgent

SRE-focused agent with live infrastructure tools:
- **GKE read tools**: `kubectl_get`, `kubectl_describe`, `kubectl_logs`, `kubectl_top`, `get_events`, `get_rollout_status`, `get_rollout_history`, `get_node_conditions`, `get_container_status`, `check_rbac`
- **GKE write tools**: `kubectl_scale`, `kubectl_restart`, `kubectl_label`, `kubectl_annotate`
- **GKE exec tools**: `exec_command` (disabled by default — requires `gke.exec_enabled: true`)
- **GCloud tools**: `gcloud_logging_query`, `gcloud_monitoring_query`

Total: **15 GKE tools + 2 GCloud tools = 17 tools**.

Activated via `vaig live` or `vaig ask --live`.

The InfraAgent also uses `ToolLoopMixin` and can autonomously chain multiple tool calls to investigate infrastructure issues.

```bash
# Live infrastructure investigation
vaig live "Why is payment-service returning 503s?" \
  --cluster prod --namespace default
```

### Orchestrator

Coordinates multi-agent execution. Takes a list of agents defined by a skill and runs them according to a strategy.

**Orchestration Strategies:**

| Strategy | Description |
|----------|-------------|
| `sequential` | Run agents one after another, passing output as context |
| `fan-out` | Run all agents in parallel, then merge results |
| `lead-delegate` | Lead agent decides which specialists to invoke |

```python
# The Orchestrator returns structured results
@dataclass
class OrchestratorResult:
    final_response: str
    agent_results: list[AgentResult]
    strategy: str
    total_tokens: int
```

The default strategy is `sequential` — Agent 1's output becomes context for Agent 2, and so on.

### ChunkedProcessor

Handles files that exceed the model's context window using a Map-Reduce approach:

1. **Chunk** — Split the input into overlapping chunks
2. **Map** — Process each chunk independently with a specialist agent
3. **Reduce** — Merge all chunk results into a final analysis

Configuration:

```yaml
chunking:
  chunk_overlap_ratio: 0.1      # 10% overlap between chunks
  token_safety_margin: 0.1      # Reserve 10% of context for prompt
  chars_per_token: 2.0          # Character-to-token ratio estimate
  inter_chunk_delay: 2.0        # Seconds between chunk API calls
```

### ToolLoopMixin

Shared mixin used by `CodingAgent` and `InfraAgent`. Provides:
- Autonomous tool-use loop with configurable max iterations
- Tool call parsing and execution
- Result injection back into the conversation
- Error handling and retry logic

## Agent Roles

Agents are tagged with a role via `AgentRole`:

| Role | Description |
|------|-------------|
| `orchestrator` | Coordinates other agents |
| `specialist` | Domain expert (analysis, planning) |
| `assistant` | General-purpose helper |
| `coder` | File and code operations |
| `sre` | Infrastructure and operations |

## How Skills Use Agents

Each skill defines a list of 2-3 agents in `get_agents_config()`. For example, the **RCA** skill defines:

```
1. evidence_collector → Gathers and correlates evidence
2. hypothesis_builder → Generates hypotheses using 5 Whys
3. rca_synthesizer    → Produces the final RCA report
```

These are run sequentially by the Orchestrator:

```
[User Input] → evidence_collector → hypothesis_builder → rca_synthesizer → [Final Report]
```

## Configuration

```yaml
agents:
  max_concurrent: 3              # Max parallel agents (for fan-out)
  orchestrator_model: gemini-2.5-pro
  specialist_model: gemini-2.5-flash
```

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
| `response_schema` | `type[BaseModel] \| None` | `None` | Pydantic model for JSON schema output |
| `response_mime_type` | `str \| None` | `None` | Set to `"application/json"` with `response_schema` |

When `response_schema` is set, `SpecialistAgent` passes it to `GeminiClient.generate()` via `types.GenerateContentConfig`. Gemini constrains its output to match the schema and returns a JSON string. The skill's `post_process_report()` method is responsible for converting the JSON to a display format.

## Viewing Agent Config

In the REPL, use `/agents` to see the current skill's agent pipeline:

```
/agents

Agents for skill: rca (sequential)
  1. evidence_collector (specialist) — Evidence Collector
  2. hypothesis_builder (specialist) — Hypothesis Builder
  3. rca_synthesizer (specialist) — RCA Synthesizer
```

---

[Back to index](README.md)
