# VAIG v1.0 — Next-Generation Architecture Specs

5 themes for the next major evolution of the toolkit.

---

## Theme 1: Chat Mode Parity with Live Mode

### Problem

Chat mode (`vaig chat`) and Live mode (`vaig live`) have fundamentally different
capabilities. Live mode has infrastructure tools, multi-agent pipelines, structured
output, tool logging, cost tracking, and report export. Chat mode has none of these
when a user activates a live skill via `/skill service-health`.

### Gap Analysis

| Capability | `vaig live` | `vaig chat` + `/skill service-health` |
|---|---|---|
| GKE/GCloud tools (kubectl, logs, monitoring) | ✅ Full registry | ❌ Not available — uses `execute_skill_phase` which has NO tool registry |
| Multi-agent pipeline (gatherer→analyzer→verifier→reporter) | ✅ `execute_with_tools` | ❌ Uses `execute_skill_phase` → single-agent per phase |
| Structured JSON output (HealthReport schema) | ✅ Reporter uses `response_schema` | ❌ Raw Markdown from `execute_skill_phase` |
| Tool call logging (ToolCallLogger) | ✅ Live tool-by-tool progress | ❌ No progress feedback |
| Tool result cache (dedup across agents) | ✅ `ToolResultCache` | ❌ Not wired |
| Tool call persistence (JSONL) | ✅ `ToolCallStore` | ❌ Not wired |
| Severity coloring in output | ✅ `print_colored_report` | ❌ Plain Markdown |
| Autopilot detection | ✅ Injected into agent prompts | ❌ Not detected |
| `--dry-run` / `--watch` | ✅ Available | ❌ Not applicable (interactive) |
| Session persistence of results | ✅ (via export) | ✅ Session history |

### Root Cause

The REPL's `_handle_skill_chat` calls `orchestrator.execute_skill_phase()` which:
1. Creates agents WITHOUT a tool registry
2. Does NOT use `execute_with_tools()` pipeline
3. Does NOT wire ToolCallStore, ToolResultCache, or Autopilot detection

### Solution: `/live` Slash Command in REPL

Add a new mode to the REPL: `/live` that activates infrastructure tools within
the chat session. When active, messages route through `execute_with_tools()`
instead of `execute_skill_phase()`.

#### New REPL state

```python
class REPLState:
    # ... existing fields ...
    live_mode: bool = False
    tool_registry: ToolRegistry | None = None
    gke_config: GKEConfig | None = None
    tool_call_store: ToolCallStore | None = None
    tool_result_cache: ToolResultCache | None = None
    is_autopilot: bool | None = None
```

#### `/live` command handler

```python
def _cmd_live(state: REPLState, args: str) -> None:
    """Toggle live infrastructure mode in the REPL."""
    if state.live_mode:
        # Deactivate
        state.live_mode = False
        state.tool_registry = None
        console.print("[green]✓ Live mode OFF — back to normal chat[/green]")
        return

    # Activate — build GKE config and tool registry
    from vaig.cli.commands.live import _build_gke_config, _register_live_tools

    gke_config = _build_gke_config(state.settings)
    registry = _register_live_tools(gke_config, settings=state.settings)
    tool_count = len(registry.list_tools())

    if tool_count == 0:
        console.print("[red]No infrastructure tools available. Install: pip install vertex-ai-toolkit[live][/red]")
        return

    state.live_mode = True
    state.gke_config = gke_config
    state.tool_registry = registry
    state.tool_result_cache = ToolResultCache()

    # Detect Autopilot
    try:
        from vaig.tools.gke_tools import detect_autopilot
        state.is_autopilot = detect_autopilot(gke_config)
    except ImportError:
        pass

    console.print(Panel.fit(
        f"[bold green]🔍 Live Mode ON[/bold green]\n"
        f"[dim]Cluster: {gke_config.cluster_name or '(default)'}[/dim]\n"
        f"[dim]{tool_count} tools loaded[/dim]",
        border_style="green",
    ))
```

#### Modified `_handle_chat` routing

```python
if state.live_mode and state.active_skill:
    # Route through execute_with_tools (same as vaig live)
    _handle_live_skill_chat(state, user_input, context_str)
elif state.live_mode:
    # Route through InfraAgent (single agent with tools)
    _handle_live_direct_chat(state, user_input, context_str)
elif state.code_mode:
    _handle_code_chat(state, user_input, context_str)
# ... rest unchanged
```

#### Files to modify

- `src/vaig/cli/repl.py` — add `/live` command, `_handle_live_skill_chat`, `_handle_live_direct_chat`
- `src/vaig/cli/repl.py` — update `REPLState`, `_handle_command`, `prompt_prefix`

#### Effort: Medium
#### Impact: 🔴 High — chat mode becomes as powerful as live mode

---

## Theme 2: Autonomous Discovery Mode

### Concept

A mode where the user provides only a cluster (or namespace), and VAIG autonomously
scans every service looking for problems — without the user having to know what to ask.

```bash
# One-shot
vaig discover --namespace production
vaig discover --all-namespaces

# In REPL
/discover production
```

### How It Works

#### Phase 1: Service Inventory (1 LLM call + N tool calls)

The discovery agent calls:
1. `kubectl_get("namespaces")` → list all namespaces
2. For each namespace (or the specified one):
   - `kubectl_get("deployments", namespace=ns)`
   - `kubectl_get("statefulsets", namespace=ns)`
   - `kubectl_get("daemonsets", namespace=ns)`
3. Build an inventory of all workloads with their status

#### Phase 2: Triage (1 LLM call)

The LLM receives the inventory and classifies each workload:
- 🟢 **Healthy** — all replicas ready, no restarts → skip
- 🟡 **Needs investigation** — some restarts, degraded replicas → shallow check
- 🔴 **Failing** — CrashLoopBackOff, 0 ready, events → deep investigation

This is the key optimization: instead of running the full 4-agent service-health
pipeline on every service, the triage step filters to only the ones that need it.

#### Phase 3: Deep Investigation (per failing service)

For each 🔴 and 🟡 service, run the full service-health skill pipeline
(gatherer→analyzer→verifier→reporter) scoped to that specific namespace/deployment.

These run **in parallel** (fan-out) for independent services.

#### Phase 4: Cluster-Level Report

Aggregate all per-service reports into a single cluster-level summary:
- Executive summary with total healthy/degraded/critical counts
- Per-namespace breakdown
- Cross-service correlations (e.g., "3 services in namespace X all failing
  with the same image pull error → likely a registry issue")
- Priority-ordered action list across all findings

### Implementation

#### New file: `src/vaig/agents/discovery.py`

```python
class DiscoveryAgent:
    """Autonomous cluster scanner that finds problems without being asked."""

    def __init__(self, client, settings, gke_config, tool_registry):
        self._client = client
        self._settings = settings
        self._gke_config = gke_config
        self._registry = tool_registry

    async def discover(
        self,
        namespaces: list[str] | None = None,
        *,
        on_progress: Callable | None = None,
    ) -> DiscoveryReport:
        """Run autonomous discovery across the cluster.

        1. Inventory all workloads
        2. Triage (LLM classifies health)
        3. Deep investigation (parallel, only for failing services)
        4. Aggregate into cluster-level report
        """
```

#### New file: `src/vaig/agents/discovery_schema.py`

Pydantic models for:
- `WorkloadInventory` — list of workloads with basic status
- `TriageResult` — classification per workload (healthy/needs-investigation/failing)
- `DiscoveryReport` — aggregated cluster report with per-service findings

#### New CLI command: `src/vaig/cli/commands/discover.py`

```bash
vaig discover --namespace production --project my-project
vaig discover --all-namespaces --format json -o cluster-report.json
vaig discover --namespace production --skip-healthy  # only show issues
```

#### REPL integration: `/discover` command

```python
def _cmd_discover(state: REPLState, args: str) -> None:
    """Run autonomous service discovery."""
    if not state.live_mode:
        console.print("[yellow]Enable live mode first: /live[/yellow]")
        return
    # ... run discovery agent
```

#### Optimization: Skip healthy services

The triage step is what makes discovery fast. Without it, scanning 50 services
would require 50 × 4-agent pipelines = 200 agent executions. With triage:
- 50 services inventoried (1 tool call each = 50 tool calls, ~15s)
- LLM triages: 45 healthy, 3 degraded, 2 failing (1 LLM call, ~2s)
- Only 5 services get deep investigation (5 × 4 agents = 20 agent executions)
- Failing services investigated in parallel (fan-out)

Total: ~3-5 minutes instead of ~2 hours.

#### Files to create/modify
- New: `src/vaig/agents/discovery.py`
- New: `src/vaig/agents/discovery_schema.py`
- New: `src/vaig/cli/commands/discover.py`
- Modify: `src/vaig/cli/repl.py` — add `/discover` command
- Modify: `src/vaig/cli/app.py` — register discover command

#### Effort: High
#### Impact: 🔴 Transformational — "scan my cluster for problems" in one command

---

## Theme 3: Dynamic Multi-Agent Orchestration

### 3.1 LLM-as-Router (Adaptive Pipeline)

#### Concept

Replace the hardcoded agent pipeline with a `RouterAgent` that uses the LLM to
create a dynamic execution plan based on the query.

#### New file: `src/vaig/agents/router.py`

```python
class ExecutionStep(BaseModel):
    agent_name: str
    depends_on: list[str] = []
    condition: str = ""        # e.g. "only if findings with confidence < CONFIRMED"
    parallel_group: str = ""   # steps with same group run concurrently

class ExecutionPlan(BaseModel):
    steps: list[ExecutionStep]
    reasoning: str
    skip_verifier: bool = False  # router can skip verifier if unnecessary

class RouterAgent(BaseAgent):
    """LLM-powered agent that creates dynamic execution plans."""

    ROUTER_PROMPT = """You are an execution planner for an SRE diagnostic system.
    Given a user's infrastructure query and the available agents, create an
    optimal execution plan.

    Available agents:
    {agents_descriptions}

    Rules:
    - Simple queries ("is the cluster healthy?") → gatherer + reporter (skip analyzer/verifier)
    - Complex queries ("why are pods crashing?") → full pipeline
    - If query mentions a specific resource → scope gatherer to that resource only
    - Verifier is optional — skip if gatherer already produces CONFIRMED evidence
    - Agents in the same parallel_group run concurrently
    """

    def plan(self, query: str, available_agents: list[dict]) -> ExecutionPlan:
        """Ask the LLM to create an execution plan (structured output)."""
```

#### New file: `src/vaig/agents/plan_executor.py`

```python
class PlanExecutor:
    """Executes an ExecutionPlan respecting dependencies and conditions."""

    async def execute(self, plan: ExecutionPlan, agents: dict[str, BaseAgent],
                      context: str) -> OrchestratorResult:
        """Execute plan steps respecting dependency graph.

        - Steps with no dependencies start immediately
        - Steps with dependencies wait for their prerequisites
        - Steps with conditions are evaluated before execution
        - Steps in same parallel_group run concurrently
        """
```

#### Modified: `src/vaig/agents/orchestrator.py`

New method `execute_dynamic()`:

```python
async def execute_dynamic(self, query, skill, tool_registry, **kwargs):
    """Dynamic pipeline: router plans → executor runs."""

    # 1. Router creates plan
    router = RouterAgent(self._client, self._settings)
    plan = router.plan(query, skill.get_agents_config())

    # 2. Create only the agents needed by the plan
    needed_agents = {step.agent_name for step in plan.steps}
    agents = self.create_agents_for_skill(skill, tool_registry,
                                           filter_names=needed_agents)

    # 3. Execute plan
    executor = PlanExecutor()
    return await executor.execute(plan, agents, context=query)
```

#### Effort: High
#### Impact: 🔴 Reduces unnecessary agent executions by 30-50%

### 3.2 Agents as Tools (AgentTool Pattern)

#### Concept

Any agent can invoke another agent as a tool during its execution.

#### New file: `src/vaig/agents/agent_tool.py`

```python
def agent_as_tool(agent: BaseAgent) -> ToolDef:
    """Wrap a BaseAgent as a ToolDef for dynamic invocation.

    The wrapped agent can be called by other agents during their tool-use loop.
    Usage: gatherer discovers Istio issues → invokes mesh_specialist dynamically.
    """
    return ToolDef(
        name=f"invoke_{agent.name}",
        description=f"Invoke the {agent.role} specialist: {agent.config.system_instruction[:200]}",
        parameters=[
            ToolParam(name="question", type="string",
                      description="What to analyze or investigate"),
            ToolParam(name="context", type="string",
                      description="Relevant data to pass to the specialist",
                      required=False),
        ],
        execute=lambda question, context="": agent.execute(question, context=context).content,
    )
```

#### Effort: Medium
#### Impact: 🟠 High — agents dynamically call specialists as needed

### 3.3 Shared Pipeline State

#### New file: `src/vaig/agents/pipeline_state.py`

```python
@dataclass
class PipelineState:
    """Shared mutable state across all agents in a pipeline execution."""

    findings: list[dict] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    management_context: dict[str, str] = field(default_factory=dict)
    tool_results_summary: dict[str, str] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Serialize state for injection into agent prompts."""
```

Agents read/write state during execution. The orchestrator passes state to each
agent. Reduces data loss between agents (currently, only text is passed).

#### Effort: Medium
#### Impact: 🟡 Medium — structured inter-agent communication

---

## Theme 4: Execution Speed Improvements

### 4.1 Parallel Gatherer Split

#### Concept

Split the monolithic gatherer (15-25 sequential tool calls, 3-4 min) into
parallel sub-gatherers that run concurrently.

#### Design

```
Current:  1 gatherer → 25 tools sequentially → 3-4 min

Proposed: 4 parallel sub-gatherers → ~30-45s total
  ┌─ node_gatherer    (nodes, conditions, top nodes)       → 15s
  ├─ workload_gatherer (pods, deployments, services, hpa)  → 30s
  ├─ event_gatherer   (warning events, RS events)          → 15s
  └─ logging_gatherer (error logs, warning logs)           → 20s

  merger_agent → combines 4 outputs into unified format    → 5s
```

#### Implementation

New skill configuration in `service_health/skill.py`:

```python
def get_agents_config(self):
    return [
        # Stage 1: parallel fan-out gatherers
        {"name": "node_gatherer", "requires_tools": True, "parallel_group": "gather",
         "system_instruction": NODE_GATHERER_PROMPT, "model": "gemini-2.5-flash"},
        {"name": "workload_gatherer", "requires_tools": True, "parallel_group": "gather",
         "system_instruction": WORKLOAD_GATHERER_PROMPT, "model": "gemini-2.5-flash"},
        {"name": "event_gatherer", "requires_tools": True, "parallel_group": "gather",
         "system_instruction": EVENT_GATHERER_PROMPT, "model": "gemini-2.5-flash"},
        {"name": "logging_gatherer", "requires_tools": True, "parallel_group": "gather",
         "system_instruction": LOGGING_GATHERER_PROMPT, "model": "gemini-2.5-flash"},
        # Stage 2: sequential analysis on merged data
        {"name": "health_analyzer", ...},
        {"name": "health_verifier", ...},
        {"name": "health_reporter", ...},
    ]
```

The orchestrator needs to support `parallel_group` in agent configs — agents with
the same group execute concurrently, their outputs merged before the next stage.

#### Effort: High
#### Impact: 🔴 4-6x speedup on the gatherer phase (the bottleneck)

### 4.2 Speculative Tool Execution

#### Concept

Pre-execute likely next tools while the LLM is thinking.

#### New file: `src/vaig/agents/speculative.py`

```python
class SpeculativeExecutor:
    """Predicts and pre-executes tools the model will likely call next."""

    def predict_next_tools(self, step: int, completed: list[str],
                           checklist: list[str]) -> list[tuple[str, dict]]:
        """Rule-based prediction of next tool calls."""
        predictions = []
        if "get_node_conditions" in completed and "kubectl_get pods" not in completed:
            predictions.append(("kubectl_get", {"resource": "pods", "namespace": ns}))
        if "get_events" in completed and any("FailedCreate" in r for r in results):
            predictions.append(("kubectl_get", {"resource": "replicasets", "namespace": ns}))
        return predictions

    async def pre_execute(self, predictions, tool_registry, cache):
        """Execute predicted tools and store results in cache."""
        tasks = [self._execute_one(name, args, tool_registry, cache)
                 for name, args in predictions]
        await asyncio.gather(*tasks)
```

Integrates with existing `ToolResultCache` — pre-executed results become cache
hits when the model actually requests them.

#### Effort: High
#### Impact: 🔴 40-60% latency reduction in tool loop

### 4.3 Model Tiering

Change gatherer from `gemini-2.5-pro` to `gemini-2.5-flash`:
- Gatherer follows a procedure — doesn't need Pro-level reasoning
- Flash is ~2x faster and ~70% cheaper for tool-calling tasks
- Keep analyzer on Flash with thinking for deep reasoning
- Keep reporter on Flash for structured output

One-line change in `service_health/skill.py`:
```python
{"name": "health_gatherer", "model": "gemini-2.5-flash", ...}  # was gemini-2.5-pro
```

#### Effort: Trivial (1 line)
#### Impact: 🟡 ~30% faster + ~70% cheaper for gatherer

### 4.4 Streaming Report

Show report sections as they're generated instead of waiting for the full report.

Since the reporter now uses JSON schema (can't stream JSON), the approach is:
1. Reporter generates JSON (current behavior)
2. Parse the JSON
3. Stream the Markdown rendering section by section with Rich Live display

```python
async def stream_report(report: HealthReport):
    """Render report sections progressively."""
    with Live(console=console) as live:
        # Executive summary appears first
        live.update(render_executive_summary(report))
        await asyncio.sleep(0.1)
        # Then findings
        live.update(render_findings(report))
        # Then recommendations
        live.update(render_recommendations(report))
```

#### Effort: Medium
#### Impact: 🟡 Better perceived speed (actual speed unchanged)

---

## Theme 5: Self-Improvement via Execution Metrics

### Concept

The system records every tool call, every LLM response, every cost metric, and
every user interaction. Use this data to automatically improve future executions.

### 5.1 Feedback Loop: Tool Call Effectiveness

#### Data source: `ToolCallStore` (JSONL records)

Each record has: tool_name, args, duration, success, output_size, error_type.

#### New file: `src/vaig/core/optimizer.py`

```python
class ToolCallOptimizer:
    """Analyzes historical tool call data to suggest optimizations."""

    def analyze_tool_patterns(self, records: list[ToolCallRecord]) -> ToolInsights:
        """Analyze tool call patterns from JSONL records.

        Returns insights like:
        - Tools that always fail with specific args → suggest removal from prompt
        - Tools that are called but output is never used → wasted cost
        - Tools that take >10s → suggest caching or parallelization
        - Redundant tool calls (same tool+args called twice in one run)
        """

    def suggest_prompt_refinements(self, insights: ToolInsights) -> list[str]:
        """Generate prompt refinement suggestions.

        Example outputs:
        - "get_events with event_type='Normal' was called 15 times across runs
           but the output was never referenced in findings. Consider removing
           Normal events from the gatherer prompt."
        - "kubectl_top(resource_type='nodes') fails 100% on Autopilot clusters.
           The Autopilot injection should prevent this call."
        - "gcloud_logging_query averages 8.2s per call. Pre-caching the last
           hour of error logs would eliminate 60% of these calls."
        """
```

#### CLI command: `vaig optimize`

```bash
vaig optimize --last 50   # analyze last 50 runs
vaig optimize --since 7d  # analyze last 7 days

# Output:
# 📊 Tool Call Analysis (50 runs)
# ┌────────────────────────┬───────┬──────────┬──────────┬────────┐
# │ Tool                   │ Calls │ Avg Time │ Failures │ Waste  │
# ├────────────────────────┼───────┼──────────┼──────────┼────────┤
# │ kubectl_get            │ 312   │ 0.8s     │ 2%       │ 5%     │
# │ gcloud_logging_query   │ 145   │ 8.2s     │ 12%      │ 0%     │
# │ kubectl_top (nodes)    │ 48    │ 0.3s     │ 100%*    │ 100%   │
# └────────────────────────┴───────┴──────────┴──────────┴────────┘
# * kubectl_top nodes fails on Autopilot — Autopilot detection may not be working
#
# 💡 Suggestions:
# 1. gcloud_logging_query has 12% failure rate — check Cloud Logging API quotas
# 2. kubectl_top nodes: 100% failure on Autopilot — verify Autopilot injection
# 3. Average run cost: $0.35 → potential savings with Flash gatherer: ~$0.12
```

#### Effort: Medium
#### Impact: 🟡 Operational visibility + actionable optimization

### 5.2 Adaptive Prompt Tuning

#### Concept

Use the structured report data (HealthReport JSON) from past runs to automatically
detect quality issues and adjust prompts.

#### New file: `src/vaig/core/prompt_tuner.py`

```python
class PromptTuner:
    """Analyzes past report quality and suggests prompt adjustments."""

    def analyze_report_quality(self, reports: list[HealthReport]) -> QualityInsights:
        """Score past reports on quality dimensions.

        Checks:
        - Hallucination rate: findings with empty evidence lists
        - Scope accuracy: CRITICAL status with resource-level scope (over-escalation)
        - Actionability: recommendations without commands
        - Completeness: empty sections (timeline, cluster_overview)
        - Conciseness: finding descriptions > 500 chars
        - Evidence depth: findings with < 2 evidence items
        """

    def suggest_prompt_changes(self, insights: QualityInsights) -> list[PromptPatch]:
        """Generate specific prompt patches.

        Example:
        - If 30% of reports have empty timeline → suggest strengthening
          the "Timeline MANDATORY" rule in reporter prompt
        - If hallucination_rate > 5% → suggest lowering temperature
        - If avg_report_tokens > 20000 → suggest stricter conciseness rule
        """
```

#### Integration with `vaig optimize`

```bash
vaig optimize --reports --last 20

# 📊 Report Quality Analysis (20 reports)
# ┌────────────────────┬──────────┬──────────┐
# │ Metric             │ Score    │ Trend    │
# ├────────────────────┼──────────┼──────────┤
# │ Hallucination rate │ 2%       │ ↓ improving │
# │ Evidence depth     │ 4.2/5    │ → stable │
# │ Actionability      │ 95%      │ ↑ improving │
# │ Scope accuracy     │ 88%      │ → stable │
# │ Avg cost           │ $0.28    │ ↓ cheaper │
# │ Avg time           │ 2.1 min  │ ↓ faster │
# └────────────────────┴──────────┴──────────┘
```

#### Effort: High
#### Impact: 🟡 Continuous quality improvement over time

### 5.3 Runtime Auto-Correction

#### Concept

During execution (not post-hoc), the system monitors its own performance and
makes real-time adjustments.

#### Mechanisms

**A. Dynamic Temperature Adjustment**

If the gatherer produces output that fails validation (missing sections, shallow
content), the retry automatically uses a lower temperature:

```python
# In orchestrator, during gatherer retry:
if validation.needs_retry:
    # First attempt used temp=0.0, retry with even more deterministic settings
    retry_kwargs = {
        "temperature": 0.0,
        "frequency_penalty": 0.5,  # higher than normal to avoid repeating mistakes
    }
```

**B. Tool Call Budget Monitoring**

If an agent has used 80% of its `max_iterations` budget without producing all
required sections, inject a "budget warning" into the next prompt:

```python
if iteration >= max_iterations * 0.8 and not all_sections_covered:
    budget_warning = (
        f"WARNING: You have used {iteration}/{max_iterations} iterations. "
        f"You MUST produce your final output within the next {max_iterations - iteration} iterations. "
        f"Focus on completing the mandatory sections: {missing_sections}"
    )
    # Inject into next LLM call
```

**C. Cost Circuit Breaker**

If a single pipeline execution exceeds a configurable cost threshold, stop
early and return partial results:

```python
if accumulated_cost > settings.budget.max_cost_per_run:
    logger.warning("Cost circuit breaker triggered: $%.2f > $%.2f limit",
                    accumulated_cost, settings.budget.max_cost_per_run)
    # Return partial results from completed agents
    return OrchestratorResult(
        success=False,
        synthesized_output="Analysis stopped: cost limit reached. Partial results below.\n" + partial,
    )
```

**D. Model Fallback on Repeated Failures**

If the primary model fails 2+ times on the same agent (rate limit, timeout),
automatically switch to the fallback model:

```python
if retry_count >= 2 and settings.models.fallback:
    logger.warning("Switching to fallback model: %s", settings.models.fallback)
    agent.switch_model(settings.models.fallback)
```

#### Effort: Medium (A-D are independent, implement incrementally)
#### Impact: 🟠 High — resilience + cost control + quality floor

---

## Implementation Priority

| Phase | Theme | Items | Effort | Timeline |
|---|---|---|---|---|
| **Phase 1** | Speed | 4.3 Model tiering (1 line) | Trivial | Day 1 |
| **Phase 1** | Chat parity | `/live` command in REPL | Medium | Week 1 |
| **Phase 1** | Self-improvement | 5.3A-D Auto-correction (runtime) | Medium | Week 1 |
| **Phase 2** | Dynamic agents | 3.1 Router + PlanExecutor | High | Week 2-3 |
| **Phase 2** | Speed | 4.1 Parallel gatherer split | High | Week 2-3 |
| **Phase 2** | Self-improvement | 5.1 Tool call optimizer + `vaig optimize` | Medium | Week 2 |
| **Phase 3** | Discovery | Full autonomous discovery mode | High | Week 3-4 |
| **Phase 3** | Dynamic agents | 3.2 Agents as Tools | Medium | Week 3 |
| **Phase 3** | Speed | 4.2 Speculative execution | High | Week 4 |
| **Phase 4** | Self-improvement | 5.2 Adaptive prompt tuning | High | Week 4-5 |
| **Phase 4** | Dynamic agents | 3.3 Shared Pipeline State | Medium | Week 4 |
| **Phase 4** | Speed | 4.4 Streaming report display | Medium | Week 5 |
