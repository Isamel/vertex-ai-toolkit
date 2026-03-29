# Pending Improvements — Post v0.10.0

Status as of v0.10.0: 207 Python files, 142 tests, 31 skills (8 with live tools),
prompts split into 9 modules, Rich CLI overhaul, Argo Rollouts full support.

Items are grouped by category with detailed implementation specs.
Each item shows: what to do, which files to change, and implementation notes.

---

## Previously Identified → Now RESOLVED

These were identified in earlier reviews and have been implemented:

- ✅ **Prompts.py monolith split** → now 9 files in `prompts/` directory (2,891 lines total)
- ✅ **prompts split**: `_system.py`, `_gatherer.py`, `_analyzer.py`, `_verifier.py`, `_reporter.py`, `_sub_gatherers.py`, `_phases.py`, `_shared.py`
- ✅ **8 skills with live tools** (was 1): service_health, rca, log_analysis, perf_analysis, config_audit, compliance_check, code_migration, greenfield
- ✅ **Datadog APM metric templates**: `trace.http.request.hits`, `.errors`, `.duration`, `error_rate`, `apdex` — added alongside infra metrics
- ✅ **Datadog SSL proxy support** (`ssl_verify` config) — PR #121
- ✅ **Secret redaction** in kubectl_get secrets — `_redact_secret_item()` in `_formatters.py`
- ✅ **Context window monitoring** — `_monitor_context_window()` in `mixins.py`
- ✅ **Dynamic tool selection** — category-based per-agent filtering (PR #87)
- ✅ **Thought signatures** — capture/replay across function calling turns (PR #97)
- ✅ **Cloud Monitoring metrics** — `get_pod_metrics` tool (PR #101)
- ✅ **Dependency mapping** — `discover_dependencies` tool (PR #102)
- ✅ **Argo Rollouts** — full support with CRD probe, ownership chain, analysis templates (PRs #90, #96, #117)
- ✅ **Severity levels aligned** — Analyzer and Reporter both use CRITICAL/HIGH/MEDIUM/LOW/INFO
- ✅ **COT instruction** — positioned before output format, no visible `<thinking>` blocks
- ✅ **Verifier CRITICAL OUTPUT REQUIREMENT** — moved to top of prompt
- ✅ **kubectl_get_labels reference removed from analyzer** (analyzer has no tools)
- ✅ **Export module** — `DataExporter` with BQ + GCS, wired to `vaig cloud export` and `auto_export_report()` in live pipeline
- ✅ **Rich CLI overhaul** — real-time tree, log-only noise (PR #139, #140)
- ✅ **Programmatic kubectl_top pre-call** — reliable metrics (PR #132)
- ✅ **Vertex AI 429 retry** — two-layer retry for RESOURCE_EXHAUSTED (PR #135)
- ✅ **Unknown tool fuzzy matching** — helpful message when model calls wrong tool name (PR #127)
- ✅ **Code migration skill** — 5-phase state machine with idiom mappings (PR #110)
- ✅ **3-agent coding pipeline** + GreenfieldSkill (PR #111)

---

## STILL PENDING — Detailed Implementation Specs

### 1. Datadog: Cluster Name Override + Fallback Metric Strategy

**Problem**: `query_datadog_metrics` uses `cluster_name` from GKE config as the
Datadog tag value. But the Datadog Agent may register the cluster with a different
name (e.g. GKE says `gke_project_zone_cluster`, Datadog says `prod-cluster`).
Also, when infra metrics (`kubernetes.cpu.usage.total`) return no data, there's
no automatic fallback to APM metrics (`trace.http.request.hits`).

**Files to modify**:
- `src/vaig/core/config.py` — add `cluster_name_override: str = ""` to `DatadogAPIConfig`
- `src/vaig/tools/gke/datadog_api.py` — in `query_datadog_metrics()`, use
  `config.cluster_name_override or cluster_name` for the tag filter
- `src/vaig/skills/service_health/prompts/_sub_gatherers.py` — in the Datadog
  Step 12 instructions, add fallback guidance:
  "If metric=cpu returns 'No data', retry with metric=requests (APM trace metric).
  If metric=memory returns 'No data', retry with metric=latency."

**Config example**:
```yaml
datadog:
  cluster_name_override: "prod-cluster"  # Name as it appears in Datadog tags
```

**Effort**: Low

---

### 2. ArgoCD Management Cluster Support (API + Separate Context)

**Problem**: ArgoCD `_get_argocd_client()` raises `NotImplementedError` for both
API server mode and separate kubeconfig context mode. Only same-cluster mode works.

**Files to modify**:
- `src/vaig/tools/gke/argocd.py` — implement the two stubs:

**Mode 1 — API Server** (lines 414-415):
```python
if server and token:
    # Use httpx/requests to call ArgoCD REST API
    # GET /api/v1/applications → list
    # GET /api/v1/applications/{name} → status
    # GET /api/v1/applications/{name}/resource-tree → managed resources
    # Headers: {"Authorization": f"Bearer {token}"}
    # Return a wrapper that translates REST responses to same dict format as CRD reads
```

**Mode 2 — Separate Context** (lines 419-420):
```python
if context:
    # Load kubeconfig with specific context
    from kubernetes import config as k8s_config, client as k8s_client
    k8s_config.load_kube_config(context=context)
    return ("context", k8s_client.CustomObjectsApi())
```

**Mode 2 is simpler** — just load kubeconfig with a different context. Start with that.

**Effort**: Medium (Mode 2: Low, Mode 1: Medium)

---

### 3. Discovery Mode — `vaig discover`

**Problem**: No autonomous scanning mode. User must know what to ask.

**Files to create**:
- `src/vaig/cli/commands/discover.py` — CLI command
- `src/vaig/agents/discovery.py` — Discovery agent logic

**Implementation**:

```
vaig discover --namespace production
vaig discover --all-namespaces
vaig discover --namespace production --skip-healthy
```

**Phase 1: Inventory** (1 LLM call + N tool calls):
- `kubectl_get("namespaces")` → list namespaces
- For target namespace(s): `kubectl_get("deployments")`, `kubectl_get("statefulsets")`
- Build workload inventory with basic status

**Phase 2: Triage** (1 LLM call):
- LLM classifies each workload: 🟢 Healthy (skip), 🟡 Needs investigation, 🔴 Failing
- This is the key optimization: only failing/degraded services get the full pipeline

**Phase 3: Deep Investigation** (per failing service, in parallel):
- Run service-health skill pipeline scoped to each failing service
- Use fan-out for independent services

**Phase 4: Cluster Report**:
- Aggregate per-service reports into cluster-level summary
- Cross-service correlations

**Register in CLI**:
```python
# src/vaig/cli/commands/discover.py
@app.command()
def discover(
    namespace: str = typer.Option("", help="Target namespace (empty = all non-system)"),
    all_namespaces: bool = typer.Option(False, help="Scan all namespaces"),
    skip_healthy: bool = typer.Option(False, help="Only show problematic services"),
    ...
):
```

**Effort**: High

---

### 4. Chat Mode `/live` — REPL Parity with Live Mode

**Problem**: REPL's `/skill service-health` uses `execute_skill_phase()` (no tools).
Live mode uses `execute_with_tools()` (full pipeline).

**Files to modify**:
- `src/vaig/cli/repl.py` — add `/live` command handler + REPLState fields

**New state fields**:
```python
class REPLState:
    live_mode: bool = False
    tool_registry: ToolRegistry | None = None
    gke_config: GKEConfig | None = None
    tool_result_cache: ToolResultCache | None = None
```

**`/live` handler**:
- Build GKE config from settings
- Register live tools (same as `_register_live_tools()` in live.py)
- Detect Autopilot
- Set `state.live_mode = True`

**Modified chat routing**:
```python
if state.live_mode and state.active_skill:
    # Route through execute_with_tools (full pipeline)
    _handle_live_skill_chat(state, user_input, context_str)
elif state.live_mode:
    # Route through InfraAgent (single agent with tools)
    _handle_live_direct_chat(state, user_input, context_str)
```

**Effort**: Medium

---

### 5. Router Agent — Dynamic Pipeline

**Problem**: Pipeline is static: always gatherer→analyzer→verifier→reporter.
Simple queries ("is the cluster healthy?") don't need analyzer+verifier.

**Files to create**:
- `src/vaig/agents/router.py` — RouterAgent + ExecutionPlan schema

**Design**:
```python
class ExecutionStep(BaseModel):
    agent_name: str
    depends_on: list[str] = []
    condition: str = ""          # "only if findings exist"
    parallel_group: str = ""     # steps with same group run concurrently

class ExecutionPlan(BaseModel):
    steps: list[ExecutionStep]
    reasoning: str

class RouterAgent(BaseAgent):
    def plan(self, query: str, available_agents: list[dict]) -> ExecutionPlan:
        """LLM decides which agents to run and in what order."""
```

**Example plans**:
- "is cluster healthy?" → [gatherer, reporter] (skip analyzer+verifier)
- "why pods crashing?" → [gatherer, analyzer, verifier, reporter] (full)
- "compare with yesterday" → [gatherer, diff_agent, reporter]

**Integration**: New method `Orchestrator.execute_dynamic()` that uses RouterAgent.

**Effort**: High

---

### 6. Vertex AI RAG Engine Integration

**Problem**: Export module exists (DataExporter with BQ+GCS) and is wired to
auto_export in live pipeline. But no RAG corpus or retrieval integration.

**Files to create**:
- `src/vaig/core/rag.py` — RAG Engine client wrapper

**Implementation**:
```python
class RAGKnowledgeBase:
    def __init__(self, config: ExportConfig):
        self._corpus_name = config.rag_corpus_name

    def create_corpus(self) -> str:
        """Create a Vertex AI RAG Engine corpus."""
        from vertexai.preview import rag
        corpus = rag.create_corpus(display_name="vaig-knowledge")
        return corpus.name

    def ingest_reports(self, gcs_paths: list[str]) -> int:
        """Import health reports from GCS into the RAG corpus."""
        from vertexai.preview import rag
        rag.import_files(
            corpus_name=self._corpus_name,
            paths=gcs_paths,
            chunk_size=1024,
            chunk_overlap=200,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant past report chunks for a query."""
        from vertexai.preview import rag
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=self._corpus_name)],
            text=query,
            similarity_top_k=top_k,
        )
        return [chunk.text for chunk in response.contexts]
```

**Integration point**: In the orchestrator, before building the gatherer context,
call `rag.retrieve(query)` to get relevant past reports and inject them as
"Historical context" in the analyzer's input.

**Config**:
```yaml
export:
  rag_corpus_name: "projects/my-project/locations/us-central1/ragCorpora/my-corpus"
```

**Effort**: High

---

### 7. `vaig feedback` — Quality Feedback Collection

**Problem**: No way for users to rate report quality. Needed for RAG quality
improvement and future fine-tuning.

**Files to create**:
- `src/vaig/cli/commands/feedback.py` — CLI command

**Implementation**:
```bash
vaig feedback --run-id <id> --rating 4 --comment "Root cause was correct"
vaig feedback --last --rating 3 --comment "Missing HPA analysis"
```

The command writes to the `feedback` BigQuery table (schema already exists in
`DataExporter`).

**Also add**: Prompt at end of live mode execution:
```
📊 Was this analysis helpful? Rate 1-5: [press Enter to skip]
```

**Effort**: Medium

---

### 8. `vaig doctor` — Healthcheck Command

**Problem**: No way to verify that auth, APIs, GKE connectivity, and optional
deps are all working before running a diagnosis.

**Files to create**:
- `src/vaig/cli/commands/doctor.py`

**Checks**:
```
vaig doctor

✓ GCP Authentication      — ADC valid, project: my-project
✓ Vertex AI API           — gemini-2.5-flash accessible
✓ GKE Connectivity        — cluster: prod-cluster (Autopilot)
✓ Cloud Logging           — API enabled, can query
✓ Cloud Monitoring         — API enabled, can query
⚠ Helm Integration        — enabled, 3 releases found
✗ ArgoCD Integration      — disabled (set argocd.enabled=true)
⚠ Datadog Integration     — enabled, but SSL error (check ssl_verify config)
✓ Optional deps           — kubernetes, google-cloud-logging, datadog-api-client
✗ MCP Servers             — disabled
```

**Effort**: Medium

---

### 9. Watch Mode Diff — Show Only Changes

**Problem**: `--watch` re-renders the full report every N seconds. For a 15-minute
watch session, the SRE re-reads the same report 10+ times.

**Implementation**:
- After each watch iteration, compare the new `HealthReport` JSON with the previous one
- Show diff: "+1 NEW finding", "1 RESOLVED", "2 UNCHANGED", "1 SEVERITY CHANGED (MEDIUM→CRITICAL)"
- Only print the full report on the first iteration; subsequent iterations show the diff summary

**Files to modify**:
- `src/vaig/cli/commands/live.py` — in `_run_watch_loop`, store previous `HealthReport`
- `src/vaig/skills/service_health/schema.py` — add `diff(other: HealthReport)` method

```python
def diff(self, previous: HealthReport) -> ReportDiff:
    prev_ids = {f.id for f in previous.findings}
    curr_ids = {f.id for f in self.findings}
    return ReportDiff(
        new_findings=[f for f in self.findings if f.id not in prev_ids],
        resolved_findings=[f for f in previous.findings if f.id not in curr_ids],
        unchanged=[f for f in self.findings if f.id in prev_ids],
        severity_changes=[...],
    )
```

**Effort**: Medium

---

### 10. Speculative Tool Execution

**Problem**: In the tool loop, the model proposes ONE tool call, waits for result,
then proposes next. Each round-trip adds ~1-2s latency.

**Files to create**:
- `src/vaig/agents/speculative.py`

**Implementation**:
```python
class SpeculativeExecutor:
    def predict_next_tools(self, completed: list[str], checklist: list[str]) -> list[tuple[str, dict]]:
        """Rule-based prediction of likely next tool calls."""
        # If Step 1 done and Step 2 not done → predict pods, deployments
        # If events found FailedCreate → predict RS describe + deployment YAML
```

Pre-executed results go into `ToolResultCache`. When the model actually requests
them, the response is instant (cache hit).

**Effort**: High

---

### 11. Adaptive Prompt Tuning from Past Reports

**Problem**: No system to analyze past report quality and suggest prompt improvements.

**Files to create**:
- `src/vaig/core/prompt_tuner.py`

**Implementation**:
```python
class PromptTuner:
    def analyze_quality(self, reports: list[HealthReport]) -> QualityInsights:
        """Score reports on: hallucination rate, evidence depth, actionability."""
        # Empty evidence lists → hallucination
        # CRITICAL status + resource-level scope → over-escalation
        # Recommendations without commands → low actionability
        # Empty timeline → incomplete

    def suggest_changes(self, insights: QualityInsights) -> list[str]:
        """Generate prompt refinement suggestions."""
```

**Trigger**: `vaig optimize --reports --last 20`

**Effort**: High

---

### 12. Tool Call Optimizer

**Problem**: No analysis of tool call efficiency across runs.

**Files to create**:
- `src/vaig/core/optimizer.py`

**Implementation**:
```python
class ToolCallOptimizer:
    def analyze(self, records: list[ToolCallRecord]) -> ToolInsights:
        """Detect: always-failing tools, redundant calls, slow tools."""

    def suggest(self, insights: ToolInsights) -> list[str]:
        """E.g. 'kubectl_top nodes fails 100% on Autopilot — verify detection'"""
```

**Trigger**: `vaig optimize --last 50`

**Effort**: Medium

---

### 13. Agents-as-Tools Pattern

**Problem**: An agent cannot dynamically invoke another agent during execution.
The gatherer should be able to call a `mesh_specialist` if it discovers Istio issues.

**Files to create**:
- `src/vaig/agents/agent_tool.py`

```python
def agent_as_tool(agent: BaseAgent) -> ToolDef:
    """Wrap a BaseAgent as a ToolDef for dynamic invocation."""
    return ToolDef(
        name=f"invoke_{agent.name}",
        description=f"Invoke {agent.role} specialist",
        parameters=[
            ToolParam(name="question", type="string", description="What to analyze"),
        ],
        execute=lambda question: agent.execute(question).content,
    )
```

**Effort**: Medium

---

### 14. Shared Pipeline State

**Problem**: Agents communicate via text. Structured data (findings list,
affected resources, management context) gets lost between agents.

**Files to create**:
- `src/vaig/agents/pipeline_state.py`

```python
@dataclass
class PipelineState:
    findings: list[dict] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    management_context: dict[str, str] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Serialize for injection into agent prompts."""
```

**Integration**: Orchestrator passes state to each agent. Agents read/write structured data
instead of parsing text.

**Effort**: Medium

---

### 15. DATABASE_URL Pattern Redaction in Helm Values

**Problem**: `_redact_sensitive_values` in helm.py only matches key names
(password, token, secret). Values like `DATABASE_URL=postgres://user:pass@host`
are not redacted because the key is "url" not "password".

**Files to modify**:
- `src/vaig/tools/gke/helm.py` — add value-pattern matching:

```python
_SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"://[^:]+:[^@]+@"),  # credentials in URLs
    re.compile(r"-----BEGIN .* KEY-----"),  # PEM keys
    re.compile(r"^eyJ"),  # JWT tokens (base64 JSON)
]

def _is_sensitive_value(value: str) -> bool:
    return any(p.search(str(value)) for p in _SENSITIVE_VALUE_PATTERNS)
```

**Effort**: Low

---

### 16. Multiline Input in REPL

**Problem**: Cannot paste multi-line content (stack traces, YAML) in REPL.

**Files to modify**:
- `src/vaig/cli/repl.py` — use prompt_toolkit's multiline support

```python
from prompt_toolkit import PromptSession
session = PromptSession(multiline=True)  # Alt+Enter to submit
```

Or use a delimiter-based approach: `"""` to start/end multiline input.

**Effort**: Low

---

### 17. PRIORITY HIERARCHY Deduplication

**Problem**: The 5 "Kubernetes vs Datadog" priority rules appear in 2 places
in the gatherer prompts (shared section and Datadog step).

**Files to modify**:
- `src/vaig/skills/service_health/prompts/_shared.py` — extract to constant
- `src/vaig/skills/service_health/prompts/_gatherer.py` — reference the constant
- `src/vaig/skills/service_health/prompts/_sub_gatherers.py` — reference the constant

**Effort**: Trivial

---

### 18. Anti-Hallucination Rules Consolidation

**Problem**: Rules exist in 3 places: `prompt_defense.py` (6 rules),
`_system.py` (2 more), analyzer `_analyzer.py` (2 duplicates in STRICT rules).

**Files to modify**:
- `src/vaig/skills/service_health/prompts/_analyzer.py` — remove rules 2 and 4
  from STRICT Analysis Rules (they duplicate the system instruction rules).
  Add comment: "(covered by Anti-Hallucination Rules in system instruction)"

**Effort**: Trivial

---

## Implementation Priority

| Phase | Items | Effort | Timeline | Status |
|-------|-------|--------|----------|--------|
| **Phase 1** (quick wins) | #1, #15, #16, #17, #18 | Low-Trivial | 1-2 days | ✅ DONE (PR #138-#153) |
| **Phase 2** (core features) | #4, #7, #8, #9 | Medium | 1 week | ✅ DONE (#4: PR #154, #7: PR #155, #8/#9: pre-existing) |
| **Phase 3** (infrastructure) | #2, #3, #6, #12 | Medium-High | 2 weeks | ✅ DONE (#2/#3: pre-existing, #12: PR #156, #6: PR #157) |
| **Phase 4** (advanced) | #5, #10, #11, #13, #14 | High | 3-4 weeks | ✅ DONE (#11: PR #160, #13/#14: PR #158, #5/#10: deferred to future) |
