# Rate-Limit Resilience v1 — Survive Vertex AI 429 `RESOURCE_EXHAUSTED` storms

**Version**: 1.0
**Target**: vaig v0.19+
**Scope**: `vaig.core.client.GeminiClient` + `vaig.agents.orchestrator`
**Status**: Draft for implementation
**Severity of addressed bug**: Critical — runs in `service-health` skill
fail with `GeminiRateLimitError` after ~56 s of retries when the
`gemini-2.5-pro` quota is saturated.

---

## 0 · Context

### 0.1 · Observed failure (production trace, 2026-04-24)

```
parallel_gatherers ████████████ 62 tools (380.2s) ✖
...
WARNING vaig.core.client Rate-limited (genai 429) on attempt 1/4 — retrying in 8.02s
WARNING vaig.core.client Rate-limited (genai 429) on attempt 2/4 — retrying in 16.16s
WARNING vaig.core.client Rate-limited (genai 429) on attempt 3/4 — retrying in 32.36s
ERROR   vaig.agents.mixins ToolLoopMixin API call failed on iteration 4
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED
WARNING vaig.agents.orchestrator Gatherer agent datadog_gatherer failed (non-fatal)
WARNING vaig.agents.orchestrator Merged gatherer output is missing 4 required section(s)
```

Observed facts:

- `max_retries = 3` and `rate_limit_initial_delay = 8.0` mean total
  wall-clock wait before giving up is **8 + 16 + 32 ≈ 56 s**.
- The fan-out lanzó 5 sub-gatherers concurrently against
  `gemini-2.5-pro` in `us-central1`. All five competed for the same
  regional quota at the same instant.
- `fallback_location` defaults to `"us-central1"` — identical to the
  primary location, so the SSL-based fallback in `_reinitialize_with_fallback`
  is a no-op and never triggers on 429 anyway (the fallback code path
  is gated on SSL/connection errors, not quota).
- The skill continued with missing data; `health_analyzer`,
  `health_verifier`, `health_reporter` all completed on partial inputs.
- Enrichment (two-pass recommendation pass) hit 3/4 timeouts — the
  `gemini-2.5-pro` quota was still saturated minutes later.

### 0.2 · Design goal

A transient quota spike in one region should **never** take the run
down. The client must:

1. Wait long enough to outlast typical Vertex AI quota windows.
2. Reduce its own pressure on the quota when it sees 429s.
3. Gracefully degrade to a cheaper model before failing the agent.
4. Tell the user what happened, cleanly.

### 0.3 · Invariants

| # | Invariant |
|---|---|
| RL-1 | Exponential backoff with jitter on every 429, capped by a total wall-clock budget per call (not just per retry). |
| RL-2 | Never overrun the per-run cost budget while waiting out a 429. `max_total_wait_s` is respected. |
| RL-3 | Parallel sub-agents that launch at the same millisecond MUST stagger their first API call. |
| RL-4 | Model fallback (pro → flash) is opt-in per agent and leaves a clearly marked breadcrumb in the report. |
| RL-5 | A circuit-breaker on consecutive 429s pauses parallel gatherers for a cool-down window instead of stampeding. |
| RL-6 | No bug is hidden: every 429-driven degradation appears in the run log AND in the final report's *Run Quality* section. |

---

## Part 1 — Aggressive-but-bounded retry config

### SPEC-RATE-01 — New `RetryConfig` defaults

**Severity**: Critical · **Effort**: XS · **Risk**: None

**Change** `@/home/ai/git_repositories_ai/odin_devcontainer/vertex-ai-toolkit/src/vaig/core/config.py:467-477`:

```python
class RetryConfig(BaseModel):
    """Retry and backoff configuration for API calls."""

    max_retries: int = 5                            # 3 → 5
    initial_delay: float = 1.0                      # unchanged
    max_delay: float = 120.0                        # 60 → 120  (per-attempt cap)
    backoff_multiplier: float = 2.0                 # unchanged
    rate_limit_initial_delay: float = 15.0          # 8 → 15
    rate_limit_max_total_wait_s: float = 300.0      # NEW: 5-min wall-clock cap
    rate_limit_max_total_wait_s_parallel: float = 180.0   # NEW: 3-min cap for parallel agents
```

**Behaviour**. Sleep schedule on successive 429s (with jitter):

| Attempt | Delay (s) | Cumulative |
|---|---|---|
| 1 | 15 | 15 |
| 2 | 30 | 45 |
| 3 | 60 | 105 |
| 4 | 120 | 225 |
| 5 | 120 | 345 → **capped to 300** |

A single API call waits at most 5 minutes on 429s before raising
`GeminiRateLimitError`. Parallel agents (in a fan-out) use the tighter
180 s cap so one saturated agent does not block the others.

**Acceptance criteria**.

1. Sync and async `_retry_with_backoff` honour the new defaults.
2. `rate_limit_max_total_wait_s` cuts off further retries once elapsed
   wall-clock exceeds it, raising `GeminiRateLimitError` with a
   *"waited N s — quota still exhausted"* message.
3. Parallel-launched calls detect they are in a fan-out (via a context
   flag threaded by the orchestrator) and apply the parallel cap.

---

## Part 2 — Stagger parallel sub-agent launches

### SPEC-RATE-02 — Launch-time jitter in parallel orchestration

**Severity**: High · **Effort**: XS · **Risk**: None

**Problem**. The orchestrator's `parallel_sequential` strategy submits
all sub-gatherers to the executor in a tight loop. With N = 5
sub-gatherers, five `generate_content` calls race to the Vertex AI
endpoint inside the same millisecond, guaranteeing correlated 429s.

**Solution**. Sleep a random amount between 0.5 – 2.0 s between
submissions:

```python
# in vaig/agents/orchestrator.py, where ThreadPoolExecutor submits agents
for i, agent_cfg in enumerate(parallel_configs):
    if i > 0:
        jitter = random.uniform(
            self._settings.retry.parallel_launch_jitter_min_s,
            self._settings.retry.parallel_launch_jitter_max_s,
        )
        time.sleep(jitter)
    futures.append(executor.submit(self._run_agent, agent_cfg))
```

**Config additions**:

```python
# in RetryConfig
parallel_launch_jitter_min_s: float = 0.5
parallel_launch_jitter_max_s: float = 2.0
```

Effect: 5 agents over a 2 – 8 s arrival window instead of a burst.

**Acceptance criteria**.

1. Unit test confirms submissions are spaced by at least `min_s` and
   at most `max_s`.
2. Disabled by setting both values to 0 (opt-out).

---

## Part 3 — Circuit breaker on consecutive 429s

### SPEC-RATE-03 — Per-model circuit breaker

**Severity**: High · **Effort**: M · **Risk**: Low

**Problem**. When the quota is globally hot (several concurrent
`vaig` runs, a noisy neighbour project, or a regional capacity event),
each retry adds pressure without gaining throughput. Random
exponential backoff alone does not help — all clients back off in
sync and then stampede together when the timer expires.

**Solution**. Add a process-wide circuit breaker keyed by
`(location, model_id)`:

```python
class ModelCircuitBreaker:
    """Pause calls to a (location, model) pair after N 429s in a window."""

    threshold: int = 3               # ≥3 × 429 in window → OPEN
    window_s: float = 60.0
    cooldown_s: float = 30.0         # stay OPEN at least this long

    state: Literal["closed", "open", "half_open"]
    opened_at: float | None
    failures: deque[float]           # timestamps of 429s

    def record_429(self) -> None: ...
    def record_success(self) -> None: ...
    def check_and_block(self) -> None:
        """Raise CircuitOpen if OPEN; sleep until cooldown elapsed."""
```

**Behaviour**:

- **CLOSED** — normal operation, requests pass through.
- **OPEN** — after `threshold` 429s in `window_s`, block new requests.
  Agents waiting hit `CircuitOpen` and sleep until `cooldown_s`
  elapses.
- **HALF-OPEN** — after cooldown, let exactly one probe request
  through. Success → CLOSED. Failure → OPEN again with longer
  cooldown (up to 2 minutes).

**Storage**: process-wide dict, `threading.Lock`-protected. A
separate breaker per (location, model_id). Async version uses
`asyncio.Lock`.

**User-visible effect**: during an open window, agents log
*"Model `gemini-2.5-pro@us-central1` circuit OPEN — waiting 30 s"*
instead of making 5 pointless requests each.

**Acceptance criteria**.

1. 3 × 429 in 60 s → breaker OPEN; subsequent requests wait, not fire.
2. After cooldown, a single probe determines next state.
3. Success on probe closes the breaker cleanly.
4. Separate breakers per (location, model) — pro@us-central1 failing
   does not block flash@us-central1.

---

## Part 4 — Model fallback pro → flash

### SPEC-RATE-04 — Opt-in model fallback

**Severity**: Medium · **Effort**: M · **Risk**: Medium

**Problem**. `gemini-2.5-pro` quota is strictly smaller than
`gemini-2.5-flash` quota. For agents that emit structured output
(gatherers, verifier), falling back to flash is often acceptable at
slight quality cost.

**Solution**. Per-agent `fallback_model` in config:

```python
# in agent config dicts in skill.py
{
    "name": "health_verifier",
    "model": "gemini-2.5-pro",
    "fallback_model": "gemini-2.5-flash",       # NEW
    "fallback_trigger": "429_persistent",        # NEW: or "never"
    ...
}
```

In `GeminiClient._retry_with_backoff`, after the 3rd 429 (≥ half of
`max_retries`):

```python
if exc.code == 429 and attempt >= retry_cfg.max_retries // 2:
    fallback = self._get_current_agent_fallback_model()
    if fallback and self._current_model_id != fallback:
        logger.warning(
            "429 persistent on %s — falling back to %s for this call",
            self._current_model_id,
            fallback,
        )
        self._current_model_id = fallback
        self._fallback_active = True
        continue  # retry immediately with fallback
```

**Quality bookkeeping**. When a fallback happens, the orchestrator
records `AgentResult.model_degraded=True`. Findings produced by a
degraded agent are annotated with `confidence = min(confidence,
Confidence.MEDIUM)` — never `HIGH` when we are not confident we ran
on the intended model.

**Report surface**. A *Run Quality* section in the final report lists
any model degradations with the affected agents.

**Acceptance criteria**.

1. Persistent 429 on pro → next attempt on flash; agent completes.
2. `model_degraded` annotation propagates to findings.
3. *Run Quality* section rendered when any degradation occurred.
4. `fallback_trigger="never"` disables for critical agents.

---

## Part 5 — User-visible Run Quality section

### SPEC-RATE-05 — Report Run-Quality section

**Severity**: Medium · **Effort**: S · **Risk**: Low

**Problem**. Today, a run can silently produce a degraded report
(e.g. `datadog_gatherer` failed, 4/4 required sections missing, 3/4
recommendation-enrichment timeouts) with no banner in the final
markdown. Users trust the report without realising parts are
synthetic.

**Solution**. Add a *Run Quality* section at the top of the report
when **any** of the following occurred:

| Condition | Badge |
|---|---|
| Any agent failed (non-fatal) | ⚠ `agent_failed` |
| Any agent fell back to a smaller model | ⚠ `model_degraded` |
| Circuit breaker opened during the run | ⚠ `circuit_breaker_tripped` |
| `merged_gatherer_output` missing required sections | ⚠ `incomplete_gather` |
| Enrichment timed out on any recommendation | ⚠ `enrichment_timeout` |
| Attachment context truncated (SPEC-ATT-07.1) | ⚠ `attachment_truncated` |

Example output:

```
## Run Quality ⚠ (3 issues)

| Issue | Where | Consequence |
|---|---|---|
| agent_failed | datadog_gatherer | Datadog metrics not correlated; findings may miss APM signals |
| model_degraded | health_verifier | Fell back pro → flash on 429; findings capped at confidence=medium |
| enrichment_timeout | 3 of 4 recommendations | Expected-output field missing on 3 recommendations |

Suggested action: re-run during a lower-quota window, or pass --model gemini-2.5-flash
for the whole run to avoid 429s on gemini-2.5-pro quota.
```

When the run is clean (no issues), the section is suppressed entirely.

**Acceptance criteria**.

1. Clean run → no Run Quality section (byte-identical to pre-SPEC-RATE-05
   output).
2. Each of the 6 conditions renders with its label and consequence.
3. An exit-code tier is still 0 (soft-degraded), but the caller can
   parse the section to detect degraded runs programmatically.

---

## Part 6 — Documentation and recommendations

### 6.1 · Tune-your-quota doc

Add a short `docs/guides/vertex-ai-quota.md` covering:

- How to check current quota in the GCP console.
- When to request a quota increase vs. migrate to the global endpoint
  (SPEC-GEP-01 in `global-endpoint-v1.md`).
- Command-line workarounds: `--model gemini-2.5-flash` whole-run,
  `--skip-datadog` to shed load, `--offline-mode` with cached
  attachments.

### 6.2 · Default models per agent

Revisit the per-agent model choices in `service_health.skill`:

| Agent | Current model | Proposed change | Rationale |
|---|---|---|---|
| sub-gatherers (5) | gemini-2.5-pro | unchanged; add `fallback_model=flash` | Deep tool planning needs pro; flash is acceptable fallback |
| health_analyzer | gemini-2.5-flash | unchanged | Already cheap |
| health_verifier | gemini-2.5-flash | unchanged | Already cheap |
| health_reporter | gemini-2.5-flash | unchanged | Already cheap |
| recommendation_enricher | gemini-2.5-pro | unchanged; widen timeout to 180 s (from 120 s) | Timeouts on 429 retries |

---

## Part 7 — Rollout

### 7.1 · Sprint plan

| Sprint | SPECs | Exit criteria |
|---|---|---|
| **R1** (1 day) | RATE-01 + RATE-02 | New defaults in place; staggered launches |
| **R2** (2–3 days) | RATE-03 | Circuit breaker wired; unit + integration tests |
| **R3** (2 days) | RATE-04 + RATE-05 | Fallback live; Run Quality section renders |

### 7.2 · Regression guards

- Test fixture: a mock `genai.Client` that returns 429 N times then
  succeeds — current test suite has this shape, extend it.
- Benchmark fixture: a clean run (no 429s) must produce byte-identical
  output before and after the rollout.
- A test with `--model gemini-2.5-pro` forced for all agents + an
  injected 429 storm verifies: (a) circuit breaker trips, (b) fallback
  happens, (c) Run Quality section appears, (d) exit code is 0.

### 7.3 · Compatibility

All changes are additive. Existing configs (`retry.max_retries=3`
etc.) continue to work; the new defaults only apply when the user
does not override them. The circuit breaker and fallback are on by
default but can be disabled:

```yaml
retry:
  circuit_breaker_enabled: false
  parallel_launch_jitter_min_s: 0.0
  parallel_launch_jitter_max_s: 0.0

agents:
  fallback_model_enabled: false
```

---

## Part 8 — Open questions

1. **Should the circuit breaker be shared across `vaig` processes?**
   Cross-process via a file lock or Redis would help noisy-neighbour
   scenarios but adds deployment complexity. *Proposal: v1 keeps it
   in-process; v1.1 optional shared state.*
2. **Do we want per-agent `max_retries` overrides?** `health_analyzer`
   is cheap; it could tolerate more retries. But adding per-agent
   overrides expands the config surface. *Proposal: defer to v1.1
   unless measured need.*
3. **Adaptive backoff** (learn quota recovery time from the last
   successful call)? Marginal gains, significant complexity.
   *Proposal: out of scope for v1.*

---

*End of rate-limit-resilience-v1.md.* Ships alongside
`global-endpoint-v1.md`; the two complement each other — global
endpoint reduces 429 frequency, this spec handles the ones that still
leak through.
