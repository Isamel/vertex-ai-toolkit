# Global Endpoint v1 — Use Vertex AI global endpoint when possible, regional otherwise

**Version**: 1.0
**Target**: vaig v0.19+
**Scope**: `vaig.core.client.GeminiClient` + `vaig.core.config.GCPConfig`
**Status**: Draft for implementation
**Related**: `rate-limit-resilience-v1.md` (SPEC-RATE-* series — global
endpoint reduces 429 frequency; rate-limit resilience handles the
rest).

---

## 0 · Context

### 0.1 · What the global endpoint is

Vertex AI exposes two kinds of endpoints for the `google-genai` SDK:

| Type | Hostname pattern | `location` arg | Quota |
|---|---|---|---|
| **Regional** (current) | `us-central1-aiplatform.googleapis.com` | `"us-central1"` | Per-region per-project |
| **Global** (new) | `aiplatform.googleapis.com` | `"global"` | Aggregate across regions — significantly higher for most models |

The global endpoint:

- Uses cross-region load balancing (Google routes requests to the
  region with spare capacity).
- Has aggregate quota counters — a single project can push far more
  QPM without tripping regional caps.
- Supports all the model IDs we use (`gemini-2.5-pro`, `gemini-2.5-flash`).
- Accepts the same `genai.Client(vertexai=True, location="global", ...)` API.

Caveats:

- **Not** every project is enabled for global from day one; some
  organisations must explicitly opt in.
- **Data residency** semantics change — if the project is subject to
  a regional data-processing restriction, global may be non-compliant.
  Requires explicit opt-in.
- Some **preview** models are region-only.

### 0.2 · Current state in vaig

- `GCPConfig.location` defaults to `"us-central1"` (hardcoded
  regional): `@/home/ai/git_repositories_ai/odin_devcontainer/vertex-ai-toolkit/src/vaig/core/config.py:61`.
- `GCPConfig.fallback_location` also defaults to `"us-central1"` —
  **identical to primary**, making the SSL-fallback path in
  `_reinitialize_with_fallback` a no-op
  (`@/home/ai/git_repositories_ai/odin_devcontainer/vertex-ai-toolkit/src/vaig/core/config.py:62`).
- No awareness of the global endpoint; no probe logic; no automatic
  fallback on quota issues.

### 0.3 · Design goal

> When the project supports the global endpoint, vaig should use it.
> If it becomes unavailable (unsupported, permission denied, or
> persistent 429) the client should transparently fall back to a
> regional endpoint without breaking the run.

### 0.4 · Invariants

| # | Invariant |
|---|---|
| GE-1 | Default behaviour for new installs uses global when available. |
| GE-2 | Users who explicitly set `location` in their config keep that behaviour (backwards compatibility). |
| GE-3 | Fallback from global → regional is automatic and logged; never silent. |
| GE-4 | Users subject to regional data-residency requirements can disable global entirely with one setting. |
| GE-5 | The probe to determine global availability happens **once per client lifetime** and is cached. |
| GE-6 | Every model call records which endpoint it used; the Run Quality section (SPEC-RATE-05) surfaces endpoint flips. |

---

## Part 1 — Configuration model

### SPEC-GEP-01 — Endpoint-mode configuration

**Severity**: Critical · **Effort**: XS · **Risk**: None

**Change** to `GCPConfig` in `@/home/ai/git_repositories_ai/odin_devcontainer/vertex-ai-toolkit/src/vaig/core/config.py:58-70`:

```python
class GCPConfig(BaseModel):
    """GCP project and endpoint configuration."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = ""
    location: str = "global"                                    # was "us-central1"
    fallback_location: str = "us-central1"                      # regional fallback
    endpoint_mode: Literal["auto", "global", "regional"] = "auto"
    global_probe_timeout_s: float = 2.0
    global_probe_cache_ttl_s: int = 3_600                       # 1 h
    available_projects: list[ProjectEntry] = Field(default_factory=list)
```

**Behaviour of `endpoint_mode`**:

| Mode | Behaviour |
|---|---|
| `"auto"` (default) | Probe the global endpoint at client init. Use global if reachable; otherwise `fallback_location`. Probe result cached for `global_probe_cache_ttl_s`. |
| `"global"` | Always use `location="global"`. Fail hard if global is unreachable (no silent downgrade). |
| `"regional"` | Always use `fallback_location` (or `location` when it is not "global"). Never touch the global endpoint. For data-residency-restricted projects. |

**Backwards-compatibility migration**:

- Users who had `location: "us-central1"` in their YAML config keep
  exactly that behaviour because `endpoint_mode` defaults to `"auto"`
  **and** `location` is `"global"` only when unspecified. When
  `location` is explicitly non-global, vaig uses it as-is (regional
  mode implicit).

**Acceptance criteria**.

1. Fresh install → `endpoint_mode="auto"`, `location="global"`. Probe runs. If OK, calls hit `aiplatform.googleapis.com`.
2. Existing install with `location: "us-central1"` in YAML → unchanged behaviour, no probe runs.
3. `endpoint_mode: "regional"` → never probes, never calls global.
4. `endpoint_mode: "global"` + global unreachable → startup error with actionable message *"Global endpoint not enabled for project X; set `endpoint_mode: auto` or `regional`."*

---

## Part 2 — Probe and initial selection

### SPEC-GEP-02 — Startup probe

**Severity**: High · **Effort**: S · **Risk**: Low

**Problem**. The client cannot blindly use `location="global"` —
projects without the global endpoint enabled will get 404 or 403 on
every call. A one-time probe at init determines availability.

**Solution**. In `GeminiClient.initialize()` (and async variant), add
a `_resolve_initial_location` method:

```python
def _resolve_initial_location(self) -> str:
    """Pick between global and regional at init time."""
    cfg = self._settings.gcp

    # Explicit modes bypass the probe.
    if cfg.endpoint_mode == "regional":
        return cfg.fallback_location
    if cfg.endpoint_mode == "global":
        return "global"

    # Auto mode: check cache first.
    cached = self._read_probe_cache()
    if cached is not None:
        logger.debug("Endpoint probe cache hit: %s", cached)
        return cached

    # Perform a cheap probe.
    probed = self._probe_global()
    self._write_probe_cache(probed)
    return "global" if probed else cfg.fallback_location

def _probe_global(self) -> bool:
    """Return True if the global endpoint is reachable for this project.

    The probe creates a temporary Client against ``location="global"``
    and calls ``models.list()`` with a short timeout. Any 2xx = available;
    404/403/timeout = unavailable.
    """
    try:
        from google import genai

        probe_client = genai.Client(
            vertexai=True,
            project=self._settings.gcp.project_id,
            location="global",
            credentials=get_credentials(self._settings),
            http_options=types.HttpOptions(
                timeout=self._settings.gcp.global_probe_timeout_s * 1_000,
            ),
        )
        next(iter(probe_client.models.list(config={"page_size": 1})), None)
        logger.info("Global Vertex AI endpoint reachable — using global")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "Global endpoint unavailable (%s) — using regional %s",
            exc.__class__.__name__,
            self._settings.gcp.fallback_location,
        )
        return False
```

**Probe cache**. Store last probe result in
`~/.vaig/cache/endpoint-probe.json` with TTL of 1 hour (configurable).
Cache key: `(project_id, credentials_hash)`. Invalidate when
credentials change.

**Cache structure**:

```json
{
    "project_id": "my-project",
    "credentials_hash": "sha256:…",
    "global_available": true,
    "probed_at": 1740000000,
    "ttl_s": 3600
}
```

**Acceptance criteria**.

1. First run (no cache) → probe runs, result cached, initialization succeeds.
2. Second run within TTL → cache hit, no network call for probe.
3. Cache miss after TTL → re-probe happens.
4. Probe times out → treated as "unavailable", falls back to regional.
5. Changing `GOOGLE_APPLICATION_CREDENTIALS` → cache key changes, probe re-runs.

---

## Part 3 — Runtime fallback on persistent 429

### SPEC-GEP-03 — Global → regional fallback on 429

**Severity**: High · **Effort**: S · **Risk**: Low

**Problem**. Even when global is normally available, a sudden quota
spike on the global endpoint can produce 429s. Today that 429 burns
through all retries and fails the agent. The right response is to
fall back to the regional endpoint (which has separate quota).

**Solution**. Extend `_retry_with_backoff` to detect the fallback
condition and trigger `_reinitialize_with_fallback` even for 429
(today it only triggers on SSL/connection errors).

```python
# in _retry_with_backoff, after the 2nd consecutive 429:
if exc.code == 429 and self._active_location == "global" and not self._using_fallback:
    if self._consecutive_429_count >= 2:
        logger.warning(
            "Persistent 429 on global endpoint — switching to regional %s",
            self._settings.gcp.fallback_location,
        )
        self._reinitialize_with_fallback()
        continue  # retry against the new (regional) client
```

**State tracking**. Add to `GeminiClient`:

```python
self._consecutive_429_count: int = 0
self._endpoint_flips: list[EndpointFlip] = []
```

Incremented on each 429, reset on success. `EndpointFlip` logs
`(at: timestamp, from: "global", to: "us-central1", reason:
"persistent_429")` for the Run Quality section.

**Behaviour**:

- After switching to regional, stay there for the rest of the run.
- Next `vaig live` invocation starts fresh (global again if probe
  still says available, unless cache TTL expired and probe fails).

**Acceptance criteria**.

1. Persistent 429 on global → flip to regional; run succeeds.
2. Flip recorded; Run Quality section (SPEC-RATE-05) shows
   `endpoint_flipped` badge.
3. Same run does not flip back to global.
4. Next run's cache-hit behaviour respected (global still cached
   as available for this TTL window).

---

## Part 4 — CLI override

### SPEC-GEP-04 — CLI flag `--endpoint`

**Severity**: Low · **Effort**: XS · **Risk**: None

**Solution**. New global CLI option:

```bash
vaig live --endpoint {auto|global|regional} ...
```

Maps directly to `settings.gcp.endpoint_mode` for that run. Does not
persist. Useful for:

- Users who want to force regional during an incident.
- CI runs that need deterministic endpoint behaviour.
- Debugging.

**Acceptance criteria**.

1. `--endpoint regional` forces regional for the run.
2. `--endpoint global` forces global; errors if unavailable.
3. `--endpoint auto` uses probe logic (default).
4. Environment variable `VAIG_ENDPOINT` provides the same override.

---

## Part 5 — Observability

### SPEC-GEP-05 — Endpoint observability

**Severity**: Low · **Effort**: XS · **Risk**: None

**Logging additions**:

```
INFO  vaig.core.client Using endpoint: global (project=my-project)
INFO  vaig.core.client Endpoint flipped: global → us-central1 (reason: persistent_429)
DEBUG vaig.core.client Endpoint probe cache hit: global (age 142 s, TTL 3600 s)
DEBUG vaig.core.client Endpoint probe cache miss — probing…
INFO  vaig.core.client Global endpoint unavailable (PermissionDenied) — using regional us-central1
```

**Run Quality surface** (SPEC-RATE-05):

| Condition | Rendered as |
|---|---|
| Client initialized on regional despite `endpoint_mode="auto"` | `endpoint_degraded: using us-central1 (global unavailable for project)` |
| Runtime flip global → regional | `endpoint_flipped: 429-driven fallback at 14:02 UTC` |
| `endpoint_mode="global"` forced and failed | Run aborted before any agent — user sees error immediately |

**Metric** (for future `vaig.core.metrics`):

- Counter: `vaig_endpoint_calls_total{endpoint=global|regional, model=..., result=ok|429|error}`.
- Histogram: `vaig_endpoint_flip_duration_seconds` — wall-clock time
  from first 429 to successful regional call.

**Acceptance criteria**.

1. `vaig live --verbose` shows the endpoint INFO log lines.
2. `vaig live --diag endpoint` sub-command prints probe cache status
   and last flip history (new diagnostic utility).

---

## Part 6 — Rollout

### 6.1 · Sprint plan

| Sprint | SPECs | Exit criteria |
|---|---|---|
| **G1** (1 day) | GEP-01 + GEP-02 | Probe works; `endpoint_mode` respected |
| **G2** (1–2 days) | GEP-03 | Runtime fallback on 429 |
| **G3** (1 day) | GEP-04 + GEP-05 | CLI flag + observability |

### 6.2 · Regression guards

- **Existing users with `location: "us-central1"` in YAML** → zero
  behavioural change (probe does not run; SPEC-GEP-01 #2).
- **Golden test** for a full `vaig live` run against a mocked Vertex
  AI — must work identically when the mock rejects global (forcing
  regional fallback).
- **Unit test** for the probe cache (TTL, key composition).

### 6.3 · Security / compliance

- **Data residency**: organizations subject to regional restrictions
  MUST set `endpoint_mode: "regional"` in their config (documented).
- **Audit log**: every `EndpointFlip` is logged at INFO so SIEM tools
  can track cross-region fallbacks.
- **Credentials**: the probe uses the same ADC credentials as the
  main client. No separate credential path.

### 6.4 · Migration guide (for `docs/guides/upgrading.md`)

```markdown
## vaig 0.20 — Global endpoint by default

vaig 0.20 introduces automatic use of the Vertex AI global endpoint.
New installs will probe for global availability at first run and use
it when reachable. This typically reduces 429 errors by ≥ 10×.

### What changed

- Default `gcp.location` is now `"global"` (was `"us-central1"`).
- New `gcp.endpoint_mode` setting: `"auto"` (default), `"global"`, `"regional"`.
- Persistent 429s on global now trigger automatic regional fallback.

### Backwards compatibility

If your `~/.vaig/config.yaml` sets `gcp.location` explicitly to
a region name, **nothing changes** — you continue to use that region.

### For data-residency-restricted projects

Set `gcp.endpoint_mode: "regional"` in your config to disable the
global probe entirely.
```

---

## Part 7 — Open questions

1. **Should we probe on `vaig` startup or on first model call?** Startup
   means one pre-flight call that most users don't need (they are
   going to make a model call anyway). First-call probing is cheaper
   but adds a first-call latency spike. *Proposal: lazy probe — on
   first call, cache the result.*
2. **Separate probe cache per model?** Some models may be region-only
   preview; we could probe `gemini-2.5-pro` specifically. *Proposal:
   single probe (assume all our models support global); revisit if a
   future preview model breaks this.*
3. **Invalidate probe cache on 429 flip?** If global flipped due to
   429 this run, should the cache still say global-available for the
   next run? *Proposal: yes — the flip is transient quota, not a
   permanent unavailability; keep the cache.*

---

*End of global-endpoint-v1.md.* Complements
`rate-limit-resilience-v1.md`: use this spec to raise ceiling (more
quota via global), use the other to handle the 429s that still happen.
