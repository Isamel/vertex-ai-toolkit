"""FastAPI webhook server — receives Datadog alert webhooks and runs vaig analysis.

The webhook server:
1. Receives Datadog alert webhooks (POST /webhook/datadog)
2. Parses the alert, extracts service/cluster/namespace info from tags
3. Runs a vaig health analysis on the affected service (background task)
4. Dispatches results via NotificationDispatcher (PagerDuty + Google Chat)

Requires: ``pip install 'vertex-ai-toolkit[web]'`` (fastapi + uvicorn)

Deployment targets:
- Cloud Run (stateless container with ``PORT`` env var)
- Kubernetes sidecar (ClusterIP service, liveness/readiness probes)
- Local machine (``vaig webhook-server`` CLI command)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Optional dependency guard ────────────────────────────────

try:
    from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the webhook server.\n"
        "Install it with: pip install 'vertex-ai-toolkit[web]'\n"
        "Or: pip install fastapi uvicorn"
    ) from None


# ── Datadog payload models ───────────────────────────────────


class DatadogOrg(BaseModel):
    """Datadog organization info from webhook payload."""

    id: str = ""
    name: str = ""


class DatadogWebhookPayload(BaseModel):
    """Datadog Monitor webhook payload.

    Parses the JSON body that Datadog sends when a Monitor triggers.
    All fields are optional with sensible defaults since Datadog's
    payload can vary based on monitor type and configuration.
    """

    id: str = ""
    title: str = ""
    text: str = ""
    date: int = 0
    alert_type: str = ""
    alert_id: str = ""
    alert_metric: str = ""
    alert_status: str = ""
    alert_title: str = ""
    alert_transition: str = ""
    hostname: str = ""
    tags: str = ""
    priority: str = "normal"
    org: DatadogOrg = Field(default_factory=DatadogOrg)


# ── Tag parsing ──────────────────────────────────────────────


def parse_datadog_tags(tags_str: str) -> dict[str, str]:
    """Parse a Datadog comma-separated tag string into a dict.

    Datadog tags are in the format ``key:value`` separated by commas.
    Tags without a colon are stored with an empty-string value.

    Args:
        tags_str: Comma-separated tag string, e.g.
            ``"env:production,service:my-service,kube_cluster_name:my-cluster"``

    Returns:
        Dictionary mapping tag keys to values.
    """
    tags: dict[str, str] = {}
    if not tags_str:
        return tags

    for tag in tags_str.split(","):
        tag = tag.strip()
        if not tag:
            continue
        if ":" in tag:
            key, _, value = tag.partition(":")
            tags[key.strip()] = value.strip()
        else:
            tags[tag] = ""

    return tags


# ── Deduplication cache ──────────────────────────────────────


class DedupCache:
    """In-memory deduplication cache with TTL-based expiry.

    Prevents re-analyzing the same alert within a cooldown window.
    Thread-safe for single-process use (GIL protects dict operations).
    Uses an atomic ``try_record`` to prevent race conditions in async contexts.
    Bounded to ``max_size`` entries to prevent memory leaks.
    """

    def __init__(self, cooldown_seconds: int = 300, max_size: int = 10_000) -> None:
        self._cooldown = cooldown_seconds
        self._max_size = max_size
        self._cache: dict[str, float] = {}  # key → expiry timestamp
        self._lock = asyncio.Lock()

    def is_duplicate(self, key: str) -> bool:
        """Check if the key was seen within the cooldown window.

        Also cleans up expired entries lazily.

        Args:
            key: Dedup key (typically ``alert_id`` or ``alert_id:service``).

        Returns:
            ``True`` if the key is still within the cooldown window.
        """
        now = time.monotonic()
        self._cleanup(now)

        if key in self._cache and self._cache[key] > now:
            return True
        return False

    def record(self, key: str) -> None:
        """Record a key with the current cooldown window.

        Args:
            key: Dedup key to record.
        """
        self._cache[key] = time.monotonic() + self._cooldown
        self._evict_if_needed()

    async def try_record(self, key: str) -> bool:
        """Atomically check-and-record a dedup key.

        Returns:
            ``True`` if the key was NOT a duplicate and has been recorded.
            ``False`` if the key is a duplicate (already in cooldown).
        """
        async with self._lock:
            if self.is_duplicate(key):
                return False
            self.record(key)
            return True

    def _cleanup(self, now: float) -> None:
        """Remove expired entries from the cache."""
        expired = [k for k, v in self._cache.items() if v <= now]
        for k in expired:
            del self._cache[k]

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max_size."""
        while len(self._cache) > self._max_size:
            # Remove the first (oldest) key
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._cache)


# ── Daily budget counter ─────────────────────────────────────


class DailyBudgetCounter:
    """Tracks analyses per day, resetting at midnight UTC.

    Prevents Vertex AI cost blow-up from alert storms by enforcing
    a configurable ``max_analyses_per_day`` limit.
    """

    def __init__(self, max_per_day: int = 50) -> None:
        self._max = max_per_day
        self._count: int = 0
        self._date: str = self._today()
        self._lock = asyncio.Lock()

    def can_analyze(self) -> bool:
        """Check if budget allows another analysis.

        Returns:
            ``True`` if the daily budget is not exhausted.
        """
        self._maybe_reset()
        return self._count < self._max

    def increment(self) -> int:
        """Increment the counter and return the new count.

        Returns:
            The updated analysis count for today.
        """
        self._maybe_reset()
        self._count += 1
        return self._count

    async def try_increment(self) -> bool:
        """Atomically check budget and increment if allowed.

        Returns:
            ``True`` if budget was available and has been consumed.
            ``False`` if the daily budget is exhausted.
        """
        async with self._lock:
            if not self.can_analyze():
                return False
            self.increment()
            return True

    @property
    def count(self) -> int:
        """Return the current count for today."""
        self._maybe_reset()
        return self._count

    @property
    def remaining(self) -> int:
        """Return the number of analyses remaining today."""
        self._maybe_reset()
        return max(0, self._max - self._count)

    @property
    def max_per_day(self) -> int:
        """Return the configured daily limit."""
        return self._max

    def _maybe_reset(self) -> None:
        """Reset counter if the UTC date has changed."""
        today = self._today()
        if today != self._date:
            logger.info(
                "Daily budget reset: %d analyses yesterday (%s), new day %s",
                self._count,
                self._date,
                today,
            )
            self._count = 0
            self._date = today

    @staticmethod
    def _today() -> str:
        """Return today's date as a string in UTC."""
        return datetime.now(UTC).strftime("%Y-%m-%d")


# ── HMAC signature verification ──────────────────────────────


def verify_hmac_signature(
    body: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify a Datadog webhook HMAC-SHA256 signature.

    Datadog signs webhooks with the shared secret using HMAC-SHA256.
    The signature is sent in the ``X-DD-Signature`` header as a
    hex-encoded digest.

    Args:
        body: Raw request body bytes.
        signature: Hex-encoded HMAC-SHA256 signature from the header.
        secret: Shared HMAC secret configured in Datadog.

    Returns:
        ``True`` if the signature matches.
    """
    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Analysis result model ────────────────────────────────────


class AnalysisResult(BaseModel):
    """Result of a background analysis triggered by a webhook."""

    alert_id: str
    service_name: str
    cluster_name: str = ""
    namespace: str = ""
    status: str = "pending"  # pending | running | completed | failed
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    dispatch_errors: list[str] = Field(default_factory=list)


# ── Background analysis runner ───────────────────────────────


async def run_analysis_background(
    alert_id: str,
    service_name: str,
    cluster_name: str,
    namespace: str,
    results_store: OrderedDict[str, AnalysisResult],
    budget_counter: DailyBudgetCounter,
    analysis_timeout: int = 600,
) -> None:
    """Run a vaig health analysis in the background and dispatch results.

    This function is designed to be passed to FastAPI's ``BackgroundTasks``.
    It attempts to:
    1. Load vaig settings and create a NotificationDispatcher
    2. Run the service health analysis via the orchestrator
    3. Dispatch results to PagerDuty + Google Chat

    If any step fails, the error is logged and stored in the results dict.

    Args:
        alert_id: The Datadog alert ID (for tracking).
        service_name: Service to analyze.
        cluster_name: GKE cluster name.
        namespace: Kubernetes namespace.
        results_store: Shared ordered dict to track analysis progress.
        budget_counter: Daily budget counter (already incremented).
        analysis_timeout: Maximum seconds to wait for analysis.
    """
    result = results_store.get(alert_id)
    if result is None:
        result = AnalysisResult(
            alert_id=alert_id,
            service_name=service_name,
            cluster_name=cluster_name,
            namespace=namespace,
        )
    result.status = "running"
    result.started_at = datetime.now(UTC).isoformat()
    results_store[alert_id] = result

    try:
        # Late import to avoid circular dependencies and allow standalone use
        from vaig.core.config import get_settings
        from vaig.integrations.dispatcher import AlertContext, NotificationDispatcher

        settings = get_settings()
        dispatcher = NotificationDispatcher.from_config(settings)

        alert_context = AlertContext(
            alert_id=alert_id,
            source="datadog",
            service_name=service_name,
            cluster_name=cluster_name,
            namespace=namespace,
        )

        # Try to import and run the orchestrator
        try:
            from vaig.agents.orchestrator import Orchestrator
            from vaig.core.client import GeminiClient
            from vaig.core.config import GKEConfig
            from vaig.core.gke import register_live_tools
            from vaig.core.prompt_defense import _sanitize_namespace
            from vaig.skills.registry import SkillRegistry

            # Sanitize external inputs before embedding in LLM prompt
            safe_namespace = _sanitize_namespace(namespace) if namespace else ""
            safe_service = re.sub(r"[^a-zA-Z0-9._-]", "", service_name)[:128]
            safe_cluster = re.sub(r"[^a-zA-Z0-9._-]", "", cluster_name)[:128]

            query = (
                f"Run a service health analysis for service '{safe_service}'"
            )
            if safe_namespace:
                query += f" in namespace '{safe_namespace}'"
            if safe_cluster:
                query += f" on cluster '{safe_cluster}'"

            client = GeminiClient(settings)
            orchestrator = Orchestrator(client, settings)

            # Resolve service-health skill from the registry
            skill_registry = SkillRegistry(settings)
            skill = skill_registry.get("service-health")
            if skill is None:
                raise RuntimeError("service-health skill not found in registry")

            # Build GKE config and tool registry for live tools
            gke_config = GKEConfig(
                cluster_name=cluster_name or "",
                default_namespace=namespace or "",
            )
            tool_registry = register_live_tools(gke_config, settings=settings)

            # Run sync orchestrator call in a thread to avoid blocking the event loop,
            # and enforce the configured analysis timeout.
            orchestrator_result = await asyncio.wait_for(
                asyncio.to_thread(
                    orchestrator.execute_with_tools,
                    query,
                    skill,
                    tool_registry,
                    gke_namespace=namespace or "",
                    gke_cluster_name=cluster_name or "",
                ),
                timeout=analysis_timeout,
            )

            # Try to extract HealthReport from the orchestrator result
            report = None
            if hasattr(orchestrator_result, "report") and orchestrator_result.report is not None:
                report = orchestrator_result.report
            elif hasattr(orchestrator_result, "output"):
                # Attempt to parse the output as a HealthReport
                try:
                    from vaig.skills.service_health.schema import HealthReport
                    from vaig.utils.json_cleaner import clean_llm_json

                    cleaned = clean_llm_json(orchestrator_result.output)
                    report = HealthReport.model_validate_json(cleaned)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    logger.debug(
                        "Could not parse orchestrator output as HealthReport for alert %s",
                        alert_id,
                    )

            if report is not None:
                dispatch_result = dispatcher.dispatch(report, alert_context=alert_context)
                if dispatch_result.has_errors:
                    result.dispatch_errors = dispatch_result.errors
                    logger.warning(
                        "Dispatch completed with errors for alert %s: %s",
                        alert_id,
                        dispatch_result.errors,
                    )
                result.status = "completed"
            else:
                logger.warning(
                    "Analysis for alert %s completed but no HealthReport was produced",
                    alert_id,
                )
                result.status = "completed"

        except ImportError:
            logger.warning(
                "Orchestrator not available — analysis skipped for alert %s. "
                "Ensure vaig core dependencies are installed.",
                alert_id,
            )
            result.status = "failed"
            result.error = "Orchestrator not available — core dependencies missing"

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        logger.exception("Background analysis failed for alert %s", alert_id)
        result.status = "failed"
        result.error = str(exc)[:500]
    finally:
        result.completed_at = datetime.now(UTC).isoformat()
        results_store[alert_id] = result


# ── FastAPI app factory ──────────────────────────────────────


def create_webhook_app(
    hmac_secret: str = "",
    max_analyses_per_day: int = 50,
    dedup_cooldown_seconds: int = 300,
    analysis_timeout_seconds: int = 600,
) -> FastAPI:
    """Create and configure the FastAPI webhook server application.

    When called with no arguments (e.g. via ``uvicorn --factory``), reads
    configuration from internal env vars set by the CLI command, falling
    back to the function-signature defaults.

    Args:
        hmac_secret: Optional HMAC secret for Datadog webhook verification.
            When empty, signature verification is skipped.
        max_analyses_per_day: Maximum analyses per UTC day (cost protection).
        dedup_cooldown_seconds: Seconds to wait before re-analyzing
            the same alert (dedup window).
        analysis_timeout_seconds: Maximum seconds for a single analysis.

    Returns:
        Configured FastAPI application.
    """
    import os

    # Support uvicorn --factory mode: CLI sets these env vars before launch
    hmac_secret = hmac_secret or os.environ.get("_VAIG_WEBHOOK_HMAC_SECRET", "")
    max_analyses_per_day = max_analyses_per_day if max_analyses_per_day != 50 else int(
        os.environ.get("_VAIG_WEBHOOK_MAX_ANALYSES", "50")
    )
    dedup_cooldown_seconds = dedup_cooldown_seconds if dedup_cooldown_seconds != 300 else int(
        os.environ.get("_VAIG_WEBHOOK_DEDUP_COOLDOWN", "300")
    )
    analysis_timeout_seconds = analysis_timeout_seconds if analysis_timeout_seconds != 600 else int(
        os.environ.get("_VAIG_WEBHOOK_ANALYSIS_TIMEOUT", "600")
    )
    app = FastAPI(
        title="VAIG Webhook Server",
        description="Receives Datadog alert webhooks and runs vaig health analysis",
        version="1.0.0",
    )

    # Shared state
    dedup_cache = DedupCache(cooldown_seconds=dedup_cooldown_seconds)
    budget_counter = DailyBudgetCounter(max_per_day=max_analyses_per_day)
    results_store: OrderedDict[str, AnalysisResult] = OrderedDict()
    _MAX_RESULTS = 1000  # Bounded store to prevent memory leaks

    # ── Health check ─────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint for readiness probes.

        Returns a simple JSON response indicating the server is running.
        """
        return {"status": "healthy"}

    # ── Stats endpoint ───────────────────────────────────────

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        """Return analysis statistics.

        Includes today's analysis count, remaining budget, and
        dedup cache size.
        """
        # Snapshot recent items to avoid RuntimeError from concurrent dict mutation
        recent_items = list(results_store.items())[-10:]
        return {
            "analyses_today": budget_counter.count,
            "budget_remaining": budget_counter.remaining,
            "budget_max": budget_counter.max_per_day,
            "dedup_cache_size": dedup_cache.size,
            "recent_analyses": {
                k: {
                    "service": v.service_name,
                    "status": v.status,
                    "started_at": v.started_at,
                    "completed_at": v.completed_at,
                }
                for k, v in recent_items
            },
        }

    # ── Datadog webhook endpoint ─────────────────────────────

    @app.post("/webhook/datadog", status_code=202)
    async def datadog_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_dd_signature: str | None = Header(None, alias="X-DD-Signature"),
    ) -> JSONResponse:
        """Receive a Datadog alert webhook.

        Flow:
        1. Verify HMAC signature (if configured)
        2. Parse the Datadog webhook payload
        3. Check dedup cache (skip if recently analyzed)
        4. Check daily budget (reject if exhausted)
        5. Queue background analysis
        6. Return 202 Accepted immediately

        Args:
            request: The incoming HTTP request.
            background_tasks: FastAPI background task runner.
            x_dd_signature: Optional HMAC signature header from Datadog.

        Returns:
            202 Accepted with analysis tracking info.
        """
        body = await request.body()

        # 1. HMAC verification (if secret is configured)
        if hmac_secret:
            if not x_dd_signature:
                raise HTTPException(
                    status_code=401,
                    detail="Missing X-DD-Signature header",
                )
            if not verify_hmac_signature(body, x_dd_signature, hmac_secret):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid HMAC signature",
                )

        # 2. Parse payload
        try:
            payload = DatadogWebhookPayload.model_validate_json(body)
        except Exception as exc:
            logger.warning("Failed to parse Datadog webhook payload: %s", exc)
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON payload",
            ) from exc

        # Extract service info from tags
        tags = parse_datadog_tags(payload.tags)
        service_name = tags.get("service", payload.hostname or "unknown")
        cluster_name = tags.get("kube_cluster_name", "")
        namespace = tags.get("kube_namespace", "")

        alert_id = payload.alert_id or payload.id or f"dd-{int(time.time())}"
        dedup_key = f"{alert_id}:{service_name}"

        logger.info(
            "Received Datadog webhook: alert_id=%s service=%s cluster=%s "
            "namespace=%s transition=%s",
            alert_id,
            service_name,
            cluster_name,
            namespace,
            payload.alert_transition,
        )

        # 3. Atomic dedup check-and-record
        if not await dedup_cache.try_record(dedup_key):
            logger.info(
                "Skipping duplicate alert %s for service %s (cooldown active)",
                alert_id,
                service_name,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "status": "skipped",
                    "reason": "duplicate",
                    "alert_id": alert_id,
                    "service": service_name,
                    "message": f"Alert already analyzed within {dedup_cooldown_seconds}s cooldown",
                },
            )

        # 4. Atomic budget check-and-increment
        if not await budget_counter.try_increment():
            logger.warning(
                "Daily analysis budget exhausted (%d/%d) — rejecting alert %s",
                budget_counter.count,
                budget_counter.max_per_day,
                alert_id,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "status": "rejected",
                    "reason": "budget_exhausted",
                    "alert_id": alert_id,
                    "analyses_today": budget_counter.count,
                    "budget_max": budget_counter.max_per_day,
                    "message": "Daily analysis budget exhausted",
                },
            )

        # 5. Queue background analysis
        results_store[alert_id] = AnalysisResult(
            alert_id=alert_id,
            service_name=service_name,
            cluster_name=cluster_name,
            namespace=namespace,
            status="pending",
        )
        # Evict oldest entries if results store exceeds max size
        while len(results_store) > _MAX_RESULTS:
            results_store.popitem(last=False)

        background_tasks.add_task(
            run_analysis_background,
            alert_id=alert_id,
            service_name=service_name,
            cluster_name=cluster_name,
            namespace=namespace,
            results_store=results_store,
            budget_counter=budget_counter,
            analysis_timeout=analysis_timeout_seconds,
        )

        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "alert_id": alert_id,
                "service": service_name,
                "cluster": cluster_name,
                "namespace": namespace,
                "analyses_today": budget_counter.count,
                "budget_remaining": budget_counter.remaining,
                "message": "Analysis queued",
            },
        )

    return app
