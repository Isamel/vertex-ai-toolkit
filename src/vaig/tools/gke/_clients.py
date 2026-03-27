"""K8s client infrastructure — lazy import guard, client creation, caching, and autopilot detection."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import yaml

from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── K8s client cache ──────────────────────────────────────────
# Keyed on (kubeconfig_path, context, proxy_url) so each unique
# GKEConfig combination creates clients only once.
_CLIENT_CACHE: dict[tuple[str, str, str], tuple[Any, ...]] = {}

# ── Autopilot detection cache ─────────────────────────────────
# Keyed on (project_id, location, cluster_name).
# Values: True (Autopilot), False (Standard), None (detection failed).
_AUTOPILOT_CACHE: dict[tuple[str, str, str], bool | None] = {}

# ── Lazy import guard ─────────────────────────────────────────
# The ``kubernetes`` package is an optional dependency (``pip install vertex-ai-toolkit[live]``).
# All public functions fail gracefully with a descriptive ToolResult when it is missing.

_K8S_AVAILABLE = True
_K8S_IMPORT_ERROR: str | None = None

try:
    from kubernetes import client as k8s_client  # noqa: WPS433
    from kubernetes import config as k8s_config  # noqa: WPS433
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433, F401
except ImportError as _exc:
    _K8S_AVAILABLE = False
    _K8S_IMPORT_ERROR = (
        "The 'kubernetes' package is not installed. "
        "Install it with: pip install vertex-ai-toolkit[live]"
    )


def _k8s_unavailable() -> ToolResult:
    """Return a ToolResult indicating that the kubernetes SDK is not installed."""
    return ToolResult(output=_K8S_IMPORT_ERROR or "kubernetes SDK not available", error=True)


# ── Autopilot detection ──────────────────────────────────────


def _query_autopilot_status(
    project: str,
    location: str,
    cluster: str,
    credentials: Credentials | None = None,
) -> bool:
    """Query the GKE API for the Autopilot status of a cluster.

    This is an internal helper extracted so tests can mock it without
    fighting Python's import machinery.

    Args:
        project: GCP project ID.
        location: GKE cluster location (zone or region).
        cluster: GKE cluster name.
        credentials: Optional GCP credentials.  When ``None``, the
            ``ClusterManagerClient`` uses Application Default Credentials.

    Raises:
        ImportError: If ``google-cloud-container`` is not installed.
        Exception: On any API or network error.
    """
    import google.cloud.container_v1 as container_v1  # noqa: WPS433

    kwargs: dict[str, Any] = {}
    if credentials is not None:
        kwargs["credentials"] = credentials
    client = container_v1.ClusterManagerClient(**kwargs)
    name = f"projects/{project}/locations/{location}/clusters/{cluster}"
    cluster_obj = client.get_cluster(name=name)
    return bool(cluster_obj.autopilot and cluster_obj.autopilot.enabled)


def detect_autopilot(
    gke_config: GKEConfig,
    credentials: Credentials | None = None,
) -> bool | None:
    """Detect whether the GKE cluster is running in Autopilot mode.

    Uses the ``google-cloud-container`` library to query the GKE API for
    ``cluster.autopilot.enabled``.  Results are cached per
    (project_id, location, cluster_name) tuple.

    Args:
        gke_config: GKE configuration with ``project_id``, ``location``,
            and ``cluster_name`` populated.
        credentials: Optional GCP credentials for the GKE API.
            When ``None``, the client uses Application Default Credentials.

    Returns:
        ``True`` if Autopilot, ``False`` if Standard, ``None`` if detection
        failed (missing config, missing library, API error).
    """
    project = gke_config.project_id
    location = gke_config.location
    cluster = gke_config.cluster_name

    if not project or not location or not cluster:
        logger.debug(
            "Autopilot detection skipped: missing project_id=%r, location=%r, cluster_name=%r",
            project, location, cluster,
        )
        return None

    cache_key = (project, location, cluster)
    if cache_key in _AUTOPILOT_CACHE:
        return _AUTOPILOT_CACHE[cache_key]

    try:
        is_autopilot = _query_autopilot_status(project, location, cluster, credentials=credentials)
        _AUTOPILOT_CACHE[cache_key] = is_autopilot
        logger.info("GKE Autopilot detection: cluster=%s autopilot=%s", cluster, is_autopilot)
        return is_autopilot

    except ImportError:
        logger.warning(
            "google-cloud-container not installed — Autopilot detection unavailable. "
            "Install with: pip install vertex-ai-toolkit[live]"
        )
        _AUTOPILOT_CACHE[cache_key] = None
        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("Autopilot detection failed for %s: %s", cluster, exc)
        _AUTOPILOT_CACHE[cache_key] = None
        return None


def clear_autopilot_cache() -> None:
    """Clear the Autopilot detection cache (useful for testing)."""
    _AUTOPILOT_CACHE.clear()


# ── K8s client helper (Task 2.6) ─────────────────────────────


def _extract_proxy_url_from_kubeconfig(
    kubeconfig_path: str | None = None,
    context: str | None = None,
) -> str | None:
    """Extract ``proxy-url`` from the active kubeconfig cluster entry.

    The ``kubernetes`` Python client (v35) ignores the ``proxy-url`` field
    that ``kubectl`` honours.  This helper reads the raw YAML so we can
    apply it manually via ``Configuration.proxy``.
    """
    kube_path = kubeconfig_path or os.environ.get(
        "KUBECONFIG", str(Path.home() / ".kube" / "config"),
    )
    try:
        with open(kube_path) as fh:
            kube_config = yaml.safe_load(fh)
    except (FileNotFoundError, yaml.YAMLError):
        return None

    if not isinstance(kube_config, dict):
        return None

    # Determine the active context
    ctx_name = context or kube_config.get("current-context")
    if not ctx_name:
        return None

    # Locate the context entry
    contexts = kube_config.get("contexts", [])
    ctx_entry = next((c for c in contexts if c.get("name") == ctx_name), None)
    if not ctx_entry:
        return None

    cluster_name = ctx_entry.get("context", {}).get("cluster")
    if not cluster_name:
        return None

    # Locate the cluster entry
    clusters = kube_config.get("clusters", [])
    cluster_entry = next((c for c in clusters if c.get("name") == cluster_name), None)
    if not cluster_entry:
        return None

    return cluster_entry.get("cluster", {}).get("proxy-url")  # type: ignore[no-any-return]  # parsed YAML dict


def _cache_key(gke_config: GKEConfig) -> tuple[str, str, str]:
    """Build a hashable cache key from the GKEConfig fields that affect client creation."""
    return (
        gke_config.kubeconfig_path or "",
        gke_config.context or "",
        gke_config.proxy_url or "",
    )


def clear_k8s_client_cache() -> None:
    """Clear the cached Kubernetes API clients.

    Also clears ArgoCD and Argo Rollouts detection caches, which are
    cluster-scoped and may become stale when the process switches contexts.

    Useful in tests or when kubeconfig/credentials change at runtime.
    """
    _CLIENT_CACHE.clear()
    # Lazy imports to avoid circular dependencies (argocd.py imports _clients).
    try:
        from .argocd import _argocd_ns_cache, _crd_exists_cache  # noqa: WPS433
        _argocd_ns_cache.clear()
        _crd_exists_cache.clear()
    except ImportError:
        pass
    try:
        from .argo_rollouts import _rollouts_ns_cache  # noqa: WPS433
        _rollouts_ns_cache.clear()
    except ImportError:
        pass


def ensure_client_initialized(gke_config: GKEConfig) -> None:
    """Initialize the K8s client cache for *gke_config* if not already cached.

    This is a safe, idempotent pre-warming function designed to be called
    **once, sequentially**, before any parallel threads are launched.  It
    eliminates the thread-safety hazard in :func:`_suppress_stderr` — which
    mutates ``sys.stdout`` and OS fd 2 — by ensuring that the client is fully
    constructed and stored in ``_CLIENT_CACHE`` before concurrent execution
    begins.

    If the client is already cached, this is a cheap no-op (a single dict
    lookup).  If client initialization fails (e.g. missing kubeconfig, no
    ``kubernetes`` package installed), the error is logged as a warning and
    silently suppressed — parallel execution will still proceed, and the
    individual tools will surface the error through their normal
    :class:`~vaig.tools.base.ToolResult` mechanism.

    Args:
        gke_config: GKE configuration used to derive the cache key and build
            the K8s clients.  Passed unchanged to :func:`_create_k8s_clients`.
    """
    key = _cache_key(gke_config)
    if key in _CLIENT_CACHE:
        return  # already warm — fast path

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        logger.warning(
            "K8s client pre-warm failed (parallel execution will continue): %s",
            result.output,
        )


class _NonTTYStream:
    """Wrapper around a stream that forces ``isatty()`` to return ``False``.

    The ``kubernetes`` Python library's ``ExecProvider.run()`` checks
    ``sys.stdout.isatty()`` to decide whether to capture stderr via
    ``subprocess.PIPE`` or pass it through as ``sys.stderr``.  When it
    passes through (TTY mode), ``process.communicate()`` returns
    ``(stdout, None)`` and a subsequent ``stderr.strip()`` crashes with
    ``AttributeError: 'NoneType' object has no attribute 'strip'``.

    By wrapping ``sys.stdout`` with this class during kubeconfig loading,
    we force the non-interactive path (``stderr=subprocess.PIPE``) and
    avoid the crash.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def _suppress_stderr() -> Iterator[None]:
    """Suppress stderr at the file-descriptor level and force non-TTY stdout.

    The kubernetes Python client may spawn a Go-based credential helper
    (e.g. ``gke-gcloud-auth-plugin``) whose fatal log messages (``F0315
    cred.go ...``) are written directly to fd 2, bypassing Python's
    ``sys.stderr``.  This context manager redirects the *real* fd 2 to
    ``/dev/null`` so those messages never reach the terminal.

    Additionally, ``sys.stdout`` is temporarily wrapped with
    ``_NonTTYStream`` so that the kubernetes library's ``is_interactive``
    check (``sys.stdout.isatty()``) returns ``False``.  This forces it
    to capture subprocess stderr via ``subprocess.PIPE`` instead of
    passing ``sys.stderr`` directly, which avoids an ``AttributeError``
    when the exec plugin fails and the library tries to call
    ``stderr.strip()`` on ``None``.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    original_stdout = sys.stdout
    try:
        os.dup2(devnull_fd, 2)
        sys.stdout = _NonTTYStream(sys.stdout)
        yield
    finally:
        sys.stdout = original_stdout
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


class _InClusterClient(NamedTuple):
    """Wrapper to distinguish an in-cluster ``ApiClient`` from a ``Configuration``."""

    api_client: Any


def _load_k8s_config(
    gke_config: GKEConfig,
) -> Any | _InClusterClient | ToolResult:
    """Load kubeconfig and return a proxy-aware ``Configuration``, or an in-cluster ``ApiClient``.

    This is the shared config-loading logic used by both ``_create_k8s_clients``
    and ``_get_exec_client``.  It resolves kubeconfig path, context, proxy URL,
    and handles the ``AttributeError`` workaround for broken auth plugins.

    Returns:
        - A ``kubernetes.client.Configuration`` object on success (kubeconfig mode).
        - An ``_InClusterClient`` wrapping an ``ApiClient`` for in-cluster config.
        - A ``ToolResult`` with ``error=True`` on failure.
    """
    try:
        with _suppress_stderr():
            kubeconfig_path = gke_config.kubeconfig_path or None
            context = gke_config.context or None

            # ── Resolve proxy URL ────────────────────────────────
            proxy_url = _extract_proxy_url_from_kubeconfig(kubeconfig_path, context)

            # Explicit GKEConfig override takes precedence
            if gke_config.proxy_url:
                proxy_url = gke_config.proxy_url

            # ── Build a Configuration object with proxy ──────────
            config = k8s_client.Configuration()
            if proxy_url:
                config.proxy = proxy_url
                logger.info("Using proxy URL for K8s API: %s", proxy_url)

            # ── Disable urllib3-level retries globally ────────────
            # The kubernetes Python client passes urllib3's Retry object to each
            # request.  By default urllib3 uses Retry(total=3) which causes ~84s
            # hangs when the API server is unreachable (3 × ~28s TCP timeout).
            # vaig already has its own retry logic (_RETRY_ATTEMPTS), so we
            # disable urllib3 retries to fail fast and let the app-level logic
            # decide when to retry.
            config.retries = False

            # ── Load kubeconfig into the Configuration ───────────
            if kubeconfig_path:
                try:
                    k8s_config.load_kube_config(
                        config_file=kubeconfig_path,
                        context=context,
                        client_configuration=config,
                    )
                except AttributeError as attr_err:
                    if "'NoneType' object has no attribute 'strip'" in str(attr_err):
                        raise RuntimeError(
                            "Kubernetes auth plugin failed. Run "
                            "'gcloud container clusters get-credentials' "
                            "to refresh your credentials.",
                        ) from attr_err
                    raise
            else:
                try:
                    k8s_config.load_kube_config(
                        context=context,
                        client_configuration=config,
                    )
                except AttributeError as attr_err:
                    if "'NoneType' object has no attribute 'strip'" in str(attr_err):
                        raise RuntimeError(
                            "Kubernetes auth plugin failed. Run "
                            "'gcloud container clusters get-credentials' "
                            "to refresh your credentials.",
                        ) from attr_err
                    raise
                except k8s_config.ConfigException:
                    # Fallback to in-cluster config (workload identity).
                    # Return wrapped ApiClient — caller handles this case.
                    k8s_config.load_incluster_config()
                    # Disable retries on the in-cluster path too — the same
                    # ~84s hang risk applies if the API server is unreachable.
                    ic_cfg = k8s_client.Configuration.get_default_copy()
                    ic_cfg.retries = False
                    return _InClusterClient(k8s_client.ApiClient(ic_cfg))

    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            output=f"Failed to configure Kubernetes client: {exc}",
            error=True,
        )

    return config


def _create_k8s_clients(
    gke_config: GKEConfig,
) -> tuple[Any, Any, Any, Any] | ToolResult:
    """Create and configure Kubernetes API clients from GKEConfig.

    Returns a tuple of ``(CoreV1Api, AppsV1Api, CustomObjectsApi, ApiClient)``
    on success, or a ``ToolResult`` with ``error=True`` on failure.

    Results are cached per unique ``(kubeconfig_path, context, proxy_url)``
    combination so that repeated tool invocations within the same session
    reuse the same authenticated clients instead of rebuilding them on
    every call.

    Supports:
    - Explicit ``kubeconfig_path`` + optional ``context``
    - Default kubeconfig (``~/.kube/config``)
    - In-cluster config (for workload identity / GKE pods)

    The ``proxy-url`` field in kubeconfig cluster entries is **not** supported
    by the ``kubernetes`` Python client (v35).  This function works around the
    limitation by parsing the raw YAML, extracting ``proxy-url``, and injecting
    it into ``kubernetes.client.Configuration.proxy``.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    key = _cache_key(gke_config)
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    result = _load_k8s_config(gke_config)
    if isinstance(result, ToolResult):
        return result

    # In-cluster fallback returns a wrapped ApiClient
    if isinstance(result, _InClusterClient):
        api_client_ic = result.api_client
        clients = (
            k8s_client.CoreV1Api(api_client_ic),
            k8s_client.AppsV1Api(api_client_ic),
            k8s_client.CustomObjectsApi(api_client_ic),
            api_client_ic,
        )
        _CLIENT_CACHE[key] = clients
        return clients

    # result is a Configuration — build API clients with proxy-aware config
    api_client = k8s_client.ApiClient(result)
    clients = (
        k8s_client.CoreV1Api(api_client),
        k8s_client.AppsV1Api(api_client),
        k8s_client.CustomObjectsApi(api_client),
        api_client,
    )
    _CLIENT_CACHE[key] = clients
    return clients


# ── Dedicated exec client (NOT cached) ───────────────────────


def _get_exec_client(
    gke_config: GKEConfig,
) -> Any | ToolResult:
    """Create a **fresh, disposable** ``CoreV1Api`` for ``kubernetes.stream.stream()``.

    ``kubernetes.stream.stream()`` mutates the ``ApiClient`` instance it
    receives — it replaces the HTTP request method to set up a WebSocket
    connection.  If the shared (cached) client is passed, subsequent
    non-exec API calls (list pods, get logs, etc.) may fail or behave
    unpredictably because the underlying transport was overwritten.

    This factory intentionally creates a **new** ``ApiClient`` on every
    call and does **not** cache it.  The caller is responsible for closing
    the client after use (``ApiClient`` supports the context-manager
    protocol, or call ``exec_client.api_client.close()`` manually).

    Uses the same kubeconfig / context / proxy resolution logic as
    ``_create_k8s_clients`` (via ``_load_k8s_config``).

    Returns:
        A ``CoreV1Api`` wrapping a fresh ``ApiClient`` on success,
        or a ``ToolResult`` with ``error=True`` on failure.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    result = _load_k8s_config(gke_config)
    if isinstance(result, ToolResult):
        return result

    # In-cluster fallback returns a wrapped ApiClient
    if isinstance(result, _InClusterClient):
        return k8s_client.CoreV1Api(result.api_client)

    # result is a Configuration — build a fresh ApiClient
    api_client = k8s_client.ApiClient(result)
    return k8s_client.CoreV1Api(api_client)


# ══════════════════════════════════════════════════════════════
# DefaultK8sClientProvider — protocol-satisfying wrapper
# ══════════════════════════════════════════════════════════════


class DefaultK8sClientProvider:
    """Default implementation of ``K8sClientProvider`` protocol.

    Delegates to the existing module-level functions and ``_CLIENT_CACHE``.
    This class exists solely to satisfy the ``K8sClientProvider`` protocol
    and provide an injectable, mockable interface for DI.
    """

    __slots__ = ()

    def get_clients(self, gke_config: GKEConfig) -> tuple[Any, Any, Any, Any] | Any:
        """Return cached K8s API clients, delegating to ``_create_k8s_clients``."""
        return _create_k8s_clients(gke_config)

    def get_exec_client(self, gke_config: GKEConfig) -> Any:
        """Return a fresh, disposable ``CoreV1Api`` for exec operations."""
        return _get_exec_client(gke_config)

    def clear_cache(self) -> None:
        """Clear cached Kubernetes clients."""
        clear_k8s_client_cache()


# ── ArgoCD client helper ─────────────────────────────────────


def _create_argocd_client(
    *,
    server: str = "",
    token: str = "",
    context: str = "",
    verify_ssl: bool = True,
) -> tuple[str, Any]:
    """Create an ArgoCD client based on connection mode.

    Three connection modes are supported, resolved in priority order:

    1. **API Server mode** (``server`` + ``token`` provided):
       Connect to the ArgoCD REST API via HTTP.
       *Not yet implemented — raises ``NotImplementedError``.*

    2. **Separate context mode** (``context`` provided):
       Use a different kubeconfig context to access ArgoCD's cluster.
       *Not yet implemented — raises ``NotImplementedError``.*

    3. **Same-cluster mode** (default fallback):
       ArgoCD runs in the same cluster — use ``CustomObjectsApi`` directly.

    Args:
        server: ArgoCD API server URL (e.g. ``https://argocd.example.com``).
        token: ArgoCD API bearer token.
        context: Kubeconfig context name pointing to ArgoCD's cluster.
        verify_ssl: Whether to verify TLS certificates (API mode only).

    Returns:
        Tuple of ``(client_type, client)`` where ``client_type`` is one of
        ``"api"``, ``"context"``, or ``"cluster"``.

    Raises:
        NotImplementedError: For API and context modes (Phase 3).
        RuntimeError: If same-cluster client creation fails.
    """
    if not _K8S_AVAILABLE:
        raise RuntimeError(_K8S_IMPORT_ERROR or "kubernetes SDK not available")

    # Mode 1: API server (stub)
    if server and token:
        raise NotImplementedError(
            "ArgoCD REST API mode is not yet implemented (Phase 3). "
            "Use same-cluster mode by omitting server/token."
        )

    # Mode 2: Separate kubeconfig context (stub)
    if context:
        raise NotImplementedError(
            "ArgoCD separate-context mode is not yet implemented (Phase 3). "
            "Use same-cluster mode by omitting the context parameter."
        )

    # Mode 3: Same-cluster — build a CustomObjectsApi
    try:
        api_client = k8s_client.ApiClient()
        custom_api = k8s_client.CustomObjectsApi(api_client)
        return ("cluster", custom_api)
    except Exception as exc:
        raise RuntimeError(f"Failed to create same-cluster ArgoCD client: {exc}") from exc
