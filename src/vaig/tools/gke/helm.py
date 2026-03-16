"""Helm release introspection tools — list, status, history, and values for Helm releases."""

from __future__ import annotations

import base64
import gzip
import json
import logging
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

from . import _cache, _clients

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433, F401
except ImportError:
    _K8S_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────
_HELM_CACHE_TTL: int = 60  # seconds (matches _DISCOVERY_TTL)
_HELM_SECRET_LABEL_SELECTOR = "owner=helm"
_HELM_SECRET_TYPE = "helm.sh/release.v1"
_HELM_SECRET_NAME_PREFIX = "sh.helm.release.v1."


# ── Helpers ──────────────────────────────────────────────────


def _decode_helm_release(release_data: str) -> dict:
    """Decode a Helm release from a K8s secret.

    Helm stores releases as: base64 → gzip → JSON.

    Note: The kubernetes Python client already base64-decodes the secret
    ``.data`` fields, so we only need ONE base64 decode here (Helm's own
    encoding), then gzip decompress, then JSON parse.
    """
    # Single base64 decode (Helm's own encoding — K8s client already did the first)
    decoded = base64.b64decode(release_data)
    # Gzip decompress
    decompressed = gzip.decompress(decoded)
    # Parse JSON
    return json.loads(decompressed)


def _extract_release_info(secret: Any) -> dict[str, Any] | None:
    """Extract basic release info from a Helm secret's name and labels.

    Returns a dict with ``name``, ``revision``, ``status``, ``chart``,
    ``app_version``, or ``None`` if the secret doesn't look like a Helm release.
    """
    metadata = secret.metadata
    if not metadata:
        return None

    secret_name = metadata.name or ""
    if not secret_name.startswith(_HELM_SECRET_NAME_PREFIX):
        return None

    labels = metadata.labels or {}

    # Extract revision from the secret name: sh.helm.release.v1.<name>.v<revision>
    suffix = secret_name[len(_HELM_SECRET_NAME_PREFIX):]
    # suffix = "<release-name>.v<revision>"
    parts = suffix.rsplit(".v", 1)
    if len(parts) != 2:
        return None

    release_name = parts[0]
    try:
        revision = int(parts[1])
    except (ValueError, IndexError):
        return None

    return {
        "name": release_name,
        "revision": revision,
        "status": labels.get("status", "unknown"),
        "chart": labels.get("chart", "unknown"),
        "app_version": labels.get("app_version", ""),
        "version": labels.get("version", ""),
    }


def _find_release_secrets(
    core_v1: Any,
    release_name: str,
    namespace: str,
) -> list[Any]:
    """Find all Helm secrets for a specific release, sorted by revision (descending)."""
    try:
        secrets = core_v1.list_namespaced_secret(
            namespace=namespace,
            label_selector=_HELM_SECRET_LABEL_SELECTOR,
        )
    except Exception as exc:
        logger.warning("Error listing Helm secrets in %s: %s", namespace, exc)
        return []

    matching: list[tuple[int, Any]] = []
    for secret in secrets.items or []:
        info = _extract_release_info(secret)
        if info and info["name"] == release_name:
            matching.append((info["revision"], secret))

    # Sort by revision descending (newest first)
    matching.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in matching]


def _decode_secret_release_data(secret: Any) -> dict[str, Any] | None:
    """Decode the full Helm release data from a secret's .data['release'] field."""
    if not secret.data or "release" not in secret.data:
        return None

    try:
        return _decode_helm_release(secret.data["release"])
    except Exception as exc:
        logger.warning("Failed to decode Helm release data: %s", exc)
        return None


# ── Formatters ───────────────────────────────────────────────


def _format_releases_table(releases: list[dict[str, Any]]) -> str:
    """Format a list of Helm releases as a table."""
    if not releases:
        return "No Helm releases found."

    lines: list[str] = []
    lines.append(
        f"  {'NAME':<30} {'CHART':<30} {'VERSION':<10} {'STATUS':<12} {'APP VERSION':<15}"
    )
    lines.append("  " + "-" * 97)

    for rel in releases:
        name = rel["name"][:29]
        chart = rel["chart"][:29]
        version = str(rel.get("version", ""))[:9]
        status = rel["status"][:11]
        app_version = rel.get("app_version", "")[:14]
        lines.append(
            f"  {name:<30} {chart:<30} {version:<10} {status:<12} {app_version:<15}"
        )

    return "\n".join(lines)


def _format_history_table(revisions: list[dict[str, Any]]) -> str:
    """Format revision history as a table."""
    if not revisions:
        return "No revision history found."

    lines: list[str] = []
    lines.append(
        f"  {'REVISION':<10} {'STATUS':<14} {'CHART':<30} {'APP VERSION':<15} {'DESCRIPTION':<30} {'UPDATED':<25}"
    )
    lines.append("  " + "-" * 124)

    for rev in revisions:
        revision = str(rev["revision"])[:9]
        status = rev["status"][:13]
        chart = rev.get("chart", "")[:29]
        app_version = rev.get("app_version", "")[:14]
        description = rev.get("description", "")[:29]
        updated = rev.get("updated", "")[:24]
        lines.append(
            f"  {revision:<10} {status:<14} {chart:<30} {app_version:<15} {description:<30} {updated:<25}"
        )

    return "\n".join(lines)


# ── Public Tool Functions ────────────────────────────────────


def helm_list_releases(
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    force_refresh: bool = False,
) -> ToolResult:
    """List all Helm releases in a namespace.

    Queries Kubernetes secrets with ``owner=helm`` label selector and extracts
    release name, chart, version, status, and app_version from the secret
    metadata. Only shows the latest revision for each release.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("helm_releases", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _apps_v1, _custom_api, _api_client = result

    sections: list[str] = [
        "=== Helm Releases ===",
        f"Namespace: {namespace}",
        "",
    ]

    try:
        secrets = core_v1.list_namespaced_secret(
            namespace=namespace,
            label_selector=_HELM_SECRET_LABEL_SELECTOR,
        )
    except Exception as exc:
        msg = f"Error listing Helm secrets: {exc}"
        logger.warning(msg)
        return ToolResult(output=msg, error=True)

    # Group by release name, keeping only the latest revision
    latest: dict[str, dict[str, Any]] = {}
    for secret in secrets.items or []:
        info = _extract_release_info(secret)
        if not info:
            continue
        name = info["name"]
        if name not in latest or info["revision"] > latest[name]["revision"]:
            latest[name] = info

    releases = sorted(latest.values(), key=lambda r: r["name"])
    sections.append(_format_releases_table(releases))
    sections.append("")
    sections.append(f"Total releases: {len(releases)}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def helm_release_status(
    *,
    gke_config: GKEConfig,
    release_name: str,
    namespace: str = "default",
    force_refresh: bool = False,
) -> ToolResult:
    """Get detailed status of a specific Helm release.

    Finds the latest revision secret, decodes the release data, and extracts
    status, chart, version, app_version, first_deployed, last_deployed,
    description, and notes.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("helm_release", namespace, release_name)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _apps_v1, _custom_api, _api_client = result

    # Find the latest revision secret
    release_secrets = _find_release_secrets(core_v1, release_name, namespace)
    if not release_secrets:
        return ToolResult(
            output=f"Helm release '{release_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    latest_secret = release_secrets[0]
    release_data = _decode_secret_release_data(latest_secret)

    sections: list[str] = [
        f"=== Helm Release: {release_name} ===",
        f"Namespace: {namespace}",
        "",
    ]

    if release_data:
        info = release_data.get("info", {})
        chart_meta = release_data.get("chart", {}).get("metadata", {})

        sections.append(f"Status: {info.get('status', 'unknown')}")
        sections.append(f"Chart: {chart_meta.get('name', 'unknown')}-{chart_meta.get('version', '?')}")
        sections.append(f"App Version: {chart_meta.get('appVersion', 'unknown')}")
        sections.append(f"Revision: {release_data.get('version', '?')}")
        sections.append(f"First Deployed: {info.get('first_deployed', 'unknown')}")
        sections.append(f"Last Deployed: {info.get('last_deployed', 'unknown')}")
        sections.append(f"Description: {info.get('description', '')}")

        notes = info.get("notes", "")
        if notes:
            sections.append("")
            sections.append("--- Notes ---")
            sections.append(notes)
    else:
        # Fallback to label-based info
        info = _extract_release_info(latest_secret) or {}
        sections.append(f"Status: {info.get('status', 'unknown')}")
        sections.append(f"Chart: {info.get('chart', 'unknown')}")
        sections.append("(Could not decode full release data)")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def helm_release_history(
    *,
    gke_config: GKEConfig,
    release_name: str,
    namespace: str = "default",
    force_refresh: bool = False,
) -> ToolResult:
    """Get revision history of a Helm release.

    Finds all secrets for this release (all revisions) and for each one
    extracts: revision number, status, chart version, app_version,
    description, and updated timestamp. Returns sorted by revision
    (newest first).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("helm_history", namespace, release_name)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _apps_v1, _custom_api, _api_client = result

    release_secrets = _find_release_secrets(core_v1, release_name, namespace)
    if not release_secrets:
        return ToolResult(
            output=f"Helm release '{release_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    sections: list[str] = [
        f"=== Helm Release History: {release_name} ===",
        f"Namespace: {namespace}",
        "",
    ]

    revisions: list[dict[str, Any]] = []
    for secret in release_secrets:
        info = _extract_release_info(secret) or {}
        release_data = _decode_secret_release_data(secret)

        rev_entry: dict[str, Any] = {
            "revision": info.get("revision", 0),
            "status": info.get("status", "unknown"),
            "chart": info.get("chart", "unknown"),
            "app_version": info.get("app_version", ""),
            "description": "",
            "updated": "",
        }

        if release_data:
            rel_info = release_data.get("info", {})
            chart_meta = release_data.get("chart", {}).get("metadata", {})
            rev_entry["description"] = rel_info.get("description", "")
            rev_entry["updated"] = rel_info.get("last_deployed", "")
            rev_entry["app_version"] = chart_meta.get("appVersion", rev_entry["app_version"])
            chart_name = chart_meta.get("name", "")
            chart_version = chart_meta.get("version", "")
            if chart_name and chart_version:
                rev_entry["chart"] = f"{chart_name}-{chart_version}"

        revisions.append(rev_entry)

    # Already sorted newest-first by _find_release_secrets
    sections.append(_format_history_table(revisions))
    sections.append("")
    sections.append(f"Total revisions: {len(revisions)}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def helm_release_values(
    *,
    gke_config: GKEConfig,
    release_name: str,
    namespace: str = "default",
    all_values: bool = False,
    force_refresh: bool = False,
) -> ToolResult:
    """Get the values used in a Helm release.

    If ``all_values`` is ``True``, returns computed values (chart defaults
    merged with user overrides). If ``False`` (default), returns only
    user-supplied overrides (the ``config`` field).

    Output is formatted as YAML.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    values_type = "all" if all_values else "overrides"
    cache_key = _cache._cache_key_discovery(
        "helm_values", namespace, release_name, values_type,
    )
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _apps_v1, _custom_api, _api_client = result

    release_secrets = _find_release_secrets(core_v1, release_name, namespace)
    if not release_secrets:
        return ToolResult(
            output=f"Helm release '{release_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    latest_secret = release_secrets[0]
    release_data = _decode_secret_release_data(latest_secret)
    if not release_data:
        return ToolResult(
            output=f"Could not decode release data for '{release_name}'.",
            error=True,
        )

    sections: list[str] = [
        f"=== Helm Release Values: {release_name} ===",
        f"Namespace: {namespace}",
        f"Type: {'all (computed)' if all_values else 'user overrides'}",
        "",
    ]

    if all_values:
        # Merge chart defaults with user overrides
        chart_defaults = release_data.get("chart", {}).get("values", {}) or {}
        user_overrides = release_data.get("config", {}) or {}
        merged = {**chart_defaults, **user_overrides}
        values = merged
    else:
        # User overrides only
        values = release_data.get("config", {}) or {}

    if not values:
        sections.append("(no values)")
    else:
        # Format as YAML-like output using json for simplicity
        # (yaml is not always available, json.dumps with indent is close enough)
        try:
            import yaml as _yaml
            sections.append(_yaml.dump(values, default_flow_style=False, sort_keys=True))
        except ImportError:
            sections.append(json.dumps(values, indent=2, sort_keys=True))

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


# ── Async wrappers ───────────────────────────────────────────
# Offload blocking kubernetes-client calls to a thread pool via to_async.

from vaig.core.async_utils import to_async  # noqa: E402

async_helm_list_releases = to_async(helm_list_releases)
async_helm_release_status = to_async(helm_release_status)
async_helm_release_history = to_async(helm_release_history)
async_helm_release_values = to_async(helm_release_values)
