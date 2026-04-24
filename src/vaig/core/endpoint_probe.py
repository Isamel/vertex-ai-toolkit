"""Vertex AI global-endpoint probe and cache (SPEC-GEP-01, SPEC-GEP-02).

Small, dependency-light helper used by :class:`~vaig.core.client.GeminiClient`
to decide between the Vertex AI **global** endpoint
(``aiplatform.googleapis.com``) and a **regional** endpoint at startup.

The module is intentionally self-contained: it only knows how to

1. Read a persistent probe result from ``~/.vaig/cache/endpoint-probe.json``
   keyed by ``(project_id, credentials_hash)`` with a configurable TTL.
2. Perform a short, cheap probe call against the global endpoint when no
   fresh cached answer is available.
3. Write the result back to the cache.

See ``docs/specs/global-endpoint-v1.md`` for design notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

    from vaig.core.config import GCPConfig

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "endpoint-probe.json"


def _cache_dir() -> Path:
    """Return the directory used to store the probe cache.

    Honours ``VAIG_CACHE_DIR`` when set; otherwise defaults to
    ``~/.vaig/cache``. Missing directories are created lazily.
    """
    override = os.environ.get("VAIG_CACHE_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".vaig" / "cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _credentials_fingerprint(credentials: Credentials | None) -> str:
    """Return a short, non-reversible fingerprint of *credentials*.

    The cache key must change when the user switches credentials (different
    service account, impersonation, etc.) so a stale ``True`` result does
    not leak from one identity to another.  We hash whatever identifying
    attribute the credentials object exposes; when none of the expected
    attributes is present we fall back to hashing the credentials class
    name so the fingerprint is still deterministic. Returns an empty
    string only when *credentials* is ``None``.
    """
    if credentials is None:
        return ""
    parts: list[str] = []
    for attr in ("service_account_email", "client_id", "quota_project_id", "account"):
        val = getattr(credentials, attr, None)
        if val:
            parts.append(f"{attr}={val}")
    if not parts:
        parts.append(type(credentials).__name__)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class _CacheEntry:
    project_id: str
    credentials_hash: str
    global_available: bool
    probed_at: float


def _load_cache_entry(cache_path: Path) -> _CacheEntry | None:
    """Return the cached probe entry or ``None`` if the file is missing/corrupt."""
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return _CacheEntry(
            project_id=raw["project_id"],
            credentials_hash=raw["credentials_hash"],
            global_available=bool(raw["global_available"]),
            probed_at=float(raw["probed_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Endpoint probe cache corrupt (%s) — ignoring", exc)
        return None


def _write_cache_entry(cache_path: Path, entry: _CacheEntry) -> None:
    """Persist *entry* atomically at ``cache_path``."""
    data = {
        "project_id": entry.project_id,
        "credentials_hash": entry.credentials_hash,
        "global_available": entry.global_available,
        "probed_at": entry.probed_at,
    }
    try:
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Tighten permissions: user-read/write only.
        try:
            os.chmod(tmp, 0o600)  # noqa: S103 — exact mode intended
        except OSError:
            pass  # Windows or non-POSIX filesystem
        tmp.replace(cache_path)
    except OSError as exc:
        logger.debug("Could not write endpoint probe cache: %s", exc)


def _probe_global_endpoint(
    *,
    project_id: str,
    credentials: Credentials | None,
    timeout_s: float,
) -> bool:
    """Return ``True`` if the Vertex AI global endpoint is reachable.

    Executes a short-lived ``models.list`` call against ``location="global"``.
    Any successful response (2xx) returns ``True``. ``404``/``403``/timeouts
    return ``False`` (the project does not have the global endpoint enabled
    or refuses the call). All other unexpected errors are logged and treated
    as unavailable so startup never blocks on the probe.
    """
    try:
        from google import genai
        from google.genai import types as _types
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.debug("google-genai SDK not importable — skipping global probe")
        return False

    start = time.monotonic()
    try:
        probe_client = genai.Client(
            vertexai=True,
            project=project_id,
            location="global",
            credentials=credentials,
            http_options=_types.HttpOptions(
                timeout=int(timeout_s * 1000),
            ),
        )
        # ``models.list`` is an iterator; consume only the first page so we
        # do not download the full catalog.
        it = probe_client.models.list(config={"page_size": 1})
        next(iter(it), None)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "Global Vertex AI endpoint unavailable (%s: %s) — will use regional fallback",
            type(exc).__name__,
            exc,
        )
        return False

    elapsed = time.monotonic() - start
    logger.info(
        "Global Vertex AI endpoint reachable (probe %.2fs) — using global",
        elapsed,
    )
    return True


def resolve_endpoint_location(
    gcp_cfg: GCPConfig,
    credentials: Credentials | None,
) -> str:
    """Pick the endpoint location for a fresh :class:`GeminiClient`.

    Honours ``GCPConfig.endpoint_mode`` and caches the probe result in
    ``~/.vaig/cache/endpoint-probe.json`` (SPEC-GEP-02).

    Returns:
        One of:
          * ``"global"`` — the Vertex AI global endpoint is reachable and
            was selected (either explicitly via ``endpoint_mode='global'``
            or auto-probed in ``'auto'`` mode).
          * ``gcp_cfg.location`` — a user-pinned region (honoured in
            ``'regional'`` mode or in ``'auto'`` mode when it's set to a
            non-``"global"`` value).
          * ``gcp_cfg.fallback_location`` (or ``"us-central1"`` when that
            is unset) — the regional fallback used when no specific region
            is pinned or the global probe fails.

    Raises:
        RuntimeError: when ``endpoint_mode="global"`` but the probe fails.
            In every other case the function degrades gracefully to the
            regional fallback.
    """
    mode = gcp_cfg.endpoint_mode

    # Explicit modes bypass the probe entirely.
    if mode == "regional":
        # Prefer the user-specified location; only fall back when it's empty
        # or the sentinel "global" (which is invalid for regional mode).
        if gcp_cfg.location and gcp_cfg.location != "global":
            chosen = gcp_cfg.location
        else:
            chosen = gcp_cfg.fallback_location or "us-central1"
        logger.debug("endpoint_mode='regional' — using %s", chosen)
        return chosen

    if mode == "global":
        # Fail-hard mode: no probe shortcut, but we still run it to produce
        # an actionable error if the endpoint is unreachable.
        if _probe_global_endpoint(
            project_id=gcp_cfg.project_id,
            credentials=credentials,
            timeout_s=gcp_cfg.global_probe_timeout_s,
        ):
            return "global"
        raise RuntimeError(
            "Global Vertex AI endpoint requested (gcp.endpoint_mode='global') "
            "but unreachable. Switch to 'auto' or 'regional', or enable the "
            "global endpoint for your project."
        )

    # mode == "auto": if user already pinned a specific region, use it.
    if gcp_cfg.location and gcp_cfg.location != "global":
        logger.debug(
            "endpoint_mode='auto' but location='%s' is a region — honouring it",
            gcp_cfg.location,
        )
        return gcp_cfg.location

    # Auto-probe path. Consult the cache first.
    creds_hash = _credentials_fingerprint(credentials)
    cache_path = _cache_dir() / _CACHE_FILENAME
    entry = _load_cache_entry(cache_path)
    now = time.time()

    if (
        entry is not None
        and entry.project_id == gcp_cfg.project_id
        and entry.credentials_hash == creds_hash
        and gcp_cfg.global_probe_cache_ttl_s > 0
        and (now - entry.probed_at) < gcp_cfg.global_probe_cache_ttl_s
    ):
        chosen = "global" if entry.global_available else (gcp_cfg.fallback_location or "us-central1")
        logger.debug(
            "Endpoint probe cache hit: global_available=%s, age=%.0fs, using=%s",
            entry.global_available,
            now - entry.probed_at,
            chosen,
        )
        return chosen

    # Cache miss — probe now.
    logger.debug("Endpoint probe cache miss — probing global endpoint")
    available = _probe_global_endpoint(
        project_id=gcp_cfg.project_id,
        credentials=credentials,
        timeout_s=gcp_cfg.global_probe_timeout_s,
    )

    _write_cache_entry(
        cache_path,
        _CacheEntry(
            project_id=gcp_cfg.project_id,
            credentials_hash=creds_hash,
            global_available=available,
            probed_at=now,
        ),
    )

    return "global" if available else (gcp_cfg.fallback_location or "us-central1")


def invalidate_probe_cache() -> None:
    """Remove the probe cache file. Next call to :func:`resolve_endpoint_location`
    with ``endpoint_mode='auto'`` will re-probe."""
    cache_path = _cache_dir() / _CACHE_FILENAME
    try:
        cache_path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem quirks
        logger.debug("Could not invalidate probe cache: %s", exc)
