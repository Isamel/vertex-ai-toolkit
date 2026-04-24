"""Attachment cache and session persistence (SPEC-ATT-08).

Classes:

- :class:`AttachmentCache`   – content-addressed file cache for processed attachments
- :class:`AttachmentSession` – persistent per-session attachment list
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("vaig.core.attachment_cache")

# ── AttachmentCache ───────────────────────────────────────────────────────────


class AttachmentCache:
    """Content-addressed disk cache for processed attachments.

    Cache layout::

        {cache_dir}/
            {fingerprint[:2]}/
                {key}.manifest.json
                {key}.chunks.json
                {key}.meta.json

    where ``key = sha256(fingerprint + ":" + config_hash)[:64]``.

    Files are written with mode 0600 (SA-7).  Any JSON corruption or missing
    file silently returns ``None`` (cache miss) — never a hard error.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        ttl_seconds: int = 86400,
        config_hash: str,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._ttl_seconds = ttl_seconds
        self._config_hash = config_hash

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, fingerprint: str) -> tuple[list[Any], list[Any]] | None:
        """Return ``(manifest, chunks)`` for *fingerprint* or ``None`` on miss/expiry."""
        key = self._make_key(fingerprint)
        manifest_path, chunks_path, meta_path = self._paths(key)

        try:
            meta_text = meta_path.read_text(encoding="utf-8")
            meta = json.loads(meta_text)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("attachment_cache: meta read miss for %s: %s", key, exc)
            return None

        # Config hash mismatch → stale
        if meta.get("config_hash") != self._config_hash:
            logger.debug("attachment_cache: config_hash mismatch for %s", key)
            return None

        # TTL check
        if self._is_expired(meta):
            logger.debug("attachment_cache: TTL expired for %s", key)
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("attachment_cache: data read miss for %s: %s", key, exc)
            return None

        return manifest, chunks

    def put(
        self,
        fingerprint: str,
        manifest: list[Any],
        chunks: list[Any],
        *,
        adapter_spec: dict[str, Any] | None = None,
    ) -> None:
        """Store *(manifest, chunks)* for *fingerprint*."""
        key = self._make_key(fingerprint)
        manifest_path, chunks_path, meta_path = self._paths(key)

        # Ensure the shard directory exists
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "config_hash": self._config_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "fingerprint": fingerprint,
        }
        if adapter_spec:
            meta["adapter_spec"] = adapter_spec

        _write_json_safe(meta_path, meta)
        _write_json_safe(manifest_path, manifest)
        _write_json_safe(chunks_path, chunks)

    def invalidate(self, fingerprint: str) -> bool:
        """Remove cached entry for *fingerprint*. Returns True if anything was removed."""
        key = self._make_key(fingerprint)
        removed = False
        for path in self._paths(key):
            try:
                path.unlink(missing_ok=True)
                removed = True
            except OSError:
                pass
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, fingerprint: str) -> str:
        """Derive cache key from fingerprint + config_hash."""
        import hashlib

        raw = f"{fingerprint}:{self._config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        """Return (manifest, chunks, meta) paths for *key*."""
        shard = self._cache_dir / key[:2]
        return (
            shard / f"{key}.manifest.json",
            shard / f"{key}.chunks.json",
            shard / f"{key}.meta.json",
        )

    def _is_expired(self, meta: dict[str, Any]) -> bool:
        """Return True if the cache entry has exceeded its TTL."""
        created_str = meta.get("created_at", "")
        if not created_str:
            return True
        try:
            created = datetime.fromisoformat(created_str)
            age = (datetime.now(UTC) - created).total_seconds()
            return age > self._ttl_seconds
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001
            return True


def _write_json_safe(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* with mode 0600 (SA-7)."""
    text = json.dumps(data, ensure_ascii=False, indent=None)
    path.write_text(text, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ── AttachmentSession ─────────────────────────────────────────────────────────


class SessionAttachment:
    """In-memory record for one attachment entry in a session."""

    __slots__ = ("source", "name", "fingerprint", "kind", "added_at")

    def __init__(
        self,
        *,
        source: str,
        name: str | None,
        fingerprint: str,
        kind: str,
        added_at: str,
    ) -> None:
        self.source = source
        self.name = name
        self.fingerprint = fingerprint
        self.kind = kind
        self.added_at = added_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionAttachment:
        return cls(
            source=str(d.get("source", "")),
            name=d.get("name"),
            fingerprint=str(d.get("fingerprint", "")),
            kind=str(d.get("kind", "")),
            added_at=str(d.get("added_at", "")),
        )


class AttachmentSession:
    """Persistent per-session attachment list.

    File path: ``{session_dir}/{session_id}.json``

    Content schema::

        {
            "attachments": [
                {
                    "source": str,
                    "name": str | null,
                    "fingerprint": str,
                    "kind": str,
                    "added_at": "<iso8601>"
                }
            ]
        }

    Merge semantics: adding a source already present updates ``added_at``
    without creating a duplicate.
    """

    def __init__(self, session_dir: Path, session_id: str) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = session_id
        self._path = self._session_dir / f"{session_id}.json"
        self._attachments: list[SessionAttachment] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> list[SessionAttachment]:
        """Load attachments from disk. Returns empty list if file is absent."""
        if not self._path.exists():
            self._attachments = []
            return self._attachments

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._attachments = [SessionAttachment.from_dict(entry) for entry in data.get("attachments", [])]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("attachment_session: load failed for %s: %s", self._path, exc)
            self._attachments = []

        return self._attachments

    def add(
        self,
        *,
        source: str,
        name: str | None,
        fingerprint: str,
        kind: str,
    ) -> None:
        """Add or update an attachment entry (merge-not-replace by source)."""
        now = datetime.now(UTC).isoformat()
        for entry in self._attachments:
            if entry.source == source:
                entry.fingerprint = fingerprint
                entry.name = name
                entry.kind = kind
                entry.added_at = now
                return
        self._attachments.append(
            SessionAttachment(
                source=source,
                name=name,
                fingerprint=fingerprint,
                kind=kind,
                added_at=now,
            )
        )

    def save(self) -> None:
        """Persist the current attachment list to disk."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "attachments": [a.to_dict() for a in self._attachments],
        }
        _write_json_safe(self._path, data)
