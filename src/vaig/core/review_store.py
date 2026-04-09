"""Local persistence for ReportReview JSON — one file per run_id.

Stores each :class:`ReportReview` as a single JSON file (mutable,
not JSONL) in ``~/.vaig/reviews/``.  Atomic writes via
``tempfile.NamedTemporaryFile`` + ``os.replace()`` prevent corruption.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from vaig.skills.service_health.schema import ReportReview, ReviewStatus

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".vaig" / "reviews"

# Only allow safe characters in run_id to prevent path traversal.
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

__all__ = ["ReviewStore"]


class ReviewStore:
    """Mutable JSON store for report reviews — one file per *run_id*.

    Storage layout::

        {base_dir}/{run_id}.json

    Unlike :class:`ReportStore` (append-only JSONL), reviews are
    **mutable** (approve/reject overwrites the file).  Atomic writes
    via ``tempfile → os.replace`` guarantee all-or-nothing updates.
    """

    def __init__(self, base_dir: Path = _DEFAULT_DIR) -> None:
        self._base_dir = base_dir

        # SECURITY: Validate that the base directory is within a safe
        # location — mirrors the check in ReportStore.
        resolved = self._base_dir.resolve()
        home = Path.home()
        cwd = Path.cwd()
        tmp = Path(tempfile.gettempdir()).resolve()

        if cwd == Path("/"):
            safe = (
                (resolved == home or home in resolved.parents)
                or (resolved == tmp or tmp in resolved.parents)
            )
        else:
            safe = (
                (resolved == home or home in resolved.parents)
                or (resolved == cwd or cwd in resolved.parents)
                or (resolved == tmp or tmp in resolved.parents)
            )
        if not safe:
            raise ValueError(
                f"base_dir must be under home ({home}), cwd ({cwd}), "
                f"or temp ({tmp}), got: {resolved}"
            )

        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _SAFE_RUN_ID_RE.match(run_id):
            raise ValueError(
                f"run_id contains unsafe characters "
                f"(must match [A-Za-z0-9_-]+): {run_id!r}"
            )

    def _path_for(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self._base_dir / f"{run_id}.json"

    def _atomic_write(self, path: Path, data: str) -> None:
        """Write *data* to *path* atomically via temp-file + rename."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._base_dir,
            suffix=".tmp",
            delete=False,
        ) as fd:
            tmp_name = fd.name
            try:
                fd.write(data)
                fd.flush()
                os.fsync(fd.fileno())
            except BaseException:
                # Best-effort cleanup of the temp file.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        os.replace(tmp_name, path)

    # ── public API ────────────────────────────────────────────

    def save(self, review: ReportReview) -> Path:
        """Persist a new review.  Overwrites if one already exists.

        Args:
            review: The :class:`ReportReview` to persist.

        Returns:
            Path to the JSON file that was written.
        """
        path = self._path_for(review.run_id)
        self._atomic_write(
            path,
            review.model_dump_json(indent=2),
        )
        return path

    def update(self, run_id: str, review: ReportReview) -> Path:
        """Update an existing review (alias for :meth:`save`)."""
        if review.run_id != run_id:
            raise ValueError(
                f"run_id mismatch: path says {run_id!r}, "
                f"review says {review.run_id!r}"
            )
        return self.save(review)

    def get_by_run_id(self, run_id: str) -> ReportReview | None:
        """Load the review for *run_id*, or ``None`` if not found."""
        path = self._path_for(run_id)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            return ReportReview.model_validate_json(raw)
        except (json.JSONDecodeError, ValueError, PydanticValidationError, OSError) as exc:
            logger.warning("Corrupt review file %s: %s", path, exc)
            return None

    def list_reviews(
        self, *, status: ReviewStatus | None = None
    ) -> list[ReportReview]:
        """Return all reviews, optionally filtered by *status*.

        Results are sorted by modification time (newest first).
        """

        def _safe_mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        files = sorted(
            self._base_dir.glob("*.json"),
            key=_safe_mtime,
            reverse=True,
        )
        reviews: list[ReportReview] = []
        for f in files:
            try:
                raw = f.read_text(encoding="utf-8")
                review = ReportReview.model_validate_json(raw)
                if status is None or review.status == status:
                    reviews.append(review)
            except (json.JSONDecodeError, ValueError, PydanticValidationError, OSError) as exc:
                logger.warning("Skipping corrupt review %s: %s", f, exc)
        return reviews

    def is_approved(self, run_id: str) -> bool:
        """Return ``True`` only if a review exists and is approved."""
        review = self.get_by_run_id(run_id)
        if review is None:
            return False
        return review.status == ReviewStatus.APPROVED
