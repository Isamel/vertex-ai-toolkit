"""MigrationConfig and auto-detection of source kind."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, model_validator


def detect_source_kind(from_dir: str) -> str:
    """Auto-detect the source kind from files in *from_dir*.

    Priority order:
    1. Any ``.ktr`` or ``.kjb`` file  → ``"pentaho"``
    2. Any ``.py`` file containing ``from airflow``  → ``"airflow-v1"``
    3. Any ``.sql`` file  → ``"generic-sql"``
    4. Default  → ``"unknown"``
    """
    root = Path(from_dir)
    if not root.is_dir():
        return "unknown"

    has_sql = False
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in {".ktr", ".kjb"}:
                return "pentaho"
            if ext == ".sql":
                has_sql = True
            if ext == ".py":
                try:
                    content = (Path(dirpath) / fname).read_text(encoding="utf-8", errors="replace")
                    if "from airflow" in content:
                        return "airflow-v1"
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:  # noqa: BLE001
                    pass

    if has_sql:
        return "generic-sql"

    return "unknown"


class MigrationConfig(BaseModel):
    """Configuration for a code migration run."""

    from_dir: str = "."
    to_dir: str = "./migrated"
    examples_dir: str | None = None
    source_kind: str | None = None  # None → auto-detected on init
    target_kind: str = "aws-glue-pyspark"
    design_principles: list[str] = []
    max_migration_iterations: int = 20
    migration_budget_usd: float = 10.0

    @model_validator(mode="after")
    def _auto_detect_source_kind(self) -> MigrationConfig:
        """If source_kind was not provided, detect it from from_dir."""
        if self.source_kind is None:
            self.source_kind = detect_source_kind(self.from_dir)
        return self
