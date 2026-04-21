"""Migration configuration model and source-kind auto-detection."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["MigrationConfig", "detect_source_kind"]


DesignPrinciple = Literal["ddd", "tdd", "sdd", "hex-arch", "clean-arch", "srp"]


def detect_source_kind(from_dirs: list[Path]) -> str:
    """Auto-detect source kind from file extensions and content (first match wins).

    Priority order:
      pentaho → ssis → informatica → talend → airflow-v1 → generic-sql → cobol → legacy-java → generic
    """
    all_files: list[Path] = []
    for d in from_dirs:
        if d.is_dir():
            all_files.extend(d.rglob("*"))

    exts = {f.suffix.lower() for f in all_files if f.is_file()}
    names = {f.name.lower() for f in all_files if f.is_file()}

    # pentaho
    if ".ktr" in exts or ".kjb" in exts:
        return "pentaho"

    # ssis
    if ".dtsx" in exts:
        return "ssis"

    # informatica (xml with <workflow and Informatica in root)
    if ".xml" in exts:
        for f in all_files:
            if f.suffix.lower() == ".xml":
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")[:2000]
                    if "<workflow" in text and "Informatica" in text:
                        return "informatica"
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    pass

    # talend
    if any(n.startswith("talend-") and n.endswith(".jar") for n in names):
        return "talend"

    # airflow-v1
    for f in all_files:
        if f.suffix.lower() == ".py":
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if "from airflow import DAG" in text or "from airflow.models import DAG" in text:
                    return "airflow-v1"
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass

    # generic-sql (sql without py siblings)
    py_files = [f for f in all_files if f.suffix.lower() == ".py"]
    sql_files = [f for f in all_files if f.suffix.lower() == ".sql"]
    if sql_files and not py_files:
        return "generic-sql"

    # cobol
    if ".cbl" in exts or ".cob" in exts or ".COB" in {f.suffix for f in all_files if f.is_file()}:
        return "cobol"

    # legacy-java
    if any(n in names for n in ("pom.xml", "build.gradle", "build.gradle.kts")):
        return "legacy-java"

    return "generic"


class MigrationConfig(BaseModel):
    """Configuration for a migration run."""

    model_config = ConfigDict(extra="forbid")

    from_dirs: list[Path] = Field(default_factory=list)
    to_dir: Path = Field(default=Path("./migrated"))
    examples_dirs: list[Path] = Field(default_factory=list)
    source_kind: str | None = None  # None → auto-detected
    target_kind: str = "aws-glue-pyspark"
    design_principles: list[DesignPrinciple] = Field(
        default_factory=lambda: ["tdd", "sdd"]  # type: ignore[arg-type]
    )
    max_migration_iterations: int = 20
    migration_budget_usd: float = 10.0
    budget_warn_fraction: float = 0.8
    resume_from_state: bool = True

    @model_validator(mode="after")
    def _validate_and_autodetect(self) -> Self:
        # Validate from_dirs exist
        for d in self.from_dirs:
            if not d.is_dir():
                raise ValueError(f"--from-dir does not exist or is not a directory: {d}")
        # Validate to_dir does not overlap from_dirs
        for d in self.from_dirs:
            try:
                d.resolve().relative_to(self.to_dir.resolve())
                raise ValueError(f"--to-dir must NOT overlap --from-dir: {d}")
            except (KeyboardInterrupt, SystemExit):
                raise
            except ValueError as e:
                if "--to-dir must NOT overlap" in str(e):
                    raise
                # relative_to raises ValueError when paths don't overlap — that's fine
            try:
                self.to_dir.resolve().relative_to(d.resolve())
                raise ValueError(f"--to-dir must NOT overlap --from-dir: {d}")
            except (KeyboardInterrupt, SystemExit):
                raise
            except ValueError as e:
                if "--to-dir must NOT overlap" in str(e):
                    raise
        # Auto-detect source kind
        if self.source_kind is None and self.from_dirs:
            self.source_kind = detect_source_kind(self.from_dirs)
        return self
