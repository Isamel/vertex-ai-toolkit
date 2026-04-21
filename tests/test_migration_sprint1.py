"""Sprint 1 tests for code migration: MigrationConfig, ReadOnlyFilesystemJail, detect_source_kind."""

from __future__ import annotations

import pytest

from vaig.core.migration.config import MigrationConfig, detect_source_kind
from vaig.core.migration.jail import ReadOnlyFilesystemJail

# ─── MigrationConfig ──────────────────────────────────────────────────────────


def test_migration_config_defaults(tmp_path: pytest.TempPathFactory) -> None:
    """MigrationConfig defaults should parse correctly without any arguments."""
    cfg = MigrationConfig(from_dir=str(tmp_path))
    assert cfg.to_dir == "./migrated"
    assert cfg.examples_dir is None
    assert cfg.target_kind == "aws-glue-pyspark"
    assert cfg.design_principles == []
    assert cfg.max_migration_iterations == 20
    assert cfg.migration_budget_usd == 10.0
    # source_kind auto-detected (empty dir → "unknown")
    assert cfg.source_kind == "unknown"


def test_migration_config_custom_values(tmp_path: pytest.TempPathFactory) -> None:
    """Custom MigrationConfig values should round-trip through Pydantic."""
    cfg = MigrationConfig(
        from_dir=str(tmp_path),
        to_dir="/output",
        examples_dir="/examples",
        source_kind="pentaho",
        target_kind="databricks-pyspark",
        design_principles=["no globals", "pure functions"],
        max_migration_iterations=5,
        migration_budget_usd=2.5,
    )
    assert cfg.from_dir == str(tmp_path)
    assert cfg.to_dir == "/output"
    assert cfg.examples_dir == "/examples"
    assert cfg.source_kind == "pentaho"
    assert cfg.target_kind == "databricks-pyspark"
    assert cfg.design_principles == ["no globals", "pure functions"]
    assert cfg.max_migration_iterations == 5
    assert cfg.migration_budget_usd == 2.5


# ─── ReadOnlyFilesystemJail ───────────────────────────────────────────────────


def test_jail_safe_read_reads_file_within_root(tmp_path: pytest.TempPathFactory) -> None:
    """safe_read should return the content of a file inside the jail root."""
    (tmp_path / "hello.txt").write_text("hello world")  # type: ignore[operator]
    with ReadOnlyFilesystemJail(tmp_path) as jail:  # type: ignore[arg-type]
        content = jail.safe_read("hello.txt")
    assert content == "hello world"


def test_jail_safe_read_raises_on_path_traversal(tmp_path: pytest.TempPathFactory) -> None:
    """safe_read should raise PermissionError when the path escapes the jail."""
    with ReadOnlyFilesystemJail(tmp_path) as jail:  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            jail.safe_read("../../etc/passwd")


# ─── detect_source_kind ───────────────────────────────────────────────────────


def test_detect_source_kind_pentaho(tmp_path: pytest.TempPathFactory) -> None:
    """detect_source_kind should return 'pentaho' when a .ktr file is present."""
    (tmp_path / "job.ktr").write_text("<transformation/>")  # type: ignore[operator]
    assert detect_source_kind(str(tmp_path)) == "pentaho"


def test_detect_source_kind_unknown_for_empty_dir(tmp_path: pytest.TempPathFactory) -> None:
    """detect_source_kind should return 'unknown' for an empty directory."""
    assert detect_source_kind(str(tmp_path)) == "unknown"
