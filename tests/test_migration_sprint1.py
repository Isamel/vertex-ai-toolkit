"""Sprint 1 tests for code migration: MigrationConfig, ReadOnlyFilesystemJail, detect_source_kind."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.migration.config import MigrationConfig, detect_source_kind
from vaig.core.migration.jail import ReadOnlyFilesystemJail, ReadOnlySourceError

# ─── MigrationConfig ──────────────────────────────────────────────────────────


def test_migration_config_defaults(tmp_path: Path) -> None:
    """MigrationConfig defaults should parse correctly without any from_dirs."""
    cfg = MigrationConfig()
    assert cfg.to_dir == Path("./migrated")
    assert cfg.examples_dirs == []
    assert cfg.target_kind == "aws-glue-pyspark"
    assert cfg.design_principles == ["tdd", "sdd"]
    assert cfg.max_migration_iterations == 20
    assert cfg.migration_budget_usd == 10.0
    # No from_dirs → source_kind stays None
    assert cfg.source_kind is None


def test_migration_config_with_from_dirs(tmp_path: Path) -> None:
    """MigrationConfig with from_dirs should auto-detect source_kind."""
    cfg = MigrationConfig(from_dirs=[tmp_path])
    assert cfg.from_dirs == [tmp_path]
    assert cfg.to_dir == Path("./migrated")
    # empty dir → generic
    assert cfg.source_kind == "generic"


def test_migration_config_custom_values(tmp_path: Path) -> None:
    """Custom MigrationConfig values should round-trip through Pydantic."""
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "output"
    out.mkdir()
    cfg = MigrationConfig(
        from_dirs=[src],
        to_dir=out,
        source_kind="pentaho",
        target_kind="databricks-pyspark",
        design_principles=["tdd", "sdd"],  # type: ignore[list-item]
        max_migration_iterations=5,
        migration_budget_usd=2.5,
    )
    assert cfg.from_dirs == [src]
    assert cfg.to_dir == out
    assert cfg.source_kind == "pentaho"
    assert cfg.target_kind == "databricks-pyspark"
    assert cfg.design_principles == ["tdd", "sdd"]
    assert cfg.max_migration_iterations == 5
    assert cfg.migration_budget_usd == 2.5


# ─── ReadOnlyFilesystemJail ───────────────────────────────────────────────────


def test_jail_safe_read_reads_file_within_root(tmp_path: Path) -> None:
    """safe_read should return the content of a file inside the jail root."""
    (tmp_path / "hello.txt").write_text("hello world")
    with ReadOnlyFilesystemJail(tmp_path) as jail:
        content = jail.safe_read("hello.txt")
    assert content == "hello world"


def test_jail_safe_read_raises_on_path_traversal(tmp_path: Path) -> None:
    """safe_read should raise ReadOnlySourceError when the path escapes the jail."""
    with ReadOnlyFilesystemJail(tmp_path) as jail:
        with pytest.raises(ReadOnlySourceError):
            jail.safe_read("../../etc/passwd")


def test_jail_raises_is_permission_error(tmp_path: Path) -> None:
    """ReadOnlySourceError is a PermissionError subclass."""
    with ReadOnlyFilesystemJail(tmp_path) as jail:
        with pytest.raises(PermissionError):
            jail.safe_read("../../etc/passwd")


def test_jail_check_write_blocked_inside(tmp_path: Path) -> None:
    """check_write_blocked should raise for paths inside the jail."""
    jail = ReadOnlyFilesystemJail(tmp_path)
    with pytest.raises(ReadOnlySourceError):
        jail.check_write_blocked(tmp_path / "file.txt")


def test_jail_check_write_blocked_outside(tmp_path: Path) -> None:
    """check_write_blocked should NOT raise for paths outside the jail."""
    import tempfile
    jail = ReadOnlyFilesystemJail(tmp_path)
    outside = Path(tempfile.gettempdir()) / "outside.txt"
    jail.check_write_blocked(outside)  # should not raise


# ─── detect_source_kind ───────────────────────────────────────────────────────


def test_detect_source_kind_pentaho(tmp_path: Path) -> None:
    """detect_source_kind should return 'pentaho' when a .ktr file is present."""
    (tmp_path / "job.ktr").write_text("<transformation/>")
    assert detect_source_kind([tmp_path]) == "pentaho"


def test_detect_source_kind_ssis(tmp_path: Path) -> None:
    """detect_source_kind should return 'ssis' when a .dtsx file is present."""
    (tmp_path / "package.dtsx").write_text("")
    assert detect_source_kind([tmp_path]) == "ssis"


def test_detect_source_kind_cobol(tmp_path: Path) -> None:
    """detect_source_kind should return 'cobol' when a .cbl file is present."""
    (tmp_path / "main.cbl").write_text("")
    assert detect_source_kind([tmp_path]) == "cobol"


def test_detect_source_kind_generic_for_empty_dir(tmp_path: Path) -> None:
    """detect_source_kind should return 'generic' for an empty directory."""
    assert detect_source_kind([tmp_path]) == "generic"
