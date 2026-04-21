"""Sprint 5 tests: target packs, DDD overlay, migration state checkpointing."""
from pathlib import Path

# ---------------------------------------------------------------------------
# CM-16: AWS Glue PySpark target pack
# ---------------------------------------------------------------------------

def test_target_pack_has_required_fields() -> None:
    from vaig.skills.code_migration.targets.aws_glue_pyspark import AWS_GLUE_PYSPARK_PACK

    assert AWS_GLUE_PYSPARK_PACK.name == "aws-glue-pyspark"
    assert AWS_GLUE_PYSPARK_PACK.version == "4.0"
    assert len(AWS_GLUE_PYSPARK_PACK.imports) > 0
    assert len(AWS_GLUE_PYSPARK_PACK.forbidden_apis) > 0


def test_target_pack_has_patterns() -> None:
    from vaig.skills.code_migration.targets.aws_glue_pyspark import AWS_GLUE_PYSPARK_PACK

    assert isinstance(AWS_GLUE_PYSPARK_PACK.patterns, dict)
    assert len(AWS_GLUE_PYSPARK_PACK.patterns) >= 5


def test_target_pack_has_examples() -> None:
    from vaig.skills.code_migration.targets.aws_glue_pyspark import AWS_GLUE_PYSPARK_PACK

    assert len(AWS_GLUE_PYSPARK_PACK.examples) >= 2
    for ex in AWS_GLUE_PYSPARK_PACK.examples:
        assert ex.source
        assert ex.target


def test_databricks_pack_is_scaffold() -> None:
    from vaig.skills.code_migration.targets.databricks_pyspark import DATABRICKS_PYSPARK_PACK

    assert DATABRICKS_PYSPARK_PACK.name == "databricks-pyspark"
    assert DATABRICKS_PYSPARK_PACK.version == "13.3-lts"


# ---------------------------------------------------------------------------
# CM-08: DDD overlay
# ---------------------------------------------------------------------------

def _make_overlay():  # type: ignore[no-untyped-def]
    from vaig.core.migration.ddd_overlay import AggregateRoot, BoundedContext, DddOverlay

    ctx1 = BoundedContext(
        name="ingestion",
        aggregates=[AggregateRoot(name="OrderIngest", entities=["Order"])],
        ubiquitous_language={"order": "A customer purchase record"},
    )
    ctx2 = BoundedContext(
        name="persistence",
        aggregates=[
            AggregateRoot(name="OrderStore", entities=["StoredOrder"]),
            AggregateRoot(name="CustomerStore", entities=["Customer"]),
        ],
    )
    return DddOverlay(contexts=[ctx1, ctx2])


def test_ddd_overlay_find_context() -> None:
    overlay = _make_overlay()
    ctx = overlay.find_context("ingestion")
    assert ctx is not None
    assert ctx.name == "ingestion"

    missing = overlay.find_context("nonexistent")
    assert missing is None


def test_ddd_overlay_all_aggregates() -> None:
    overlay = _make_overlay()
    aggs = overlay.all_aggregates()
    assert len(aggs) == 3
    names = {a.name for a in aggs}
    assert "OrderIngest" in names
    assert "OrderStore" in names
    assert "CustomerStore" in names


def test_ddd_overlay_enrich_chunk_no_match() -> None:
    from vaig.core.migration.ddd_overlay import DddOverlay
    from vaig.core.migration.domain import Chunk, DomainNode

    overlay = DddOverlay(contexts=[])
    node = DomainNode(
        step_name="mystery_step",
        step_type="custom",
        semantic_kind="UNKNOWN",
    )
    chunk = Chunk(node=node, text="mystery step")
    result = overlay.enrich_chunk(chunk)
    assert result == {}


# ---------------------------------------------------------------------------
# CM-18: MigrationState checkpointing
# ---------------------------------------------------------------------------

def test_migration_state_new() -> None:
    from vaig.core.migration.state import MigrationState

    state = MigrationState.new("pentaho", "aws-glue-pyspark")
    assert state.change_id
    assert state.source_kind == "pentaho"
    assert state.target_kind == "aws-glue-pyspark"
    assert state.pending_files() == []


def test_migration_state_mark_completed() -> None:
    from vaig.core.migration.state import FileStatus, MigrationState

    state = MigrationState.new("pentaho", "aws-glue-pyspark")
    state.files["job.ktr"] = __import__(
        "vaig.core.migration.state", fromlist=["FileRecord"]
    ).FileRecord(source_path="job.ktr")
    state.mark_completed("job.ktr", "job.py", [])
    assert state.files["job.ktr"].status == FileStatus.COMPLETED
    assert state.is_complete()


def test_migration_state_mark_failed() -> None:
    from vaig.core.migration.state import FileRecord, FileStatus, MigrationState

    state = MigrationState.new("pentaho", "aws-glue-pyspark")
    state.files["bad.ktr"] = FileRecord(source_path="bad.ktr")
    state.mark_failed("bad.ktr", "parse error")
    assert state.files["bad.ktr"].status == FileStatus.FAILED
    assert state.files["bad.ktr"].error == "parse error"


def test_migration_state_save_load(tmp_path: Path) -> None:
    from vaig.core.migration.state import FileRecord, FileStatus, MigrationState

    state = MigrationState.new("pentaho", "aws-glue-pyspark")
    state.files["a.ktr"] = FileRecord(source_path="a.ktr")
    state.mark_completed("a.ktr", "a.py", [{"gate": "syntax", "passed": True}])

    save_path = tmp_path / "state.json"
    state.save(save_path)
    loaded = MigrationState.load(save_path)

    assert loaded.change_id == state.change_id
    assert loaded.source_kind == state.source_kind
    assert loaded.files["a.ktr"].status == FileStatus.COMPLETED
    assert loaded.files["a.ktr"].target_path == "a.py"


def test_migration_state_pending_files() -> None:
    from vaig.core.migration.state import FileRecord, MigrationState

    state = MigrationState.new("pentaho", "aws-glue-pyspark")
    state.files["a.ktr"] = FileRecord(source_path="a.ktr")
    state.files["b.ktr"] = FileRecord(source_path="b.ktr")
    state.files["c.ktr"] = FileRecord(source_path="c.ktr")
    state.mark_completed("a.ktr", "a.py", [])

    pending = state.pending_files()
    assert len(pending) == 2
    assert "b.ktr" in pending
    assert "c.ktr" in pending


# ---------------------------------------------------------------------------
# CM-18: Orchestrator state_path and resume
# ---------------------------------------------------------------------------

def test_orchestrator_accepts_state_path(tmp_path: Path) -> None:
    from vaig.core.migration.config import MigrationConfig
    from vaig.core.migration.orchestrator import MigrationOrchestrator

    cfg = MigrationConfig(from_dirs=[tmp_path], source_kind="generic")
    state_path = tmp_path / "state.json"
    orch = MigrationOrchestrator(cfg, state_path=state_path)
    assert orch._state_path == state_path


def test_orchestrator_resume_skips_completed(tmp_path: Path) -> None:
    from vaig.core.migration.config import MigrationConfig
    from vaig.core.migration.orchestrator import MigrationOrchestrator
    from vaig.core.migration.state import FileRecord, FileStatus, MigrationState

    # Prepare a state with one completed file
    state = MigrationState.new("pentaho", "aws-glue-pyspark")
    state.files["done.ktr"] = FileRecord(source_path="done.ktr")
    state.mark_completed("done.ktr", "done.py", [])
    state_path = tmp_path / "state.json"
    state.save(state_path)

    cfg = MigrationConfig(from_dirs=[tmp_path], source_kind="pentaho")
    orch = MigrationOrchestrator(cfg, state_path=state_path, resume=True)

    # Loaded state should have the completed file
    assert orch._state.files["done.ktr"].status == FileStatus.COMPLETED
    # pending_files should not include the completed file
    assert "done.ktr" not in orch._state.pending_files()
