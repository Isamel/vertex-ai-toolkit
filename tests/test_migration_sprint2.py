"""Sprint 2 tests: SourceAdapter interface and PentahoAdapter."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vaig.core.migration.adapters.base import SourceAdapterRegistry, UnknownSourceError
from vaig.core.migration.adapters.pentaho import PentahoAdapter

# ─── Fixtures ────────────────────────────────────────────────────────────────

KTR_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <transformation>
      <info><name>TestJob</name></info>
      <parameters>
        <parameter><name>RUN_DATE</name><default_value>2024-01-01</default_value></parameter>
      </parameters>
      <step>
        <name>Read From DB</name>
        <type>TableInput</type>
        <connection>mydb</connection>
        <sql>SELECT * FROM stats</sql>
      </step>
      <step>
        <name>Filter Valid</name>
        <type>FilterRows</type>
        <condition>valid = true</condition>
      </step>
      <step>
        <name>Write To DB</name>
        <type>TableOutput</type>
        <connection>mydb</connection>
        <table>output_stats</table>
      </step>
      <step>
        <name>Unknown Widget</name>
        <type>CustomUnsupportedWidget</type>
      </step>
      <order>
        <hop><from>Read From DB</from><to>Filter Valid</to></hop>
        <hop><from>Filter Valid</from><to>Write To DB</to></hop>
      </order>
    </transformation>
""")


@pytest.fixture()
def pentaho_dir(tmp_path: Path) -> Path:
    ktr = tmp_path / "job.ktr"
    ktr.write_text(KTR_FIXTURE)
    return tmp_path


@pytest.fixture()
def multi_ktr_dir(tmp_path: Path) -> Path:
    for i in range(5):
        f = tmp_path / f"job_{i}.ktr"
        f.write_text(KTR_FIXTURE)
    (tmp_path / "extra.kjb").write_text(
        '<?xml version="1.0"?><job><info><name>J</name></info></job>'
    )
    return tmp_path


# ─── SourceAdapterRegistry tests ─────────────────────────────────────────────

def test_registry_get_pentaho() -> None:
    adapter = SourceAdapterRegistry.get("pentaho")
    assert isinstance(adapter, PentahoAdapter)


def test_registry_get_unknown_raises() -> None:
    with pytest.raises(UnknownSourceError, match="No adapter for source kind"):
        SourceAdapterRegistry.get("cobol_mainframe_v9")


def test_registry_auto_detect_pentaho(pentaho_dir: Path) -> None:
    adapter = SourceAdapterRegistry.auto_detect([pentaho_dir])
    assert adapter.kind == "pentaho"


def test_registry_auto_detect_unknown_raises(tmp_path: Path) -> None:
    # Empty dir — nothing to detect
    with pytest.raises(UnknownSourceError):
        SourceAdapterRegistry.auto_detect([tmp_path])


# ─── PentahoAdapter.detect tests ─────────────────────────────────────────────

def test_pentaho_detect_confidence_high(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    confidence = adapter.detect([pentaho_dir])
    assert confidence >= 0.95


def test_pentaho_detect_empty_dir_returns_zero(tmp_path: Path) -> None:
    adapter = PentahoAdapter()
    assert adapter.detect([tmp_path]) == 0.0


# ─── PentahoAdapter.parse tests ──────────────────────────────────────────────

def test_pentaho_parse_nodes(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    assert domain.source_kind == "pentaho"
    assert domain.node_count >= 4  # 3 known + 1 unknown


def test_pentaho_parse_hops(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    assert domain.hop_count >= 2


def test_pentaho_parse_parameters(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    assert "RUN_DATE" in domain.parameters
    assert domain.parameters["RUN_DATE"] == "2024-01-01"


def test_pentaho_parse_unknown_step_creates_evidence_gap(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    assert any("CustomUnsupportedWidget" in gap for gap in domain.evidence_gaps)


def test_pentaho_parse_node_source_file(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    for node in domain.nodes:
        assert node.source_file, f"Node {node.step_name} missing source_file"


def test_pentaho_parse_semantic_kinds(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    kinds = {n.semantic_kind for n in domain.nodes}
    # Should have SOURCE, TRANSFORM, SINK, UNKNOWN
    assert "SOURCE" in kinds
    assert "SINK" in kinds


def test_pentaho_parse_multi_file(multi_ktr_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([multi_ktr_dir])
    # 5 ktr × 4 steps + 1 empty kjb
    assert domain.node_count >= 20


# ─── PentahoAdapter.chunk tests ──────────────────────────────────────────────

def test_pentaho_chunk_yields_chunks(pentaho_dir: Path) -> None:
    adapter = PentahoAdapter()
    domain = adapter.parse([pentaho_dir])
    chunks = list(adapter.chunk(domain))
    assert len(chunks) == domain.node_count
    for chunk in chunks:
        assert chunk.text
        assert chunk.tags


# ─── Sprint 1 patch: detect_source_kind with list[Path] ──────────────────────

def test_detect_source_kind_pentaho(tmp_path: Path) -> None:
    (tmp_path / "job.ktr").write_text("")
    from vaig.core.migration.config import detect_source_kind
    assert detect_source_kind([tmp_path]) == "pentaho"


def test_detect_source_kind_ssis(tmp_path: Path) -> None:
    (tmp_path / "package.dtsx").write_text("")
    from vaig.core.migration.config import detect_source_kind
    assert detect_source_kind([tmp_path]) == "ssis"


def test_detect_source_kind_cobol(tmp_path: Path) -> None:
    (tmp_path / "main.cbl").write_text("")
    from vaig.core.migration.config import detect_source_kind
    assert detect_source_kind([tmp_path]) == "cobol"


def test_detect_source_kind_unknown(tmp_path: Path) -> None:
    from vaig.core.migration.config import detect_source_kind
    assert detect_source_kind([tmp_path]) == "generic"
