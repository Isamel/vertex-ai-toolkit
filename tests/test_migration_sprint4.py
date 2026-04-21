"""Sprint 4 tests: SddGate, TddGate, TestPassGate, orchestrator wiring."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.core.migration.config import MigrationConfig
from vaig.core.migration.domain import Chunk, DomainNode
from vaig.core.migration.gates.base import GateResult
from vaig.core.migration.gates.sdd_gate import MigrationSpec, SddGate
from vaig.core.migration.gates.tdd_gate import TddGate, TddPhase
from vaig.core.migration.gates.test_pass import TestPassGate
from vaig.core.migration.orchestrator import MigrationOrchestrator


def _make_chunk(source_file: str = "transform.py") -> Chunk:
    node = DomainNode(
        step_name="step",
        step_type="TRANSFORM",
        semantic_kind="TRANSFORM",
        source_file=source_file,
    )
    return Chunk(node=node, text="")


def _make_spec(**kwargs: object) -> MigrationSpec:
    defaults: dict[str, object] = {
        "source_path": Path("src/old.py"),
        "target_path": Path("src/new.py"),
    }
    defaults.update(kwargs)
    return MigrationSpec(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CM-06 SddGate
# ---------------------------------------------------------------------------


class TestSddGate:
    gate = SddGate()

    def test_sdd_gate_passes_when_no_spec(self):
        result = self.gate.check(_make_chunk(), "print('hello')", spec=None)
        assert result.passed

    def test_sdd_gate_fails_on_forbidden_pattern(self):
        spec = _make_spec(forbidden_patterns=[r"print\("])
        result = self.gate.check(_make_chunk(), "print('hello')", spec=spec)
        assert not result.passed
        assert any("print" in v for v in result.violations)

    def test_sdd_gate_fails_on_missing_required_pattern(self):
        spec = _make_spec(required_patterns=["spark\\.createDataFrame"])
        result = self.gate.check(_make_chunk(), "df = pd.DataFrame()", spec=spec)
        assert not result.passed
        assert any("spark" in v for v in result.violations)

    def test_sdd_gate_passes_all_patterns(self):
        spec = _make_spec(
            required_patterns=["spark\\.createDataFrame"],
            forbidden_patterns=[r"print\("],
        )
        code = "df = spark.createDataFrame(data)"
        result = self.gate.check(_make_chunk(), code, spec=spec)
        assert result.passed
        assert result.violations == []


# ---------------------------------------------------------------------------
# CM-07 TddGate
# ---------------------------------------------------------------------------


class TestTddGate:
    gate = TddGate()

    def test_tdd_gate_red_phase_with_stubs(self):
        code = textwrap.dedent("""\
            def test_foo():
                pass
        """)
        result = self.gate.check(_make_chunk("test_transform.py"), code, phase=TddPhase.RED)
        assert result.passed

    def test_tdd_gate_red_phase_with_impl_fails(self):
        code = textwrap.dedent("""\
            def test_foo():
                result = 1 + 1
                assert result == 2
        """)
        result = self.gate.check(_make_chunk("test_transform.py"), code, phase=TddPhase.RED)
        assert not result.passed

    def test_tdd_gate_green_phase_with_impl(self):
        code = textwrap.dedent("""\
            def test_foo():
                result = 1 + 1
                assert result == 2
        """)
        result = self.gate.check(_make_chunk("test_transform.py"), code, phase=TddPhase.GREEN)
        assert result.passed

    def test_tdd_gate_non_test_file_skipped(self):
        code = textwrap.dedent("""\
            def test_foo():
                pass
        """)
        chunk = _make_chunk("transform.py")
        for phase in TddPhase:
            result = self.gate.check(chunk, code, phase=phase)
            assert result.passed
            assert "not a test file" in result.notes


# ---------------------------------------------------------------------------
# CM-12 TestPassGate
# ---------------------------------------------------------------------------


class TestTestPassGate:
    gate = TestPassGate()

    def test_test_pass_gate_pytest_not_available(self, tmp_path: Path):
        with patch("subprocess.run", side_effect=FileNotFoundError("pytest not found")):
            result = self.gate.check(_make_chunk(), "x = 1", working_dir=tmp_path)
        assert result.passed
        assert "pytest not available" in result.notes

    def test_test_pass_gate_passes_on_zero_returncode(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1 passed"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = self.gate.check(_make_chunk(), "def test_x(): pass", working_dir=tmp_path)
        assert result.passed

    def test_test_pass_gate_fails_on_nonzero_returncode(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "FAILED"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = self.gate.check(_make_chunk(), "def test_x(): assert False", working_dir=tmp_path)
        assert not result.passed
        assert any("FAILED" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


class TestOrchestratorSprint4:
    def _make_config(self, tmp_path: Path) -> MigrationConfig:
        src = tmp_path / "src"
        src.mkdir()
        return MigrationConfig(from_dirs=[src], source_kind="generic")

    def test_orchestrator_accepts_sdd_specs(self, tmp_path: Path):
        specs = {
            "src/old.py": MigrationSpec(
                source_path=Path("src/old.py"),
                target_path=Path("src/new.py"),
            )
        }
        orch = MigrationOrchestrator(self._make_config(tmp_path), sdd_specs=specs)
        assert orch.sdd_specs == specs

    def test_gate_result_has_violations_list(self):
        result = GateResult(passed=True)
        assert isinstance(result.violations, list)

        result2 = GateResult(passed=False, violations=["bad pattern"])
        assert result2.violations == ["bad pattern"]
