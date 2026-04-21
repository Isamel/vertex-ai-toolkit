"""Sprint 3 tests: PlaceholderGate, SyntaxGate, InterfaceMatchGate, MigrationOrchestrator."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vaig.core.migration.config import MigrationConfig
from vaig.core.migration.gates.interface_match import (
    InterfaceMatchGate,
    InterfaceSignature,
)
from vaig.core.migration.gates.placeholder import PlaceholderGate
from vaig.core.migration.gates.syntax import SyntaxGate
from vaig.core.migration.jail import ReadOnlySourceError
from vaig.core.migration.orchestrator import MigrationOrchestrator

# ---------------------------------------------------------------------------
# PlaceholderGate
# ---------------------------------------------------------------------------

class TestPlaceholderGate:
    gate = PlaceholderGate()

    def test_bare_pass_in_function_fails(self):
        code = textwrap.dedent("""\
            def do_something():
                pass
        """)
        result = self.gate.check(code, "foo.py")
        assert not result.passed
        assert any(v.kind == "empty_function_body" for v in result.violations)

    def test_exception_class_pass_allowed(self):
        code = textwrap.dedent("""\
            class MyError(Exception):
                pass
        """)
        result = self.gate.check(code, "foo.py")
        # Exception-derived class with pass is legit
        assert result.passed

    def test_abstractmethod_pass_allowed(self):
        code = textwrap.dedent("""\
            from abc import abstractmethod, ABC

            class MyBase(ABC):
                @abstractmethod
                def run(self):
                    pass
        """)
        result = self.gate.check(code, "foo.py")
        assert result.passed

    def test_raise_not_implemented_fails(self):
        code = textwrap.dedent("""\
            def compute():
                raise NotImplementedError
        """)
        result = self.gate.check(code, "foo.py")
        assert not result.passed

    def test_real_implementation_passes(self):
        code = textwrap.dedent("""\
            def add(a: int, b: int) -> int:
                return a + b

            def greet(name: str) -> str:
                return f"Hello, {name}!"
        """)
        result = self.gate.check(code, "foo.py")
        assert result.passed

    def test_json_xxxxx_fails(self):
        content = '{"key": "xxxxx"}'
        result = self.gate.check(content, "config.json")
        assert not result.passed

    def test_sql_todo_comment_fails(self):
        content = "SELECT * FROM users; -- TODO: add WHERE clause"
        result = self.gate.check(content, "query.sql")
        assert not result.passed

    def test_markdown_todo_passes(self):
        content = "# TODO: write documentation\n\nSome text."
        result = self.gate.check(content, "README.md")
        assert result.passed  # .md is skipped


# ---------------------------------------------------------------------------
# SyntaxGate
# ---------------------------------------------------------------------------

class TestSyntaxGate:
    gate = SyntaxGate()

    def test_valid_python_passes(self):
        code = "x = 1\nprint(x)\n"
        result = self.gate.check(code, "script.py")
        assert result.passed
        assert not result.skipped

    def test_python_syntax_error_fails(self):
        code = "def foo()\n    pass\n"
        result = self.gate.check(code, "bad.py")
        assert not result.passed
        assert result.error is not None
        assert "line" in result.error.lower() or "syntax" in result.error.lower()

    def test_valid_json_passes(self):
        content = '{"key": "value", "num": 42}'
        result = self.gate.check(content, "data.json")
        assert result.passed

    def test_invalid_json_fails(self):
        content = '{"key": "value", "num": 42'  # missing closing }
        result = self.gate.check(content, "data.json")
        assert not result.passed
        assert result.error is not None

    def test_unknown_extension_skipped(self):
        content = "some random content"
        result = self.gate.check(content, "file.xyz")
        assert result.passed
        assert result.skipped


# ---------------------------------------------------------------------------
# InterfaceMatchGate
# ---------------------------------------------------------------------------

class TestInterfaceMatchGate:
    gate = InterfaceMatchGate()

    def _make_expected(self) -> list[InterfaceSignature]:
        return [
            InterfaceSignature(kind="function", name="process", arity=2),
            InterfaceSignature(kind="class", name="Processor"),
        ]

    def test_matching_signatures_passes(self):
        code = textwrap.dedent("""\
            class Processor:
                pass

            def process(data, config):
                return data
        """)
        result = self.gate.check(code, "module.py", self._make_expected())
        assert result.passed

    def test_missing_function_fails(self):
        code = textwrap.dedent("""\
            class Processor:
                pass
        """)
        result = self.gate.check(code, "module.py", self._make_expected())
        assert not result.passed
        missing = [d for d in result.diffs if d.kind == "missing"]
        assert any(d.name == "process" for d in missing)

    def test_extra_helper_passes(self):
        """Extra functions are warnings, not failures."""
        code = textwrap.dedent("""\
            class Processor:
                pass

            def process(data, config):
                return data

            def helper():
                return True
        """)
        result = self.gate.check(code, "module.py", self._make_expected())
        assert result.passed
        extra = [d for d in result.diffs if d.kind == "extra"]
        assert any(d.name == "helper" for d in extra)

    def test_wrong_arity_fails(self):
        code = textwrap.dedent("""\
            class Processor:
                pass

            def process(data):  # missing config param
                return data
        """)
        result = self.gate.check(code, "module.py", self._make_expected())
        assert not result.passed
        arity_issues = [d for d in result.diffs if d.kind == "arity_mismatch"]
        assert any(d.name == "process" for d in arity_issues)


# ---------------------------------------------------------------------------
# MigrationOrchestrator
# ---------------------------------------------------------------------------

_MINIMAL_KTR = """\
<?xml version="1.0" encoding="UTF-8"?>
<transformation>
  <info>
    <name>test_transform</name>
  </info>
  <step>
    <name>Read Data</name>
    <type>TableInput</type>
    <connection>mydb</connection>
    <sql>SELECT * FROM orders</sql>
  </step>
  <step>
    <name>Write Output</name>
    <type>TableOutput</type>
    <connection>mydb</connection>
    <table>orders_out</table>
  </step>
  <hop>
    <from>Read Data</from>
    <to>Write Output</to>
    <enabled>Y</enabled>
  </hop>
</transformation>
"""


class TestMigrationOrchestrator:
    def _make_config(self, from_dir: Path, to_dir: Path) -> MigrationConfig:
        return MigrationConfig(
            from_dirs=[from_dir],
            to_dir=to_dir,
            source_kind="pentaho",
        )

    def test_ingest_only_returns_domain_model(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        ktr = src / "pipeline.ktr"
        ktr.write_text(_MINIMAL_KTR, encoding="utf-8")

        to_dir = tmp_path / "out"
        config = self._make_config(src, to_dir)
        orchestrator = MigrationOrchestrator(config)

        domain = orchestrator.ingest_only()
        assert domain.source_kind == "pentaho"
        assert domain.node_count >= 1

    def test_run_raises_not_implemented(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "pipeline.ktr").write_text(_MINIMAL_KTR, encoding="utf-8")

        to_dir = tmp_path / "out"
        config = self._make_config(src, to_dir)
        orchestrator = MigrationOrchestrator(config)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            orchestrator.run("migrate to glue")

    def test_check_write_blocked_inside_from_dir(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "pipeline.ktr").write_text(_MINIMAL_KTR, encoding="utf-8")

        to_dir = tmp_path / "out"
        config = self._make_config(src, to_dir)
        orchestrator = MigrationOrchestrator(config)

        with pytest.raises(ReadOnlySourceError):
            orchestrator.check_write_allowed(src / "pipeline.ktr")

    def test_check_write_allowed_outside_from_dir(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "pipeline.ktr").write_text(_MINIMAL_KTR, encoding="utf-8")

        to_dir = tmp_path / "out"
        config = self._make_config(src, to_dir)
        orchestrator = MigrationOrchestrator(config)

        # Should not raise — path is outside the jail
        unrelated = tmp_path / "out" / "result.py"
        orchestrator.check_write_allowed(unrelated)  # no exception
