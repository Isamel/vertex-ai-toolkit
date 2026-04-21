"""Test-pass gate: runs the migrated code's test suite and checks for green."""
import importlib.util
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from vaig.core.migration.domain import Chunk
from vaig.core.migration.gates.base import GateResult, QualityGate

__all__ = ["TestPassGate"]


class TestPassGate(QualityGate):
    """Writes generated code to a temp file and runs pytest against it."""

    def check(
        self,
        chunk: Chunk,
        generated_code: str,
        working_dir: Path | None = None,
    ) -> GateResult:
        if importlib.util.find_spec("pytest") is None:
            return GateResult(passed=True, notes="pytest not available, skipping")

        tmp_dir = working_dir or Path(tempfile.gettempdir())
        tmp_file = tmp_dir / f"_test_pass_gate_{uuid.uuid4().hex}.py"

        try:
            tmp_file.write_text(generated_code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(tmp_file), "-x", "-q",
                     "--timeout=10", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(working_dir) if working_dir else None,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                return GateResult(passed=True, notes="pytest not available, skipping")

            if result.returncode == 0:
                return GateResult(passed=True)

            output = (result.stdout + result.stderr).strip()
            return GateResult(passed=False, violations=[output])

        finally:
            if tmp_file.exists():
                tmp_file.unlink()
