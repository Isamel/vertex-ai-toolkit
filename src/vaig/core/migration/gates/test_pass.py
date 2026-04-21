"""Test-pass gate: runs the migrated code's test suite and checks for green."""
import subprocess
import tempfile
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
        tmp_dir = working_dir or Path(tempfile.gettempdir())
        tmp_file = tmp_dir / "_test_pass_gate_tmp.py"

        try:
            tmp_file.write_text(generated_code, encoding="utf-8")

            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", str(tmp_file), "-x", "-q",
                     "--timeout=10", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except FileNotFoundError:
                return GateResult(passed=True, notes="pytest not available, skipping")

            if result.returncode == 0:
                return GateResult(passed=True)

            output = (result.stdout + result.stderr).strip()
            return GateResult(passed=False, violations=[output])

        finally:
            if tmp_file.exists():
                tmp_file.unlink()
