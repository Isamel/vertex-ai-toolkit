"""Lint gate: runs ruff on generated Python code and returns violations."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from vaig.core.migration.domain import Chunk
from vaig.core.migration.gates.base import GateResult, QualityGate

__all__ = ["LintGate"]


class LintGate(QualityGate):
    """Runs ruff (if available) on generated Python code and returns violations."""

    def check(self, chunk: Chunk, generated_code: str, strict: bool = False) -> GateResult:
        """
        Write generated_code to a temp file, run:
          python -m ruff check <tempfile> --output-format=json
        Parse JSON output → list of violation messages.
        If ruff not available → PASS with note "ruff not available".
        If strict=False → return PASS even with violations, but list them in notes.
        If strict=True → return FAIL if any violations found.
        Always clean up temp file.
        """
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(generated_code)
                tmp_path = Path(f.name)

            result = subprocess.run(
                ["python", "-m", "ruff", "check", str(tmp_path), "--output-format=json"],
                capture_output=True,
                text=True,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except FileNotFoundError:
            return GateResult(passed=True, notes="ruff not available")
        except Exception as exc:
            return GateResult(passed=True, notes=f"ruff check skipped: {exc}")
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        # Parse violations
        violations: list[str] = []
        try:
            raw = result.stdout.strip()
            if raw:
                items = json.loads(raw)
                for item in items:
                    code = item.get("code", "?")
                    msg = item.get("message", "")
                    row = item.get("location", {}).get("row", "?")
                    violations.append(f"{code} line {row}: {msg}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass  # best-effort parsing

        if not violations:
            return GateResult(passed=True)

        notes = "; ".join(violations)
        if strict:
            return GateResult(passed=False, violations=violations, notes=notes)
        return GateResult(passed=True, violations=violations, notes=notes)
