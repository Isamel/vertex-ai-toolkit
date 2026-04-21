"""Syntax gate: parse-based rejection of syntactically invalid files."""
from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SyntaxGate", "SyntaxGateResult"]

logger = logging.getLogger(__name__)


@dataclass
class SyntaxGateResult:
    passed: bool
    file_path: str
    error: str | None = None  # parse error message with line:col if available
    skipped: bool = False  # unknown extension


class SyntaxGate:
    def check(self, content: str, file_path: str) -> SyntaxGateResult:
        ext = Path(file_path).suffix.lower()

        if ext == ".py":
            return self._check_python(content, file_path)
        elif ext == ".json":
            return self._check_json(content, file_path)
        elif ext == ".yaml" or ext == ".yml":
            return self._check_yaml(content, file_path)
        elif ext == ".sql":
            return self._check_sql(content, file_path)
        else:
            logger.debug("SyntaxGate: skipping unknown extension %s for %s", ext, file_path)
            return SyntaxGateResult(passed=True, file_path=file_path, skipped=True)

    def _check_python(self, content: str, file_path: str) -> SyntaxGateResult:
        try:
            ast.parse(content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except SyntaxError as e:
            error = f"SyntaxError at line {e.lineno}:{e.offset}: {e.msg}"
            return SyntaxGateResult(passed=False, file_path=file_path, error=error)
        return SyntaxGateResult(passed=True, file_path=file_path)

    def _check_json(self, content: str, file_path: str) -> SyntaxGateResult:
        try:
            json.loads(content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except json.JSONDecodeError as e:
            error = f"JSONDecodeError at line {e.lineno}:{e.colno}: {e.msg}"
            return SyntaxGateResult(passed=False, file_path=file_path, error=error)
        return SyntaxGateResult(passed=True, file_path=file_path)

    def _check_yaml(self, content: str, file_path: str) -> SyntaxGateResult:
        try:
            import yaml  # noqa: PLC0415
        except ImportError:
            logger.debug("SyntaxGate: pyyaml not installed, skipping YAML check for %s", file_path)
            return SyntaxGateResult(passed=True, file_path=file_path, skipped=True)

        try:
            yaml.safe_load(content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            return SyntaxGateResult(passed=False, file_path=file_path, error=str(e))
        return SyntaxGateResult(passed=True, file_path=file_path)

    def _check_sql(self, content: str, file_path: str) -> SyntaxGateResult:
        try:
            import sqlglot  # noqa: PLC0415
        except ImportError:
            logger.debug("SyntaxGate: sqlglot not installed, skipping SQL check for %s", file_path)
            return SyntaxGateResult(passed=True, file_path=file_path, skipped=True)

        try:
            sqlglot.parse_one(content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            return SyntaxGateResult(passed=False, file_path=file_path, error=str(e))
        return SyntaxGateResult(passed=True, file_path=file_path)
