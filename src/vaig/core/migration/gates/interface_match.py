"""Interface-match gate: semantic diff of public API against spec signatures."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

__all__ = ["InterfaceMatchGate", "InterfaceSignature", "InterfaceDiff", "InterfaceMatchResult"]


@dataclass(frozen=True)
class InterfaceSignature:
    kind: Literal["function", "class", "constant", "type"]
    name: str
    arity: int | None = None
    param_names: list[str] = field(default_factory=list, hash=False, compare=False)
    return_type: str | None = None


@dataclass
class InterfaceDiff:
    kind: Literal["missing", "extra", "arity_mismatch", "param_rename", "return_type_narrowed"]
    name: str
    expected: InterfaceSignature | None = None
    actual: InterfaceSignature | None = None
    description: str = ""


@dataclass
class InterfaceMatchResult:
    passed: bool
    file_path: str
    diffs: list[InterfaceDiff] = field(default_factory=list)

    def to_xml_feedback(self) -> str:
        parts = [
            f'<interface_match_result file="{self.file_path}" passed="{str(self.passed).lower()}">'
        ]
        for d in self.diffs:
            parts.append(
                f'  <diff kind="{d.kind}" name="{d.name}">{d.description}</diff>'
            )
        parts.append("</interface_match_result>")
        return "\n".join(parts)


def extract_python_signatures(content: str) -> list[InterfaceSignature]:
    """Extract public function and class signatures from Python source."""
    sigs: list[InterfaceSignature] = []
    try:
        tree = ast.parse(content)
    except (KeyboardInterrupt, SystemExit):
        raise
    except SyntaxError:
        return sigs
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            params = [a.arg for a in node.args.args if a.arg != "self"]
            sigs.append(
                InterfaceSignature(
                    kind="function",
                    name=node.name,
                    arity=len(params),
                    param_names=params,
                )
            )
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                sigs.append(InterfaceSignature(kind="class", name=node.name))
    return sigs


class InterfaceMatchGate:
    def check(
        self,
        content: str,
        file_path: str,
        expected: list[InterfaceSignature],
    ) -> InterfaceMatchResult:
        """Compare actual public API against expected signatures."""
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            actual = extract_python_signatures(content)
        else:
            # For non-Python, pass through (no extractor yet)
            return InterfaceMatchResult(passed=True, file_path=file_path)

        diffs = diff_signatures(expected, actual)
        # "extra" is warning only — not a hard failure
        hard_failures = [d for d in diffs if d.kind != "extra"]
        return InterfaceMatchResult(
            passed=len(hard_failures) == 0,
            file_path=file_path,
            diffs=diffs,
        )


def diff_signatures(
    expected: list[InterfaceSignature],
    actual: list[InterfaceSignature],
) -> list[InterfaceDiff]:
    diffs: list[InterfaceDiff] = []
    actual_by_name = {s.name: s for s in actual}
    expected_by_name = {s.name: s for s in expected}

    for exp in expected:
        act = actual_by_name.get(exp.name)
        if act is None:
            diffs.append(
                InterfaceDiff(
                    kind="missing",
                    name=exp.name,
                    expected=exp,
                    description=f"Expected {exp.kind} '{exp.name}' not found.",
                )
            )
        elif exp.kind == "function" and exp.arity is not None and act.arity != exp.arity:
            diffs.append(
                InterfaceDiff(
                    kind="arity_mismatch",
                    name=exp.name,
                    expected=exp,
                    actual=act,
                    description=f"Expected arity {exp.arity}, got {act.arity}.",
                )
            )

    for act in actual:
        if act.name not in expected_by_name:
            diffs.append(
                InterfaceDiff(
                    kind="extra",
                    name=act.name,
                    actual=act,
                    description=f"Extra {act.kind} '{act.name}' not in spec (warning only).",
                )
            )

    return diffs
