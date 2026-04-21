"""Placeholder gate: AST-based check for TODO/FIXME/placeholder strings."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PlaceholderGate", "PlaceholderViolation", "PlaceholderGateResult"]


@dataclass
class PlaceholderViolation:
    line: int
    kind: str  # "empty_function_body", "todo_comment", "placeholder_string", etc.
    description: str


@dataclass
class PlaceholderGateResult:
    passed: bool
    file_path: str
    violations: list[PlaceholderViolation] = field(default_factory=list)

    def to_xml_feedback(self, iteration: int) -> str:
        """Structured XML feedback for the Executor — never truncated."""
        parts = [
            f'<placeholder_gate_result file="{self.file_path}" passed="{str(self.passed).lower()}" iteration="{iteration}">'
        ]
        for v in self.violations:
            parts.append(
                f'  <violation line="{v.line}" kind="{v.kind}">'
                f"{v.description}"
                f"</violation>"
            )
        parts.append("</placeholder_gate_result>")
        return "\n".join(parts)


# Regex patterns per extension
_BANNED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ".py": [
        re.compile(r"^\s*\.\.\.\s*(#.*)?$", re.MULTILINE),
        re.compile(r"raise\s+NotImplementedError", re.IGNORECASE),
        re.compile(r"#\s*TODO[: ]", re.IGNORECASE),
        re.compile(r"xxxxx+", re.IGNORECASE),
    ],
    ".sql": [
        re.compile(r"--\s*TODO[: ]", re.IGNORECASE),
        re.compile(r"/\*\s*placeholder", re.IGNORECASE),
        re.compile(r"xxxxx+", re.IGNORECASE),
        re.compile(r"\bNULL\s*--\s*fix this\b", re.IGNORECASE),
    ],
    ".json": [
        re.compile(r'"xxxxx+"'),
        re.compile(r'"TODO"'),
        re.compile(r'"FIXME"'),
    ],
    ".yaml": [
        re.compile(r"xxxxx+"),
        re.compile(r"#\s*TODO[: ]"),
        re.compile(r":\s*TODO\b"),
    ],
    ".md": [],  # Markdown: intentional TOC, skip
}


class PlaceholderGate:
    def check(self, content: str, file_path: str) -> PlaceholderGateResult:
        ext = Path(file_path).suffix.lower()
        violations: list[PlaceholderViolation] = []

        # Python: AST-based pass detection + regex for other patterns
        if ext == ".py":
            violations.extend(self._check_python_ast(content, file_path))
            violations.extend(self._check_regex(content, ext, skip_pass=True))
        else:
            violations.extend(self._check_regex(content, ext))

        return PlaceholderGateResult(
            passed=len(violations) == 0,
            file_path=file_path,
            violations=violations,
        )

    def _check_python_ast(self, content: str, file_path: str) -> list[PlaceholderViolation]:
        """AST-based check for bare pass in non-legit contexts."""
        violations: list[PlaceholderViolation] = []
        try:
            tree = ast.parse(content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except SyntaxError:
            return violations  # SyntaxGate handles syntax errors

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if self._is_legit_pass_context(node, tree):
                continue
            if self._has_abstractmethod(node):
                continue
            # Check for bare pass or ... body
            body = node.body
            # Skip docstring-only bodies (docstring then pass/ellipsis)
            real_body = body
            if (
                len(body) >= 1
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                real_body = body[1:]

            if len(real_body) == 1 and isinstance(real_body[0], ast.Expr):
                expr = real_body[0].value
                # bare ...
                if isinstance(expr, ast.Constant) and expr.value is ...:
                    violations.append(PlaceholderViolation(
                        line=node.lineno,
                        kind="empty_function_body",
                        description=f"Function '{node.name}' has a bare `...` body (ellipsis placeholder).",
                    ))
            elif len(real_body) == 1 and isinstance(real_body[0], ast.Pass):
                violations.append(PlaceholderViolation(
                    line=node.lineno,
                    kind="empty_function_body",
                    description=f"Function '{node.name}' has a bare `pass` body.",
                ))

        return violations

    def _is_legit_pass_context(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        tree: ast.AST,
    ) -> bool:
        """Return True if this function is a legit stub context."""
        # Check if inside a class that inherits from Exception/Protocol/ABC
        for parent in ast.walk(tree):
            if not isinstance(parent, ast.ClassDef):
                continue
            if func_node not in ast.walk(parent):
                continue
            for base in parent.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in {
                    "Exception",
                    "BaseException",
                    "Error",
                    "Protocol",
                    "ABC",
                    "ABCMeta",
                }:
                    return True
        return False

    def _has_abstractmethod(
        self, func_node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                return True
            if isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                return True
        return False

    def _check_regex(
        self, content: str, ext: str, skip_pass: bool = False
    ) -> list[PlaceholderViolation]:
        violations: list[PlaceholderViolation] = []
        patterns = _BANNED_PATTERNS.get(ext, [])
        for pattern in patterns:
            for match in pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                violations.append(
                    PlaceholderViolation(
                        line=line_num,
                        kind="placeholder_string",
                        description=f"Pattern '{pattern.pattern}' matched: {match.group()!r}",
                    )
                )
        return violations
