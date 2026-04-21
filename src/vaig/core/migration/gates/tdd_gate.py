"""TDD Gate: enforces RED/GREEN/REFACTOR cycle for test code."""
import ast
from enum import Enum

from vaig.core.migration.domain import Chunk
from vaig.core.migration.gates.base import GateResult, QualityGate

__all__ = ["TddPhase", "TddGate"]


class TddPhase(Enum):
    RED = "red"
    GREEN = "green"
    REFACTOR = "refactor"


def _is_test_file(chunk: Chunk) -> bool:
    return "test" in chunk.node.source_file.lower()


def _parse_functions(code: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(code)
    except (KeyboardInterrupt, SystemExit):
        raise
    except SyntaxError:
        return []
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _is_trivial_body(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if body is pass / ... / raise NotImplementedError only."""
    body = func.body
    # strip docstring
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) == 0:
        return True
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
            return True
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc
            if exc is None:
                return True
            if isinstance(exc, ast.Call):
                func_node = exc.func
                name = (
                    func_node.id
                    if isinstance(func_node, ast.Name)
                    else getattr(func_node, "attr", "")
                )
                if name == "NotImplementedError":
                    return True
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
    return False


def _count_lines(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    end = getattr(func, "end_lineno", func.lineno)
    return end - func.lineno + 1


class TddGate(QualityGate):
    """Enforces TDD cycle for generated test code."""

    def check(
        self,
        chunk: Chunk,
        generated_code: str,
        phase: TddPhase = TddPhase.GREEN,
    ) -> GateResult:
        if not _is_test_file(chunk):
            return GateResult(passed=True, notes="not a test file")

        funcs = _parse_functions(generated_code)
        test_funcs = [f for f in funcs if f.name.startswith("test_")]

        if phase == TddPhase.RED:
            return self._check_red(test_funcs)
        elif phase == TddPhase.GREEN:
            return self._check_green(test_funcs)
        else:  # REFACTOR
            return self._check_refactor(test_funcs, funcs, generated_code)

    def _check_red(
        self, test_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef]
    ) -> GateResult:
        violations: list[str] = []
        if not test_funcs:
            violations.append("RED phase: no test stubs found (functions starting with test_)")
            return GateResult(passed=False, violations=violations)

        for f in test_funcs:
            if not _is_trivial_body(f):
                violations.append(
                    f"RED phase: test '{f.name}' has a non-trivial body — expected stub only"
                )

        return GateResult(passed=len(violations) == 0, violations=violations)

    def _check_green(
        self, test_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef]
    ) -> GateResult:
        violations: list[str] = []
        if not test_funcs:
            violations.append("GREEN phase: no test functions found")
            return GateResult(passed=False, violations=violations)

        for f in test_funcs:
            if _is_trivial_body(f):
                violations.append(
                    f"GREEN phase: test '{f.name}' has no implementation (trivial body)"
                )

        return GateResult(passed=len(violations) == 0, violations=violations)

    def _check_refactor(
        self,
        test_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef],
        all_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef],
        generated_code: str,
    ) -> GateResult:
        # First run GREEN checks
        result = self._check_green(test_funcs)
        violations = list(result.violations)

        # Check duplicate function names (scope-aware: qualify by parent class/function)
        try:
            tree = ast.parse(generated_code)
        except SyntaxError:
            tree = None

        if tree is not None:
            seen: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            scoped_name = f"{node.name}.{child.name}"
                            if scoped_name in seen:
                                violations.append(
                                    f"REFACTOR phase: duplicate function name '{child.name}' in '{node.name}'"
                                )
                            seen.add(scoped_name)
            # Also check top-level function duplicates
            top_names: set[str] = set()
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in top_names:
                        violations.append(f"REFACTOR phase: duplicate function name '{node.name}'")
                    top_names.add(node.name)

        # Check functions > 50 lines
        for f in all_funcs:
            line_count = _count_lines(f)
            if line_count > 50:
                violations.append(
                    f"REFACTOR phase: function '{f.name}' is {line_count} lines (> 50)"
                )

        return GateResult(passed=len(violations) == 0, violations=violations)
