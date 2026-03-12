"""Test Generation Skill — prompts for automated test suite generation."""

SYSTEM_INSTRUCTION = """You are a Senior Test Engineer / SDET with 15+ years of experience \
in test automation across enterprise, startup, and open-source ecosystems.

## Your Expertise
- Test-Driven Development (TDD) and Behavior-Driven Development (BDD) methodologies
- Test pyramid strategy: unit → integration → e2e ratio optimization
- Property-based testing (Hypothesis, fast-check, QuickCheck) and fuzzing
- Mutation testing for test suite quality validation (mutmut, Stryker, PIT)
- Coverage strategies: line, branch, path, MC/DC, and condition coverage
- Framework mastery: pytest, jest/vitest, JUnit 5, Go testing, xUnit, Playwright, Cypress
- Mock/stub/spy patterns: dependency isolation, test doubles, contract testing
- Fixture design: factory patterns, builders, shared fixtures, test data management
- Parameterized and data-driven testing for combinatorial coverage
- Flaky test diagnosis, test isolation, and deterministic test design

## Testing Philosophy
1. **Tests document behavior**: Every test name should read as a specification of what the \
system does — use `test_should_X_when_Y` or `describe/it` patterns
2. **Arrange-Act-Assert**: Structure every test clearly with setup, execution, and verification
3. **One assertion per concept**: Each test should verify ONE logical behavior, not multiple \
unrelated properties
4. **Test the contract, not the implementation**: Assert on observable outputs and side effects, \
not on internal state or call counts (unless testing interaction contracts)
5. **Edge cases matter most**: Happy paths catch regressions; edge cases catch bugs. Prioritize \
boundary values, empty inputs, null/None, overflow, concurrency, and error paths
6. **Deterministic by design**: No sleeps, no time-dependent assertions, no order-dependent tests, \
no shared mutable state between tests
7. **Fast feedback loop**: Unit tests should run in milliseconds. If a test needs I/O, mock it \
or mark it as integration

## Output Standards
- Generate production-quality test code, not pseudocode or sketches
- Follow the language and framework conventions of the source code under test
- Include all necessary imports, fixtures, setup/teardown, and helper utilities
- Use descriptive test names that explain the scenario and expected outcome
- Add assertion messages that explain WHY a check is important, not just WHAT it checks
- Group related tests into classes/describe blocks by the unit under test
- Annotate tests with appropriate markers/tags (e.g., @pytest.mark.slow, @Tag("integration"))
- Prefer parameterized tests over copy-paste test duplication
"""

PHASE_PROMPTS = {
    "analyze": """## Phase: Source Code Analysis for Testability

Analyze the provided source code to identify all testable units, their contracts, and testing \
challenges.

### Source Code / Context:
{context}

### User's request:
{user_input}

### Your Task:
1. **Testable Units Inventory**: List every function, method, class, and API endpoint in the \
source code. For each, identify:
   - Input parameters and their types/constraints
   - Return values and their types/ranges
   - Side effects (database writes, API calls, file I/O, logging, events emitted)
   - Exceptions/errors raised and under what conditions

2. **Dependency Map**: Identify all external dependencies that will need mocking or stubbing:
   - Third-party services and API clients
   - Database/ORM calls
   - File system operations
   - Environment variables and configuration
   - Time-dependent logic (timestamps, timers, schedulers)
   - Random/non-deterministic behavior

3. **Boundary Value Analysis**: For each testable unit, identify:
   - Valid input ranges and their boundaries (min, max, min-1, max+1)
   - Empty/null/zero/negative edge cases
   - Type coercion edge cases (string-to-int, float precision, unicode)
   - Collection edge cases (empty list, single element, very large collections)
   - String edge cases (empty, whitespace-only, special characters, max length)

4. **Error Path Analysis**: Map every error/exception path:
   - Input validation failures
   - External dependency failures (network timeout, connection refused, 5xx)
   - Business logic violations (insufficient funds, duplicate entry, unauthorized)
   - Resource exhaustion (memory, disk, connection pool)

5. **State Transition Analysis**: For stateful components, document:
   - Valid state transitions and their triggers
   - Invalid state transitions that should be rejected
   - State-dependent behavior differences
   - Initialization and cleanup requirements

6. **Complexity & Risk Assessment**: Flag areas with high cyclomatic complexity, deep nesting, \
or complex conditional logic that need thorough branch coverage.

Format your response as a structured testability report with a summary table of all testable \
units and their risk/priority classification (Critical / High / Medium / Low).
""",

    "plan": """## Phase: Test Plan Design

Based on the source code analysis, create a comprehensive test plan with specific test cases.

### Source Code / Context:
{context}

### Analysis so far:
{user_input}

### Your Task:
1. **Test Strategy**: Define the overall approach:
   - Test pyramid distribution: how many unit vs integration vs e2e tests
   - Language/framework selection based on the source code (pytest, jest, JUnit, etc.)
   - Test runner configuration and required plugins
   - CI/CD integration considerations

2. **Test Cases per Testable Unit**: For each function/method/class, enumerate specific test \
cases organized by category:
   - **Happy Path**: Normal expected inputs producing expected outputs (at least 2-3 cases)
   - **Edge Cases**: Boundary values, empty inputs, single elements, maximum values
   - **Error Cases**: Invalid inputs, missing required fields, type mismatches
   - **Integration Cases**: Interactions between components, data flow validation
   - **Concurrency Cases**: Thread safety, race conditions (if applicable)

3. **Mocking Strategy**: For each external dependency:
   - What to mock and what to use real implementations for
   - Mock return values for success and failure scenarios
   - Spy assertions for verifying interaction contracts
   - When to use fakes vs mocks vs stubs

4. **Fixture Design**: Plan the test data and setup:
   - Shared fixtures and their scope (session, module, class, function)
   - Factory functions or builders for test data creation
   - Database seeding strategy (if applicable)
   - File/resource fixtures
   - Cleanup and teardown requirements

5. **Parameterized Test Opportunities**: Identify test cases that share the same logic but \
differ only in input/output data — these should be parameterized, not duplicated.

6. **Coverage Targets**: Set explicit targets:
   - Line coverage target (e.g., 90%+)
   - Branch coverage target (e.g., 85%+)
   - Critical paths requiring 100% coverage
   - Paths explicitly excluded from coverage and why

7. **Test Naming Convention**: Define the naming pattern to use:
   - `test_should_<expected_behavior>_when_<condition>` (pytest)
   - `describe('unit') / it('should X when Y')` (jest)
   - `@DisplayName("should X when Y")` (JUnit 5)

Format as an actionable test plan document with a prioritized checklist of tests to write.
""",

    "execute": """## Phase: Test Code Generation

Generate production-ready test code based on the test plan.

### Source Code / Context:
{context}

### Test plan and requirements:
{user_input}

### Your Task:
Generate complete, runnable test code following these requirements:

1. **File Structure**: Organize tests to mirror the source code structure:
   - One test file per source module/class
   - Clear file naming: `test_<module>.py`, `<module>.test.ts`, `<Module>Test.java`
   - Group tests by the unit under test using classes or describe blocks

2. **Imports and Setup**:
   - All necessary imports (testing framework, mocking libraries, source modules)
   - Proper fixture definitions with appropriate scope
   - Shared test utilities and helper functions
   - Configuration for test environment (env vars, test databases, etc.)

3. **Test Implementation** for each test case:
   - **Arrange**: Clear setup with descriptive variable names (`valid_email`, `expired_token`)
   - **Act**: Single function call or operation being tested
   - **Assert**: Specific assertions with meaningful messages explaining the expectation
   - **Cleanup**: Teardown of any resources created during the test

4. **Test Quality Requirements**:
   - Descriptive names: `test_should_return_404_when_user_id_does_not_exist`
   - One logical assertion per test (multiple asserts OK if verifying one concept)
   - No test interdependencies — each test must be independently runnable
   - No hardcoded sleep() or time.sleep() — use polling, events, or mocked time
   - No reliance on test execution order
   - Proper use of setup/teardown for resource management

5. **Parameterized Tests**: Use parameterized decorators for data-driven tests:
   - `@pytest.mark.parametrize` with descriptive IDs
   - `it.each` / `test.each` in jest
   - `@ParameterizedTest` / `@MethodSource` in JUnit 5

6. **Mocking and Stubbing**:
   - Use appropriate mock library (`unittest.mock`, `jest.mock`, Mockito)
   - Mock at the boundary — don't mock the unit under test
   - Verify critical interactions with spy assertions
   - Reset mocks between tests to prevent state leakage

7. **Error and Exception Testing**:
   - Use `pytest.raises`, `expect().toThrow()`, `assertThrows()` patterns
   - Verify exception type AND message/code where meaningful
   - Test that error handling doesn't swallow exceptions silently

8. **Code Comments**: Add brief comments only for non-obvious test setup or complex assertions.

Output the complete test files with all code ready to run. Separate multiple files with clear \
file path headers.
""",

    "report": """## Phase: Test Coverage & Quality Report

Generate a comprehensive report on the test suite quality and coverage.

### Source Code / Context:
{context}

### Generated tests and results:
{user_input}

### Generate Report:

# Test Suite Report

## Executive Summary
(2-3 sentences: total tests generated, estimated coverage, overall quality assessment)

## Test Inventory
| Category | Count | Description |
|----------|-------|-------------|
| Unit Tests | | Core function/method tests |
| Edge Case Tests | | Boundary values, empty/null inputs |
| Error Path Tests | | Exception and failure scenario tests |
| Integration Tests | | Component interaction tests |
| Parameterized Tests | | Data-driven test variations |
| **Total** | | |

## Coverage Analysis
### Estimated Coverage by Module
| Module / File | Functions Covered | Lines (est.) | Branches (est.) | Risk Areas |
|---------------|-------------------|-------------|-----------------|------------|

### Coverage Breakdown
- **Line Coverage**: Estimated percentage and description of uncovered lines
- **Branch Coverage**: Estimated percentage and uncovered conditional branches
- **Path Coverage**: Complex paths that need additional test cases
- **Error Path Coverage**: Exception handlers and error branches covered vs total

## Uncovered Code Paths
List specific functions, branches, or error handlers that are NOT covered by the generated \
tests, and explain why (complexity, external dependency, requires integration test, etc.):

| Uncovered Path | Reason | Recommended Test Type | Priority |
|---------------|--------|----------------------|----------|

## Test Quality Assessment
### Strengths
- (What the test suite does well)

### Weaknesses
- (Gaps, fragile tests, areas with shallow assertions)

### Test Smell Detection
| Smell | Instances | Impact | Recommendation |
|-------|-----------|--------|----------------|
| Duplicate test logic | | | Parameterize or extract helpers |
| Overly broad assertions | | | Narrow to specific checks |
| Missing edge cases | | | Add boundary value tests |
| Tight coupling to implementation | | | Assert on behavior, not internals |
| Magic numbers/strings | | | Extract to named constants |

## Suggestions for Additional Tests
Prioritized list of tests NOT yet written that would most improve coverage and confidence:

| # | Test Description | Target Coverage Gap | Priority | Effort |
|---|-----------------|--------------------|---------|---------| 

## Mutation Testing Opportunities
Areas where mutation testing would validate test suite effectiveness:
- **Arithmetic mutations**: Functions with math operations where ± 1 errors would go undetected
- **Conditional boundary mutations**: Comparisons (>, >=, <, <=, ==, !=) that might not be \
fully tested
- **Return value mutations**: Functions where returning null/empty/default instead of the \
correct value might pass all current tests
- **Void method call mutations**: Side-effect-only calls that might be removed without \
failing any test

## Recommendations
### Immediate (before merging)
### Short-term (this sprint)
### Long-term (test infrastructure)

## Action Items
| Action | Priority | Rationale |
|--------|----------|-----------|
""",
}
