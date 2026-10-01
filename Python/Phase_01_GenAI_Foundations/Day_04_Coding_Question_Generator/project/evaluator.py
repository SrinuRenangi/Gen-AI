"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: evaluator.py
Description: Sandboxed Code Evaluation Engine featuring AST security verification,
             timeout enforcement, stdout capturing, and automated test grading.
=============================================================================
"""

from __future__ import annotations
import ast
import contextlib
import io
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

from models import (
    CodingProblem,
    EvaluationReport,
    TestExecutionResult,
)


class SecurityViolationError(Exception):
    """Raised when submitted code contains prohibited modules or malicious calls."""
    pass


class CodeSandboxEvaluator:
    """Secure local test execution harness for algorithmic coding solutions."""

    # Disallowed modules and builtins for security sandboxing
    PROHIBITED_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "ctypes",
        "builtins", "importlib", "pathlib", "posixpath", "ntpath",
        "requests", "urllib", "http", "ftplib", "pickle"
    }
    
    PROHIBITED_FUNCTIONS = {
        "eval", "exec", "__import__", "compile", "open", "getattr", "setattr", "delattr"
    }

    def __init__(self, timeout_seconds: float = 2.0):
        """
        Args:
            timeout_seconds: Hard execution limit per test case (default: 2.0s)
        """
        self.timeout_seconds = timeout_seconds

    def verify_ast_security(self, code_str: str) -> None:
        """
        Perform static AST inspection to detect prohibited imports or dangerous function calls.
        Raises SecurityViolationError if unsafe operations are detected.
        """
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in submitted code: {e}")

        for node in ast.walk(tree):
            # Check import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in self.PROHIBITED_MODULES:
                        raise SecurityViolationError(
                            f"Security Violation: Import of module '{alias.name}' is strictly prohibited in sandbox."
                        )

            # Check from ... import statements
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self.PROHIBITED_MODULES:
                        raise SecurityViolationError(
                            f"Security Violation: Import from module '{node.module}' is strictly prohibited in sandbox."
                        )

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.PROHIBITED_FUNCTIONS:
                        raise SecurityViolationError(
                            f"Security Violation: Call to dangerous built-in '{node.func.id}()' is prohibited."
                        )

    def _deep_equals(self, actual: Any, expected: Any) -> bool:
        """
        Check equality with tolerance for floating point numbers and list representations.
        """
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-5)
            except (TypeError, ValueError):
                return False

        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(self._deep_equals(a, e) for a, e in zip(actual, expected))

        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False
            return all(self._deep_equals(actual[k], expected[k]) for k in expected)

        return actual == expected

    def _execute_single_test(
        self,
        func: Any,
        inputs: Dict[str, Any],
        expected: Any,
        test_index: int,
        is_hidden: bool,
    ) -> TestExecutionResult:
        """Execute a single test case with stdout redirection and timing."""
        stdout_capture = io.StringIO()
        start_time = time.perf_counter()
        passed = False
        error_msg = None
        actual_output = None

        try:
            with contextlib.redirect_stdout(stdout_capture):
                actual_output = func(**inputs)
            passed = self._deep_equals(actual_output, expected)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {str(exc)}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return TestExecutionResult(
            test_index=test_index,
            inputs=inputs,
            expected_output=expected,
            actual_output=actual_output,
            passed=passed,
            is_hidden=is_hidden,
            execution_time_ms=round(elapsed_ms, 3),
            error_message=error_msg,
            stdout=stdout_capture.getvalue().strip() or None,
        )

    def evaluate_solution(
        self,
        problem: CodingProblem,
        submitted_code: str,
    ) -> EvaluationReport:
        """
        Full evaluation pipeline:
        1. Static AST Security Scan
        2. Compilation & Namespace preparation
        3. Timed execution across all public & hidden test cases
        4. Performance & Smell aggregation
        """
        start_total = time.perf_counter()

        # Step 1: Security Inspection
        try:
            self.verify_ast_security(submitted_code)
        except (SecurityViolationError, SyntaxError) as err:
            return EvaluationReport(
                problem_id=problem.id,
                problem_title=problem.title,
                all_passed=False,
                passed_count=0,
                total_count=len(problem.test_cases),
                score_percentage=0.0,
                total_execution_time_ms=0.0,
                results=[],
                code_smells=[str(err)],
                ai_feedback=f"Execution rejected prior to runtime due to verification failure: {err}",
            )

        # Step 2: Namespace execution
        namespace: Dict[str, Any] = {}
        try:
            exec(submitted_code, namespace)
        except Exception as exc:
            return EvaluationReport(
                problem_id=problem.id,
                problem_title=problem.title,
                all_passed=False,
                passed_count=0,
                total_count=len(problem.test_cases),
                score_percentage=0.0,
                total_execution_time_ms=0.0,
                results=[],
                code_smells=["Code raised an exception during module definition."],
                ai_feedback=f"Compilation/Runtime initialization failed: {traceback.format_exc()}",
            )

        target_func = namespace.get(problem.function_name)
        if not callable(target_func):
            return EvaluationReport(
                problem_id=problem.id,
                problem_title=problem.title,
                all_passed=False,
                passed_count=0,
                total_count=len(problem.test_cases),
                score_percentage=0.0,
                total_execution_time_ms=0.0,
                results=[],
                code_smells=[f"Target function '{problem.function_name}' was not defined in the submitted solution."],
                ai_feedback=f"Missing function definition: Ensure your code defines `def {problem.function_name}(...)`.",
            )

        # Step 3: Run all test cases under timeout protection
        test_results: List[TestExecutionResult] = []
        code_smells: List[str] = []

        # Check basic code smells via AST
        if "while True" in submitted_code:
            code_smells.append("Unbounded 'while True' loop detected — potential risk of infinite execution.")

        with ThreadPoolExecutor(max_workers=1) as executor:
            for idx, tc in enumerate(problem.test_cases):
                future = executor.submit(
                    self._execute_single_test,
                    target_func,
                    tc.inputs,
                    tc.expected_output,
                    idx + 1,
                    tc.is_hidden,
                )
                try:
                    res = future.result(timeout=self.timeout_seconds)
                    test_results.append(res)
                except FuturesTimeoutError:
                    test_results.append(
                        TestExecutionResult(
                            test_index=idx + 1,
                            inputs=tc.inputs,
                            expected_output=tc.expected_output,
                            actual_output=None,
                            passed=False,
                            is_hidden=tc.is_hidden,
                            execution_time_ms=self.timeout_seconds * 1000.0,
                            error_message=f"TimeLimitExceeded: Test exceeded limit of {self.timeout_seconds} seconds.",
                        )
                    )

        passed_count = sum(1 for r in test_results if r.passed)
        total_count = len(test_results)
        all_passed = (passed_count == total_count) and total_count > 0
        score_pct = round((passed_count / total_count) * 100.0, 1) if total_count > 0 else 0.0
        total_time_ms = round((time.perf_counter() - start_total) * 1000.0, 3)

        return EvaluationReport(
            problem_id=problem.id,
            problem_title=problem.title,
            all_passed=all_passed,
            passed_count=passed_count,
            total_count=total_count,
            score_percentage=score_pct,
            total_execution_time_ms=total_time_ms,
            results=test_results,
            code_smells=code_smells,
        )
