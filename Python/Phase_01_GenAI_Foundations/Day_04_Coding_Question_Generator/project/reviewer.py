"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: reviewer.py
Description: AI Code Reviewer and Interview Feedback Agent. Evaluates Big-O complexity,
             code smells, style, and progressive hint delivery.
=============================================================================
"""

from __future__ import annotations
import ast
import os
from typing import List, Optional

from models import CodingProblem, EvaluationReport
from prompts import CODE_REVIEW_PROMPT


class AICodeReviewer:
    """Interviewer agent providing deep code review, complexity analysis, and guidance."""

    def __init__(self, provider: str = "auto"):
        self.provider = provider.lower()
        if self.provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
            else:
                self.provider = "rule_based"

    def review_submission(
        self,
        problem: CodingProblem,
        submitted_code: str,
        report: EvaluationReport,
    ) -> str:
        """Generate comprehensive review feedback for candidate submission."""
        if self.provider == "openai":
            return self._review_openai(problem, submitted_code, report)
        elif self.provider == "gemini":
            return self._review_gemini(problem, submitted_code, report)
        else:
            return self._review_rule_based(problem, submitted_code, report)

    def _review_openai(self, problem: CodingProblem, code: str, report: EvaluationReport) -> str:
        """Call OpenAI for review."""
        try:
            from openai import OpenAI
            client = OpenAI()
            failed_summary = "\n".join(
                f"- Test {r.test_index}: Expected {r.expected_output}, got {r.actual_output} (Error: {r.error_message})"
                for r in report.results if not r.passed
            ) or "None (All test cases passed successfully!)"

            prompt = CODE_REVIEW_PROMPT.format(
                problem_title=problem.title,
                difficulty=problem.difficulty.value,
                topic=problem.topic.value,
                submitted_code=code,
                passed_count=report.passed_count,
                total_count=report.total_count,
                failed_cases_summary=failed_summary,
            )

            res = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a Principal Software Engineer conducting a thorough code review."},
                    {"role": "user", "content": prompt},
                ],
            )
            return res.choices[0].message.content or "No feedback generated."
        except Exception:
            return self._review_rule_based(problem, code, report)

    def _review_gemini(self, problem: CodingProblem, code: str, report: EvaluationReport) -> str:
        """Call Gemini for review."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            failed_summary = "\n".join(
                f"- Test {r.test_index}: Expected {r.expected_output}, got {r.actual_output} (Error: {r.error_message})"
                for r in report.results if not r.passed
            ) or "None (All test cases passed successfully!)"

            prompt = CODE_REVIEW_PROMPT.format(
                problem_title=problem.title,
                difficulty=problem.difficulty.value,
                topic=problem.topic.value,
                submitted_code=code,
                passed_count=report.passed_count,
                total_count=report.total_count,
                failed_cases_summary=failed_summary,
            )
            res = model.generate_content(prompt)
            return res.text
        except Exception:
            return self._review_rule_based(problem, code, report)

    def _estimate_ast_complexity(self, code_str: str) -> str:
        """Heuristic AST analysis of loop nesting depth and recursion."""
        try:
            tree = ast.parse(code_str)
        except Exception:
            return "Unknown (Syntax error prevented AST parsing)"

        max_loop_depth = 0
        has_recursion = False
        defined_functions = set()

        # Collect function names
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_functions.add(node.name)

        def get_loop_depth(node: ast.AST, current_depth: int) -> int:
            nonlocal has_recursion
            depths = [current_depth]
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    depths.append(get_loop_depth(child, current_depth + 1))
                elif isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id in defined_functions:
                        has_recursion = True
                    depths.append(get_loop_depth(child, current_depth))
                else:
                    depths.append(get_loop_depth(child, current_depth))
            return max(depths)

        max_loop_depth = get_loop_depth(tree, 0)

        if has_recursion:
            return "Recursive branching (likely O(2^N) or O(N) depending on memoization)"
        if max_loop_depth == 0:
            return "O(1) Constant Time"
        elif max_loop_depth == 1:
            return "O(N) Linear Time"
        elif max_loop_depth == 2:
            return "O(N^2) Quadratic Time"
        else:
            return f"O(N^{max_loop_depth}) Polynomial Time"

    def _review_rule_based(
        self,
        problem: CodingProblem,
        submitted_code: str,
        report: EvaluationReport,
    ) -> str:
        """Local heuristic evaluation without requiring an API key."""
        estimated_complexity = self._estimate_ast_complexity(submitted_code)
        
        status_line = (
            "🎉 **OUTSTANDING!** All test cases passed successfully."
            if report.all_passed
            else f"⚠️ **NEEDS IMPROVEMENT**: Passed {report.passed_count}/{report.total_count} test cases."
        )

        review = [
            f"### 📋 Candidate Code Evaluation Report",
            f"**Problem:** {problem.title} ({problem.difficulty.value})",
            f"**Status:** {status_line}",
            f"**Execution Runtime:** {report.total_execution_time_ms:.2f} ms total across all suites",
            "",
            "#### 1. ⏱️ Big-O Complexity Analysis",
            f"- **Estimated Time Complexity:** `{estimated_complexity}`",
            f"- **Theoretical Optimal Target:** `{problem.complexity.time_complexity}`",
            f"- **Target Space Complexity:** `{problem.complexity.space_complexity}`",
            f"- **Complexity Rationale:** {problem.complexity.explanation}",
            "",
            "#### 2. 🔍 Test Case Breakdown",
        ]

        for res in report.results:
            badge = "✅ PASSED" if res.passed else "❌ FAILED"
            visibility = "[Hidden Case]" if res.is_hidden else "[Public Example]"
            review.append(
                f"- **Test {res.test_index}** {visibility} — {badge} ({res.execution_time_ms} ms)"
            )
            if not res.passed:
                review.append(f"  * **Inputs:** `{res.inputs}`")
                review.append(f"  * **Expected:** `{res.expected_output}`")
                review.append(f"  * **Actual Output:** `{res.actual_output}`")
                if res.error_message:
                    review.append(f"  * **Error Detail:** `{res.error_message}`")

        if report.code_smells:
            review.append("")
            review.append("#### 3. 🚨 Code Smells & Warnings")
            for smell in report.code_smells:
                review.append(f"- ⚠️ {smell}")

        review.append("")
        review.append("#### 4. 💡 Progressive Hints for Optimization")
        for i, hint in enumerate(problem.hints, start=1):
            review.append(f"- **Hint {i}:** {hint}")

        return "\n".join(review)
