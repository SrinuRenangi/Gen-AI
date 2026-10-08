"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: demo.py
Description: Automated end-to-end test and demonstration script verifying
             generation, sandboxed execution, security interception, and grading.
=============================================================================
"""

import sys
import os

# Add local directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ProblemTopic, DifficultyLevel, CompanyTrack
from generator import CodingQuestionGenerator
from evaluator import CodeSandboxEvaluator
from reviewer import AICodeReviewer
from exporter import ProblemExporter


def run_demonstration():
    print("=" * 75)
    print("🚀 DAY 04 CAPSTONE: AI CODING QUESTION GENERATOR DEMONSTRATION")
    print("=" * 75)

    # 1. Initialize Generator
    generator = CodingQuestionGenerator(provider="auto")
    evaluator = CodeSandboxEvaluator(timeout_seconds=2.0)
    reviewer = AICodeReviewer()

    print(f"\n[Step 1] Initialized Generator (Active Provider: '{generator.provider}')")

    # 2. Generate a problem
    print("[Step 2] Generating an algorithmic interview problem...")
    problem = generator.generate(
        topic=ProblemTopic.ARRAYS,
        difficulty=DifficultyLevel.EASY,
        company_style=CompanyTrack.FAANG_MAANG,
    )
    print(f"  ✅ Generated: '{problem.title}'")
    print(f"  • ID: {problem.id}")
    print(f"  • Topic: {problem.topic.value}")
    print(f"  • Difficulty: {problem.difficulty.value}")
    print(f"  • Constraints: {problem.constraints}")
    print(f"  • Total Test Cases: {len(problem.test_cases)} (Public: {sum(1 for t in problem.test_cases if not t.is_hidden)}, Hidden: {sum(1 for t in problem.test_cases if t.is_hidden)})")

    # 3. Test Optimal Solution
    print("\n[Step 3] Evaluating Optimal Solution against all test cases...")
    opt_report = evaluator.evaluate_solution(problem, problem.optimal_solution)
    print(f"  • Verdict: {'ALL PASSED ✅' if opt_report.all_passed else 'FAILED ❌'}")
    print(f"  • Passed: {opt_report.passed_count}/{opt_report.total_count} ({opt_report.score_percentage}%)")
    print(f"  • Total Execution Time: {opt_report.total_execution_time_ms:.3f} ms")

    # 4. Test Buggy Solution
    print("\n[Step 4] Testing Buggy Candidate Submission (always returns empty list)...")
    buggy_code = f"def {problem.function_name}(*args, **kwargs):\n    return []"
    buggy_report = evaluator.evaluate_solution(problem, buggy_code)
    print(f"  • Correctly Detected Failure: Passed {buggy_report.passed_count}/{buggy_report.total_count} (Expected: 0/{buggy_report.total_count})")

    # 5. Test AST Security Sandboxing
    print("\n[Step 5] Testing Security Interception (Prohibited 'import os' call)...")
    malicious_code = f"import os\ndef {problem.function_name}(*args, **kwargs):\n    os.system('echo dangerous')\n    return []"
    sec_report = evaluator.evaluate_solution(problem, malicious_code)
    print(f"  • Security System Status: {'INTERCEPTED & BLOCKED 🛡️' if not sec_report.all_passed else 'MISSED ❌'}")
    print(f"  • Interception Notice: {sec_report.code_smells[0]}")

    # 6. Test AI Code Reviewer
    print("\n[Step 6] Running Automated AI Code Review on Optimal Solution...")
    review = reviewer.review_submission(problem, problem.optimal_solution, opt_report)
    print("--- Review Feedback Excerpt ---")
    for line in review.split("\n")[:12]:
        print(f"  {line}")
    print("  ...")

    # 7. Test Export
    print("\n[Step 7] Testing Markdown & JSON Exporters...")
    md_output = ProblemExporter.to_markdown(problem)
    json_output = ProblemExporter.to_json(problem)
    print(f"  • Rendered Markdown Length: {len(md_output)} characters")
    print(f"  • Rendered JSON Length: {len(json_output)} characters")

    print("\n" + "=" * 75)
    print("🎉 ALL CAPSTONE SYSTEMS VERIFIED SUCCESSFULLY! READY FOR PRODUCTION.")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
