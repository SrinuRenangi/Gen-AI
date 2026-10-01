"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: cli.py
Description: Interactive command-line terminal application with color formatting,
             dynamic problem generation, live code execution, and test grading.
=============================================================================
"""

import sys
import os
import argparse
from typing import Optional

# Ensure project modules can be loaded from current working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    CodingProblem,
    CompanyTrack,
    DifficultyLevel,
    ProblemTopic,
    ProgrammingLanguage,
)
from generator import CodingQuestionGenerator
from evaluator import CodeSandboxEvaluator
from reviewer import AICodeReviewer
from exporter import ProblemExporter

# Optional rich library integration for high-end CLI graphics
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def print_banner():
    if HAS_RICH:
        console.print(
            Panel.fit(
                "[bold cyan]⚡ AI Coding Question & Assessment Generator[/bold cyan]\n"
                "[dim]Zero to Hero GenAI Course — Capstone Project (Day 04)[/dim]\n"
                "[green]Features: Multi-LLM, Sandboxed Code Execution, AST Security & Auto-Grading[/green]",
                border_style="magenta",
            )
        )
    else:
        print("=" * 70)
        print("   AI Coding Question & Assessment Generator (Day 04)")
        print("   Multi-LLM, Sandbox Code Evaluation & Auto-Grading")
        print("=" * 70)


def display_problem(problem: CodingProblem):
    if HAS_RICH:
        console.print(f"\n[bold yellow]📌 Problem Title:[/bold yellow] [bold white]{problem.title}[/bold white]")
        console.print(f"[cyan]Difficulty:[/cyan] {problem.difficulty.value} | [magenta]Topic:[/magenta] {problem.topic.value} | [green]Target:[/green] {problem.company_style.value}")
        console.print(Panel(Markdown(problem.description), title="[bold]Problem Description[/bold]", border_style="blue"))
        
        console.print("\n[bold]Constraints:[/bold]")
        for c in problem.constraints:
            console.print(f"  • [yellow]{c}[/yellow]")
            
        console.print("\n[bold]Public Test Cases:[/bold]")
        table = Table(title="Sample Test Cases")
        table.add_column("Case #", justify="center", style="cyan")
        table.add_column("Inputs", style="white")
        table.add_column("Expected Output", style="green")
        table.add_column("Type", justify="center", style="magenta")

        for idx, tc in enumerate(problem.test_cases, start=1):
            visibility = "Hidden" if tc.is_hidden else "Public"
            table.add_row(str(idx), str(tc.inputs), str(tc.expected_output), visibility)
        console.print(table)
    else:
        print(f"\n--- {problem.title} [{problem.difficulty.value}] ---")
        print(f"Topic: {problem.topic.value} | Style: {problem.company_style.value}")
        print("\nDescription:")
        print(problem.description)
        print("\nConstraints:")
        for c in problem.constraints:
            print(f"- {c}")
        print("\nTest Cases:")
        for idx, tc in enumerate(problem.test_cases, start=1):
            tag = "[HIDDEN]" if tc.is_hidden else "[PUBLIC]"
            print(f"  {idx}. {tag} Inputs: {tc.inputs} -> Expected: {tc.expected_output}")


def run_cli_interactive():
    print_banner()
    generator = CodingQuestionGenerator(provider="auto")
    evaluator = CodeSandboxEvaluator(timeout_seconds=2.0)
    reviewer = AICodeReviewer()

    current_problem: Optional[CodingProblem] = None

    while True:
        print("\n" + "=" * 50)
        print("Main Menu:")
        print("1. 🎲 Generate New Coding Question")
        print("2. 📄 View Current Problem Statement")
        print("3. 💻 View Starter Code Boilerplate")
        print("4. 🧪 Run Test Cases (Test Optimal Solution)")
        print("5. ✏️  Submit Your Custom Python Solution")
        print("6. 💡 Progressive Hints (Levels 1, 2, 3)")
        print("7. 💾 Export Problem to Markdown / JSON")
        print("8. 🚪 Exit")
        print("=" * 50)

        choice = input("Select an option (1-8): ").strip()

        if choice == "1":
            print("\nSelect Topic:")
            topics = list(ProblemTopic)
            for i, t in enumerate(topics, 1):
                print(f"  {i}. {t.value}")
            t_choice = input(f"Choose (1-{len(topics)}) [Default: 1]: ").strip()
            topic = topics[int(t_choice) - 1] if t_choice.isdigit() and 1 <= int(t_choice) <= len(topics) else ProblemTopic.ARRAYS

            print("\nSelect Difficulty:")
            diffs = list(DifficultyLevel)
            for i, d in enumerate(diffs, 1):
                print(f"  {i}. {d.value}")
            d_choice = input(f"Choose (1-{len(diffs)}) [Default: 2]: ").strip()
            diff = diffs[int(d_choice) - 1] if d_choice.isdigit() and 1 <= int(d_choice) <= len(diffs) else DifficultyLevel.MEDIUM

            print("\nGenerating problem...")
            current_problem = generator.generate(topic=topic, difficulty=diff)
            print("Done! Problem generated.")
            display_problem(current_problem)

        elif choice == "2":
            if not current_problem:
                print("⚠️ Please generate a problem first (Option 1).")
                continue
            display_problem(current_problem)

        elif choice == "3":
            if not current_problem:
                print("⚠️ Please generate a problem first.")
                continue
            print("\n--- Starter Code ---")
            print(current_problem.starter_code)

        elif choice == "4":
            if not current_problem:
                print("⚠️ Please generate a problem first.")
                continue
            print("\nTesting verified optimal solution against public & hidden test cases...")
            report = evaluator.evaluate_solution(current_problem, current_problem.optimal_solution)
            
            if HAS_RICH:
                table = Table(title=f"Evaluation Results: {report.passed_count}/{report.total_count} Passed ({report.score_percentage}%)")
                table.add_column("Test #", justify="center")
                table.add_column("Type", justify="center")
                table.add_column("Verdict", justify="center")
                table.add_column("Execution Time", justify="right")

                for r in report.results:
                    tag = "Hidden" if r.is_hidden else "Public"
                    verdict = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
                    table.add_row(str(r.test_index), tag, verdict, f"{r.execution_time_ms} ms")
                console.print(table)
            else:
                print(f"\nVerdict: {report.passed_count}/{report.total_count} Passed ({report.score_percentage}%) in {report.total_execution_time_ms:.2f} ms")
                for r in report.results:
                    status = "PASS" if r.passed else "FAIL"
                    print(f"  Test {r.test_index} ({'Hidden' if r.is_hidden else 'Public'}): {status} ({r.execution_time_ms} ms)")

        elif choice == "5":
            if not current_problem:
                print("⚠️ Please generate a problem first.")
                continue
            print("\nEnter or paste your Python solution code.")
            print("(Type 'END' on a new line when finished):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            submitted_code = "\n".join(lines)
            
            if not submitted_code.strip():
                print("Empty submission.")
                continue

            print("\nEvaluating submitted code in secure sandbox...")
            report = evaluator.evaluate_solution(current_problem, submitted_code)
            feedback = reviewer.review_submission(current_problem, submitted_code, report)
            print("\n" + feedback)

        elif choice == "6":
            if not current_problem:
                print("⚠️ Please generate a problem first.")
                continue
            print("\nProgressive Hints:")
            for idx, hint in enumerate(current_problem.hints, 1):
                reveal = input(f"Reveal Hint {idx}? (y/n) [Default: y]: ").strip().lower()
                if reveal in ("", "y", "yes"):
                    print(f"  💡 Hint {idx}: {hint}\n")
                else:
                    break

        elif choice == "7":
            if not current_problem:
                print("⚠️ Please generate a problem first.")
                continue
            filename_md = f"{current_problem.id}.md"
            filename_json = f"{current_problem.id}.json"
            ProblemExporter.save_to_file(ProblemExporter.to_markdown(current_problem), filename_md)
            ProblemExporter.save_to_file(ProblemExporter.to_json(current_problem), filename_json)
            print(f"✅ Exported successfully to:\n  - {filename_md}\n  - {filename_json}")

        elif choice == "8":
            print("\nThank you for using AI Coding Question Generator! Happy coding 🚀")
            break
        else:
            print("Invalid choice, please select 1-8.")


if __name__ == "__main__":
    run_cli_interactive()
