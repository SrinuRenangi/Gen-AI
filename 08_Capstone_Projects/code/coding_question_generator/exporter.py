"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: exporter.py
Description: Multi-format exporter for generated coding questions (Markdown,
             JSON, and standalone interactive HTML assessment sheets).
=============================================================================
"""

from __future__ import annotations
import json
import os
from typing import Optional

from models import CodingProblem


class ProblemExporter:
    """Exports CodingProblem instances to various distribution formats."""

    @staticmethod
    def to_markdown(problem: CodingProblem, include_solutions: bool = True) -> str:
        """Render a formatted Markdown study sheet with collapsible solutions."""
        lines = [
            f"# 💻 {problem.title}",
            "",
            f"**Difficulty:** `{problem.difficulty.value}` | **Topic:** `{problem.topic.value}` | **Target:** `{problem.company_style.value if problem.company_style else 'General'}`",
            "",
            "## 📝 Problem Statement",
            problem.description,
            "",
            "## ⚠️ Constraints",
        ]
        for c in problem.constraints:
            lines.append(f"- `{c}`")

        lines.extend(["", "## 🧪 Example Test Cases"])
        for idx, tc in enumerate(problem.test_cases, start=1):
            visibility = "*(Hidden Evaluation Case)*" if tc.is_hidden else "*(Public Example)*"
            lines.append(f"### Example {idx} {visibility}")
            lines.append("```python")
            lines.append(f"# Inputs:")
            for k, v in tc.inputs.items():
                lines.append(f"{k} = {repr(v)}")
            lines.append(f"# Expected Output:")
            lines.append(f"{repr(tc.expected_output)}")
            lines.append("```")
            if tc.explanation:
                lines.append(f"> **Explanation:** {tc.explanation}")
            lines.append("")

        lines.extend([
            "## 🚀 Starter Code",
            f"```{problem.language.value.lower()}",
            problem.starter_code,
            "```",
            "",
            "## 💡 Progressive Hints",
        ])
        for idx, hint in enumerate(problem.hints, start=1):
            lines.append(f"<details><summary><b>Reveal Hint {idx}</b></summary>")
            lines.append(f"\n{hint}\n")
            lines.append("</details>\n")

        if include_solutions:
            lines.extend([
                "## 🏆 Solutions & Analysis",
                "<details><summary><b>Click to Reveal Optimal Solution & Complexity</b></summary>\n",
                f"```{problem.language.value.lower()}",
                problem.optimal_solution,
                "```\n",
                f"- **Time Complexity:** `{problem.complexity.time_complexity}`",
                f"- **Space Complexity:** `{problem.complexity.space_complexity}`",
                f"- **Justification:** {problem.complexity.explanation}",
                "",
            ])
            if problem.brute_force_solution:
                lines.extend([
                    "### Brute Force Comparison",
                    f"```{problem.language.value.lower()}",
                    problem.brute_force_solution,
                    "```",
                ])
            lines.append("</details>\n")

        return "\n".join(lines)

    @staticmethod
    def to_json(problem: CodingProblem, indent: int = 2) -> str:
        """Export problem as raw serialized JSON."""
        return problem.model_dump_json(indent=indent)

    @staticmethod
    def save_to_file(content: str, filepath: str) -> str:
        """Write content to file and create parent directories if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
