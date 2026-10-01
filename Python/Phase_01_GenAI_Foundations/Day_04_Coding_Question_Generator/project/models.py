"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: models.py
Description: Pydantic schemas and data models for coding problems, test cases,
             code submissions, complexity analyses, and grading reports.
=============================================================================
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    """Problem difficulty levels with calibrated acceptance thresholds."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"


class ProblemTopic(str, Enum):
    """Core algorithmic and data structure domains."""
    ARRAYS = "Arrays & Hashing"
    TWO_POINTERS = "Two Pointers"
    SLIDING_WINDOW = "Sliding Window"
    STACK_QUEUE = "Stack & Queue"
    BINARY_SEARCH = "Binary Search"
    LINKED_LIST = "Linked List"
    TREES = "Trees & Binary Search Trees"
    GRAPHS = "Graphs & BFS/DFS"
    HEAP_PRIORITY_QUEUE = "Heap & Priority Queue"
    DYNAMIC_PROGRAMMING = "Dynamic Programming"
    GREEDY = "Greedy Algorithms"
    BACKTRACKING = "Backtracking & Recursion"
    STRINGS = "String Manipulation"
    BIT_MANIPULATION = "Bit Manipulation"
    MATH_GEOMETRY = "Math & Number Theory"
    SYSTEM_DESIGN = "System Design & OOP"


class ProgrammingLanguage(str, Enum):
    """Supported target programming languages for question generation."""
    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    JAVA = "Java"
    CPP = "C++"
    GO = "Go"
    RUST = "Rust"


class CompanyTrack(str, Enum):
    """Target interview tracks to calibrate question flavor and style."""
    FAANG_MAANG = "FAANG / Big Tech"
    STARTUP = "High-Growth Startup"
    FINTECH = "FinTech / Quant"
    GENERAL = "General Software Engineering"


class TestCase(BaseModel):
    """Represents a single input/output test case for validation."""
    inputs: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping function argument names to their respective values"
    )
    expected_output: Any = Field(
        ...,
        description="The strictly expected return value from the candidate's solution"
    )
    is_hidden: bool = Field(
        default=False,
        description="True if this test case is kept private to prevent hardcoding"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of why this output is expected"
    )


class ComplexityAnalysis(BaseModel):
    """Big-O theoretical complexity breakdown."""
    time_complexity: str = Field(
        ...,
        description="Optimal Big-O time complexity (e.g., O(N), O(N log N))"
    )
    space_complexity: str = Field(
        ...,
        description="Optimal Big-O auxiliary space complexity (e.g., O(1), O(N))"
    )
    explanation: str = Field(
        ...,
        description="Detailed mathematical justification of time and space bounds"
    )


class CodingProblem(BaseModel):
    """Complete specification of an AI-generated coding question."""
    id: str = Field(
        ...,
        description="Unique slug identifier (e.g., 'two-sum-closest', 'max-sliding-window')"
    )
    title: str = Field(
        ...,
        description="Descriptive, engaging problem title"
    )
    difficulty: DifficultyLevel = Field(
        ...,
        description="Calibrated difficulty level"
    )
    topic: ProblemTopic = Field(
        ...,
        description="Primary algorithm or data structure domain"
    )
    language: ProgrammingLanguage = Field(
        default=ProgrammingLanguage.PYTHON,
        description="Target programming language for starter code"
    )
    company_style: Optional[CompanyTrack] = Field(
        default=CompanyTrack.GENERAL,
        description="Interview style and flavor"
    )
    description: str = Field(
        ...,
        description="Full markdown problem statement with story, motivation, and task"
    )
    constraints: List[str] = Field(
        ...,
        description="Strict input bounds (e.g., '1 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9')"
    )
    function_name: str = Field(
        ...,
        description="Exact name of the solution function candidates must implement"
    )
    starter_code: str = Field(
        ...,
        description="Boilerplate code provided to the candidate with function signature and type hints"
    )
    test_cases: List[TestCase] = Field(
        ...,
        description="List of public examples and private hidden test cases"
    )
    optimal_solution: str = Field(
        ...,
        description="Fully working, optimal solution in the target programming language"
    )
    brute_force_solution: Optional[str] = Field(
        default=None,
        description="Naive/brute-force approach for comparison and benchmarking"
    )
    complexity: ComplexityAnalysis = Field(
        ...,
        description="Optimal time and space complexity analysis"
    )
    hints: List[str] = Field(
        ...,
        description="Three progressive hints: 1 (Conceptual), 2 (Data Structure), 3 (Algorithm)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable topic tags (e.g., ['hash-table', 'array', 'amazon'])"
    )


class TestExecutionResult(BaseModel):
    """Result of running the candidate code against a single test case."""
    test_index: int
    inputs: Dict[str, Any]
    expected_output: Any
    actual_output: Optional[Any] = None
    passed: bool
    is_hidden: bool
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    stdout: Optional[str] = None


class EvaluationReport(BaseModel):
    """Comprehensive candidate grading report across all test cases."""
    problem_id: str
    problem_title: str
    all_passed: bool
    passed_count: int
    total_count: int
    score_percentage: float
    total_execution_time_ms: float
    results: List[TestExecutionResult]
    code_smells: List[str] = Field(default_factory=list)
    ai_feedback: Optional[str] = None
    suggested_improvements: List[str] = Field(default_factory=list)
