"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: prompts.py
Description: Production prompt engineering templates, few-shot demonstrations,
             schema enforcement instructions, and AI code review directives.
=============================================================================
"""

SYSTEM_QUESTION_ARCHITECT = """You are a Principal Software Engineer and Staff Interview Architect at a top tier technology firm (FAANG/MAANG).
Your mission is to generate novel, mathematically sound, highly engaging coding interview questions that rigorously test candidates on:
1. Algorithmic thinking and time/space complexity optimization.
2. Handling tricky edge cases (empty inputs, single elements, integer overflow, negatives, duplicates).
3. Writing clean, idiomatic, and maintainable code.

CRITICAL QUALITY DIRECTIVES:
- NO DUPLICATE CLONES: Do not simply copy LeetCode #1 Two Sum word-for-word. Frame novel scenarios, engaging story backdrops, or creative algorithmic twists.
- STRICT TYPE SAFETY: Starter code and optimal solutions must include proper typing, clean docstrings, and descriptive variable names.
- EXHAUSTIVE TEST SUITE: Generate at least 5 test cases:
  * 2 Public Example test cases with clear step-by-step explanations.
  * 3 Hidden test cases specifically targeting boundary conditions (e.g. minimum bounds, maximum bounds, negative numbers, all elements equal).
- DETERMINISTIC OUTPUT: You must output ONLY a valid JSON object strictly conforming to the requested schema. No conversational filler, no markdown wrappers, no trailing notes.
"""

FEW_SHOT_PROBLEM_EXAMPLE = {
    "id": "reorganize-server-cluster",
    "title": "Reorganize Server Cluster Tasks",
    "difficulty": "Medium",
    "topic": "Heap & Priority Queue",
    "language": "Python",
    "company_style": "FAANG / Big Tech",
    "description": "You are managing a high-performance cloud datacenter. You are given a string `tasks` where each character represents a distinct task type, and an integer `k` representing the minimum cooling delay required between two identical task types.\n\nReturn the minimum total CPU cycles required to complete all tasks, or return `\"\"` if it is impossible to schedule them without violating the cooling constraint.",
    "constraints": [
        "1 <= tasks.length <= 10^5",
        "tasks consists only of uppercase English letters 'A' through 'Z'",
        "0 <= k <= 100"
    ],
    "function_name": "reorganize_tasks",
    "starter_code": "def reorganize_tasks(tasks: str, k: int) -> str:\n    # TODO: Implement optimal task scheduling\n    pass",
    "test_cases": [
        {
            "inputs": {"tasks": "AAABBB", "k": 2},
            "expected_output": "AB.AB.AB",
            "is_hidden": False,
            "explanation": "With k=2 cooling interval, identical tasks 'A' must be separated by at least 2 other tasks or idle cycles."
        },
        {
            "inputs": {"tasks": "AAABB", "k": 0},
            "expected_output": "AAABB",
            "is_hidden": False,
            "explanation": "When cooling delay k is 0, tasks can execute in any order."
        },
        {
            "inputs": {"tasks": "A", "k": 5},
            "expected_output": "A",
            "is_hidden": True,
            "explanation": "Single task requires no cooldown."
        },
        {
            "inputs": {"tasks": "AAAA", "k": 3},
            "expected_output": "A...A...A...A",
            "is_hidden": True,
            "explanation": "Maximum cooling delay between identical elements."
        },
        {
            "inputs": {"tasks": "ABCDEF", "k": 2},
            "expected_output": "ABCDEF",
            "is_hidden": True,
            "explanation": "All distinct tasks can be executed consecutively without any delays."
        }
    ],
    "optimal_solution": "from collections import Counter\nimport heapq\n\ndef reorganize_tasks(tasks: str, k: int) -> str:\n    if not tasks: return ''\n    if k == 0: return tasks\n    counts = Counter(tasks)\n    max_heap = [(-cnt, char) for char, cnt in counts.items()]\n    heapq.heapify(max_heap)\n    result = []\n    queue = []\n    time = 0\n    while max_heap or queue:\n        time += 1\n        if max_heap:\n            cnt, char = heapq.heappop(max_heap)\n            result.append(char)\n            if cnt + 1 < 0:\n                queue.append((time + k, cnt + 1, char))\n        else:\n            result.append('.')\n        if queue and queue[0][0] <= time:\n            _, nxt_cnt, nxt_char = queue.pop(0)\n            heapq.heappush(max_heap, (nxt_cnt, nxt_char))\n    return ''.join(result)",
    "brute_force_solution": "# Permutations approach checking all valid interleave orderings: O(N!) factorial time.",
    "complexity": {
        "time_complexity": "O(N log U) where N is task length and U is unique task count",
        "space_complexity": "O(U) for frequency counter and max heap storage",
        "explanation": "Heap operations take O(log 26) = O(1) time since there are at most 26 uppercase English letters."
    },
    "hints": [
        "Focus on the most frequent task first (greedy priority). Which data structure gives instant access to maximum elements?",
        "Use a Max Heap to greedily pick the available task with the highest remaining frequency.",
        "Maintain a cooling queue of pairs `(ready_timestamp, remaining_count, task_char)` to defer re-inserting tasks into the heap until `k` cycles elapse."
    ],
    "tags": ["heap", "greedy", "hash-table", "queue", "amazon", "google"]
}

QUESTION_GENERATION_PROMPT = """Create an original coding interview problem tailored to these exact specifications:
- Domain Topic: {topic}
- Difficulty: {difficulty}
- Programming Language: {language}
- Target Company Profile: {company_style}
{custom_instruction}

STRICT JSON OUTPUT FORMAT:
You must respond with a single valid JSON object matching this schema:
{{
  "id": "slug-name-hyphenated",
  "title": "Descriptive Problem Title",
  "difficulty": "{difficulty}",
  "topic": "{topic}",
  "language": "{language}",
  "company_style": "{company_style}",
  "description": "Full problem description in markdown with context, problem definition, and explicit requirements.",
  "constraints": ["Constraint 1", "Constraint 2", "Constraint 3"],
  "function_name": "function_name_in_snake_case",
  "starter_code": "def function_name(arg1: type, arg2: type) -> return_type:\\n    # Candidate code here\\n    pass",
  "test_cases": [
    {{
      "inputs": {{"arg1": value1, "arg2": value2}},
      "expected_output": expected_val,
      "is_hidden": false,
      "explanation": "Why this output is produced"
    }},
    {{
      "inputs": {{"arg1": value3, "arg2": value4}},
      "expected_output": expected_val2,
      "is_hidden": true,
      "explanation": "Edge case boundary test"
    }}
  ],
  "optimal_solution": "Complete verified optimal Python solution code",
  "brute_force_solution": "Brief description or naive code snippet",
  "complexity": {{
    "time_complexity": "Big-O notation",
    "space_complexity": "Big-O notation",
    "explanation": "Detailed explanation of mathematical bounds"
  }},
  "hints": [
    "Hint 1: Conceptual intuition without giving away the answer",
    "Hint 2: Relevant data structure or mathematical property",
    "Hint 3: High-level algorithm blueprint"
  ],
  "tags": ["tag1", "tag2", "tag3"]
}}

REMEMBER: Respond with ONLY the JSON object. Do not include markdown codeblocks (```json ... ```) or conversational commentary.
"""

CODE_REVIEW_PROMPT = """You are an automated Senior Technical Interviewer evaluating a candidate's submitted solution.

PROBLEM:
Title: {problem_title}
Difficulty: {difficulty}
Topic: {topic}

CANDIDATE CODE SUBMISSION:
```python
{submitted_code}
```

TEST RESULTS:
Passed {passed_count} of {total_count} test cases.
Failed Cases (if any):
{failed_cases_summary}

EVALUATION TASK:
Analyze the candidate's code across 4 core dimensions:
1. Algorithmic Correctness: Did the solution pass all edge cases? If not, why?
2. Big-O Complexity: What is the ACTUAL time and auxiliary space complexity of their submission? Compare it to the theoretical optimal.
3. Code Quality & Idiomatic Style: Readability, variable naming, defensive handling, modularity.
4. Next-Level Optimization: How can the candidate refactor or optimize their solution further?

Provide constructive, encouraging, high-caliber feedback in clear markdown.
"""
