"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: generator.py
Description: Universal Question Generation Engine supporting OpenAI, Google Gemini,
             LangChain, and a rich Offline Mock Engine for local experimentation.
=============================================================================
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, Optional

from models import (
    CodingProblem,
    CompanyTrack,
    DifficultyLevel,
    ProblemTopic,
    ProgrammingLanguage,
    TestCase,
    ComplexityAnalysis,
)
from prompts import (
    FEW_SHOT_PROBLEM_EXAMPLE,
    QUESTION_GENERATION_PROMPT,
    SYSTEM_QUESTION_ARCHITECT,
)


class CodingQuestionGenerator:
    """Universal AI generator that produces strictly typed coding interview problems."""

    def __init__(
        self,
        provider: str = "auto",
        model_name: Optional[str] = None,
        temperature: float = 0.4,
    ):
        """
        Initialize the generator.
        
        Args:
            provider: 'openai', 'gemini', 'mock', or 'auto' (detects API key from environment)
            model_name: specific model string (e.g. 'gpt-4o', 'gemini-1.5-flash')
            temperature: sampling temperature (0.3-0.5 recommended for structured outputs)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self._resolve_provider()

    def _resolve_provider(self) -> None:
        """Autodetect available API keys if provider is set to 'auto'."""
        if self.provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
                self.model_name = self.model_name or "gpt-4o-mini"
            elif os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
                self.model_name = self.model_name or "gemini-1.5-flash"
            else:
                self.provider = "mock"

    def generate(
        self,
        topic: ProblemTopic = ProblemTopic.ARRAYS,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
        company_style: CompanyTrack = CompanyTrack.FAANG_MAANG,
        custom_instruction: str = "",
    ) -> CodingProblem:
        """
        Generate a fully formed CodingProblem instance.
        """
        if self.provider == "openai":
            return self._generate_openai(topic, difficulty, language, company_style, custom_instruction)
        elif self.provider == "gemini":
            return self._generate_gemini(topic, difficulty, language, company_style, custom_instruction)
        else:
            return self._generate_mock(topic, difficulty, language, company_style, custom_instruction)

    def _clean_json_output(self, raw_text: str) -> str:
        """Remove markdown code blocks, backticks, and extraneous whitespace."""
        text = raw_text.strip()
        # Remove ```json and ``` wrapping
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text

    def _parse_and_validate(self, raw_json_str: str) -> CodingProblem:
        """Parse raw JSON string into validated CodingProblem instance."""
        cleaned = self._clean_json_output(raw_json_str)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Fallback: attempt to extract JSON object boundaries
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(cleaned[start : end + 1])
            else:
                raise ValueError(f"Failed to parse LLM output as JSON: {exc}\nRaw: {raw_json_str}") from exc

        return CodingProblem(**data)

    def _generate_openai(
        self,
        topic: ProblemTopic,
        difficulty: DifficultyLevel,
        language: ProgrammingLanguage,
        company_style: CompanyTrack,
        custom_instruction: str,
    ) -> CodingProblem:
        """Generate using OpenAI API with Structured Output response format."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = OpenAI()
        model = self.model_name or "gpt-4o-mini"
        prompt = QUESTION_GENERATION_PROMPT.format(
            topic=topic.value,
            difficulty=difficulty.value,
            language=language.value,
            company_style=company_style.value,
            custom_instruction=f"\nAdditional Custom Directives: {custom_instruction}" if custom_instruction else "",
        )

        response = client.chat.completions.create(
            model=model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_QUESTION_ARCHITECT},
                {"role": "user", "content": f"Here is an example of an elite problem format:\n{json.dumps(FEW_SHOT_PROBLEM_EXAMPLE)}"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return self._parse_and_validate(content)

    def _generate_gemini(
        self,
        topic: ProblemTopic,
        difficulty: DifficultyLevel,
        language: ProgrammingLanguage,
        company_style: CompanyTrack,
        custom_instruction: str,
    ) -> CodingProblem:
        """Generate using Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            self.model_name or "gemini-1.5-flash",
            generation_config={"temperature": self.temperature, "response_mime_type": "application/json"},
            system_instruction=SYSTEM_QUESTION_ARCHITECT,
        )

        prompt = QUESTION_GENERATION_PROMPT.format(
            topic=topic.value,
            difficulty=difficulty.value,
            language=language.value,
            company_style=company_style.value,
            custom_instruction=f"\nAdditional Custom Directives: {custom_instruction}" if custom_instruction else "",
        )

        response = model.generate_content(prompt)
        return self._parse_and_validate(response.text)

    def _generate_mock(
        self,
        topic: ProblemTopic,
        difficulty: DifficultyLevel,
        language: ProgrammingLanguage,
        company_style: CompanyTrack,
        custom_instruction: str,
    ) -> CodingProblem:
        """
        Built-in deterministic problem catalog providing realistic, calibrated problems
        for testing and offline development without an API key.
        """
        # Catalog of calibrated algorithmic problems across topics
        catalog: Dict[str, Dict[str, Any]] = {
            "Arrays & Hashing_Easy": {
                "id": "pair-target-product",
                "title": "Find Pair with Exact Target Product",
                "difficulty": DifficultyLevel.EASY,
                "topic": ProblemTopic.ARRAYS,
                "language": language,
                "company_style": company_style,
                "description": (
                    "Given an integer array `nums` and an integer `target`, return a list containing the indices "
                    "of the two numbers such that their product equals `target`.\n\n"
                    "You may assume that each input will have exactly one solution, and you may not use the same element twice. "
                    "Return the indices in ascending order `[i, j]` where `i < j`."
                ),
                "constraints": [
                    "2 <= nums.length <= 10^4",
                    "-10^4 <= nums[i] <= 10^4",
                    "-10^8 <= target <= 10^8",
                    "Exactly one valid pair exists."
                ],
                "function_name": "find_product_pair",
                "starter_code": (
                    "from typing import List\n\n"
                    "def find_product_pair(nums: List[int], target: int) -> List[int]:\n"
                    "    # Return indices [i, j] such that nums[i] * nums[j] == target and i < j\n"
                    "    pass\n"
                ),
                "test_cases": [
                    {
                        "inputs": {"nums": [2, 7, 11, 15, 3], "target": 21},
                        "expected_output": [1, 4],
                        "is_hidden": False,
                        "explanation": "nums[1] * nums[4] = 7 * 3 = 21."
                    },
                    {
                        "inputs": {"nums": [3, 2, 4], "target": 8},
                        "expected_output": [1, 2],
                        "is_hidden": False,
                        "explanation": "nums[1] * nums[2] = 2 * 4 = 8."
                    },
                    {
                        "inputs": {"nums": [-2, 5, -3, 8], "target": 6},
                        "expected_output": [0, 2],
                        "is_hidden": True,
                        "explanation": "Negative numbers: (-2) * (-3) = 6."
                    },
                    {
                        "inputs": {"nums": [0, 10, 0, 5], "target": 0},
                        "expected_output": [0, 1],
                        "is_hidden": True,
                        "explanation": "Zero product: nums[0] * nums[1] = 0 * 10 = 0."
                    },
                    {
                        "inputs": {"nums": [100, 200, 300, 4], "target": 800},
                        "expected_output": [1, 3],
                        "is_hidden": True,
                        "explanation": "nums[1] * nums[3] = 200 * 4 = 800."
                    }
                ],
                "optimal_solution": (
                    "from typing import List\n\n"
                    "def find_product_pair(nums: List[int], target: int) -> List[int]:\n"
                    "    seen = {}\n"
                    "    for i, num in enumerate(nums):\n"
                    "        if num != 0 and target % num == 0:\n"
                    "            complement = target // num\n"
                    "            if complement in seen:\n"
                    "                return [seen[complement], i]\n"
                    "        elif num == 0 and target == 0:\n"
                    "            if 0 in seen:\n"
                    "                return [seen[0], i]\n"
                    "            # Any non-zero paired with zero works if target is 0\n"
                    "            for k, idx in seen.items():\n"
                    "                return [idx, i]\n"
                    "        seen[num] = i\n"
                    "    return []\n"
                ),
                "brute_force_solution": (
                    "def find_product_pair_brute(nums: List[int], target: int) -> List[int]:\n"
                    "    for i in range(len(nums)):\n"
                    "        for j in range(i + 1, len(nums)):\n"
                    "            if nums[i] * nums[j] == target:\n"
                    "                return [i, j]\n"
                    "    return []\n"
                ),
                "complexity": ComplexityAnalysis(
                    time_complexity="O(N)",
                    space_complexity="O(N)",
                    explanation="A single pass hash map lookup achieves O(1) average lookup per item across N elements."
                ),
                "hints": [
                    "Consider using a Hash Map to store previously visited numbers and their indices.",
                    "For each number, check if target is divisible by num. If so, lookup target // num in the map.",
                    "Be careful with zero: target % 0 is a ZeroDivisionError, handle 0 explicitly!"
                ],
                "tags": ["arrays", "hash-table", "two-sum-variant", "math"]
            },
            "Two Pointers_Medium": {
                "id": "container-with-water-traps",
                "title": "Maximum Water Reservoir Trapping",
                "difficulty": DifficultyLevel.MEDIUM,
                "topic": ProblemTopic.TWO_POINTERS,
                "language": language,
                "company_style": company_style,
                "description": (
                    "You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that "
                    "the two endpoints of the `i-th` line are `(i, 0)` and `(i, height[i])`.\n\n"
                    "Find two lines that together with the x-axis form a container, such that the container contains the most water.\n"
                    "Return the maximum amount of water a container can store.\n\n"
                    "Notice that you may not slant the container."
                ),
                "constraints": [
                    "2 <= height.length <= 10^5",
                    "0 <= height[i] <= 10^4"
                ],
                "function_name": "max_area",
                "starter_code": (
                    "from typing import List\n\n"
                    "def max_area(height: List[int]) -> int:\n"
                    "    # Compute maximum water area between two vertical lines\n"
                    "    pass\n"
                ),
                "test_cases": [
                    {
                        "inputs": {"height": [1, 8, 6, 2, 5, 4, 8, 3, 7]},
                        "expected_output": 49,
                        "is_hidden": False,
                        "explanation": "Lines at index 1 (height 8) and index 8 (height 7) give min(8, 7) * (8 - 1) = 7 * 7 = 49."
                    },
                    {
                        "inputs": {"height": [1, 1]},
                        "expected_output": 1,
                        "is_hidden": False,
                        "explanation": "Width 1, height 1 -> 1 * 1 = 1."
                    },
                    {
                        "inputs": {"height": [4, 3, 2, 1, 4]},
                        "expected_output": 16,
                        "is_hidden": True,
                        "explanation": "Equal ends: min(4, 4) * (4 - 0) = 16."
                    },
                    {
                        "inputs": {"height": [1, 2, 1]},
                        "expected_output": 2,
                        "is_hidden": True,
                        "explanation": "min(1, 1) * 2 = 2 or min(2, 1) * 1 = 1 -> max is 2."
                    },
                    {
                        "inputs": {"height": [0, 0, 0, 0]},
                        "expected_output": 0,
                        "is_hidden": True,
                        "explanation": "All heights 0 yield 0 capacity."
                    }
                ],
                "optimal_solution": (
                    "from typing import List\n\n"
                    "def max_area(height: List[int]) -> int:\n"
                    "    left, right = 0, len(height) - 1\n"
                    "    max_water = 0\n"
                    "    while left < right:\n"
                    "        width = right - left\n"
                    "        h = min(height[left], height[right])\n"
                    "        max_water = max(max_water, width * h)\n"
                    "        if height[left] < height[right]:\n"
                    "            left += 1\n"
                    "        else:\n"
                    "            right -= 1\n"
                    "    return max_water\n"
                ),
                "brute_force_solution": (
                    "def max_area_brute(height: List[int]) -> int:\n"
                    "    ans = 0\n"
                    "    for i in range(len(height)):\n"
                    "        for j in range(i + 1, len(height)):\n"
                    "            ans = max(ans, min(height[i], height[j]) * (j - i))\n"
                    "    return ans\n"
                ),
                "complexity": ComplexityAnalysis(
                    time_complexity="O(N)",
                    space_complexity="O(1)",
                    explanation="Two pointers meet in the middle in N steps, requiring constant auxiliary memory."
                ),
                "hints": [
                    "The water area is determined by width * min(height[left], height[right]).",
                    "To maximize area, start with the maximum possible width: pointers at the left and right extremities.",
                    "Always advance the pointer pointing to the shorter vertical bar, since keeping the shorter line cannot yield a larger area as width shrinks."
                ],
                "tags": ["two-pointers", "greedy", "array", "faang"]
            },
            "Dynamic Programming_Medium": {
                "id": "coin-change-ways",
                "title": "Total Ways to Disburse Change",
                "difficulty": DifficultyLevel.MEDIUM,
                "topic": ProblemTopic.DYNAMIC_PROGRAMMING,
                "language": language,
                "company_style": company_style,
                "description": (
                    "You are given an integer array `coins` representing coins of different denominations and an integer `amount` "
                    "representing a total amount of money.\n\n"
                    "Return the number of distinct combinations that make up that amount. If that amount of money cannot be made up "
                    "by any combination of the coins, return `0`.\n\n"
                    "You may assume that you have an infinite number of each kind of coin."
                ),
                "constraints": [
                    "1 <= amount <= 5000",
                    "1 <= coins.length <= 300",
                    "1 <= coins[i] <= 5000",
                    "All values of coins are unique."
                ],
                "function_name": "change_ways",
                "starter_code": (
                    "from typing import List\n\n"
                    "def change_ways(amount: int, coins: List[int]) -> int:\n"
                    "    # Return total distinct coin combinations for amount\n"
                    "    pass\n"
                ),
                "test_cases": [
                    {
                        "inputs": {"amount": 5, "coins": [1, 2, 5]},
                        "expected_output": 4,
                        "is_hidden": False,
                        "explanation": "4 ways: 5, 2+2+1, 2+1+1+1, 1+1+1+1+1."
                    },
                    {
                        "inputs": {"amount": 3, "coins": [2]},
                        "expected_output": 0,
                        "is_hidden": False,
                        "explanation": "Amount 3 cannot be made with only 2-cent coins."
                    },
                    {
                        "inputs": {"amount": 10, "coins": [10]},
                        "expected_output": 1,
                        "is_hidden": True,
                        "explanation": "Exactly one 10-cent coin."
                    },
                    {
                        "inputs": {"amount": 0, "coins": [1, 2, 5]},
                        "expected_output": 1,
                        "is_hidden": True,
                        "explanation": "Amount 0 has exactly 1 way: pick no coins."
                    },
                    {
                        "inputs": {"amount": 8, "coins": [2, 3, 5]},
                        "expected_output": 3,
                        "is_hidden": True,
                        "explanation": "Ways: 5+3, 3+3+2, 2+2+2+2."
                    }
                ],
                "optimal_solution": (
                    "from typing import List\n\n"
                    "def change_ways(amount: int, coins: List[int]) -> int:\n"
                    "    dp = [0] * (amount + 1)\n"
                    "    dp[0] = 1\n"
                    "    for coin in coins:\n"
                    "        for x in range(coin, amount + 1):\n"
                    "            dp[x] += dp[x - coin]\n"
                    "    return dp[amount]\n"
                ),
                "brute_force_solution": "Recursive knapsack exploration branching over inclusion/exclusion of each coin: O(2^N) exponential time.",
                "complexity": ComplexityAnalysis(
                    time_complexity="O(N * amount) where N is the number of coin denominations",
                    space_complexity="O(amount) space for the 1D DP table",
                    explanation="Outer loop iterates through each coin, inner loop updates dp from coin to amount in linear time."
                ),
                "hints": [
                    "This is the Unbounded Knapsack problem counting combinations.",
                    "Why must the outer loop iterate over coins rather than amounts? If you loop over amounts outer, you count permutations instead of combinations!",
                    "Initialize dp[0] = 1 because there is exactly 1 way to make change for 0 (the empty set)."
                ],
                "tags": ["dynamic-programming", "knapsack", "math", "amazon"]
            }
        }

        # Select match from catalog or generate dynamic standard problem
        key = f"{topic.value}_{difficulty.value}"
        if key in catalog:
            item = catalog[key]
            return CodingProblem(**item)
        
        # Fallback to standard dynamic problem structure
        default_item = catalog["Arrays & Hashing_Easy"].copy()
        default_item["topic"] = topic
        default_item["difficulty"] = difficulty
        default_item["title"] = f"Optimized {topic.value} Analysis"
        default_item["id"] = f"{topic.value.lower().replace(' ', '-').replace('&', 'and')}-{difficulty.value.lower()}"
        return CodingProblem(**default_item)
