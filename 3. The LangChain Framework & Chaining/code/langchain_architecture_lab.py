"""
LangChain Framework Architecture Lab: Wrappers, LCEL Chains, and ReAct Agents
=============================================================================
Zero to Hero Gen AI Course — Module 03: The LangChain Framework & Chaining

This standalone educational lab demonstrates the foundational architectural
mechanics of the LangChain ecosystem:
1. Standardized Model Wrapper Protocol (.invoke, .stream, .batch)
2. LangChain Expression Language (LCEL) Pipe Operator (|) from scratch
3. Branching & Parallel Execution (RunnableParallel, RunnablePassthrough)
4. Structured Output Extraction & Validation
5. Autonomous ReAct Agent Reasoning Loop (Thought -> Action -> Observation)

Usage:
    python langchain_architecture_lab.py
"""

import sys
import os
import json
import time
import math
from typing import Any, Callable, Dict, List, Optional, Generator

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# SECTION 1: THE RUNNABLE PROTOCOL & MODEL WRAPPER
# =====================================================================

class Runnable:
    """
    Base class implementing the LangChain Runnable protocol.
    Overloads the pipe operator (|) to construct RunnableSequences.
    """
    def invoke(self, input_data: Any) -> Any:
        raise NotImplementedError("Subclasses must implement .invoke()")

    def stream(self, input_data: Any) -> Generator[Any, None, None]:
        # Default fallback generator yielding full invocation result
        yield self.invoke(input_data)

    def batch(self, inputs: List[Any]) -> List[Any]:
        # Default batch execution
        return [self.invoke(item) for item in inputs]

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        if isinstance(other, Runnable):
            return RunnableSequence(self, other)
        elif callable(other):
            return RunnableSequence(self, RunnableLambda(other))
        raise TypeError(f"Cannot pipe {type(self).__name__} with {type(other).__name__}")


class RunnableSequence(Runnable):
    """
    Represents a sequential pipeline of Runnables: (f | g | h)(x)
    """
    def __init__(self, first: Runnable, second: Runnable):
        self.steps: List[Runnable] = []
        # Flatten nested sequences
        if isinstance(first, RunnableSequence):
            self.steps.extend(first.steps)
        else:
            self.steps.append(first)
        
        if isinstance(second, RunnableSequence):
            self.steps.extend(second.steps)
        else:
            self.steps.append(second)

    def invoke(self, input_data: Any) -> Any:
        current = input_data
        for step in self.steps:
            current = step.invoke(current)
        return current

    def stream(self, input_data: Any) -> Generator[Any, None, None]:
        # Stream the last step while running prior steps synchronously
        current = input_data
        for step in self.steps[:-1]:
            current = step.invoke(current)
        yield from self.steps[-1].stream(current)


class RunnableLambda(Runnable):
    """Wraps any standard Python function or lambda into a Runnable."""
    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)


class RunnableParallel(Runnable):
    """
    Executes multiple runnables concurrently on identical input,
    returning a dictionary of results.
    """
    def __init__(self, steps_dict: Dict[str, Runnable]):
        self.steps = {
            k: (v if isinstance(v, Runnable) else RunnableLambda(v))
            for k, v in steps_dict.items()
        }

    def invoke(self, input_data: Any) -> Dict[str, Any]:
        return {key: runner.invoke(input_data) for key, runner in self.steps.items()}


# =====================================================================
# SECTION 2: PROMPT TEMPLATES & OUTPUT PARSERS
# =====================================================================

class PromptTemplate(Runnable):
    """Formats parameterized string templates into prompt text."""
    def __init__(self, template: str):
        self.template = template

    def invoke(self, input_data: Dict[str, Any]) -> str:
        if not isinstance(input_data, dict):
            raise ValueError(f"PromptTemplate requires a dict input, got {type(input_data).__name__}")
        return self.template.format(**input_data)


class StrOutputParser(Runnable):
    """Extracts raw text strings from model completions."""
    def invoke(self, input_data: Any) -> str:
        if hasattr(input_data, "content"):
            return str(input_data.content)
        if isinstance(input_data, dict) and "content" in input_data:
            return str(input_data["content"])
        return str(input_data)


class JsonOutputParser(Runnable):
    """Extracts and validates structured JSON dictionaries from model responses."""
    def invoke(self, input_data: Any) -> Dict[str, Any]:
        text = StrOutputParser().invoke(input_data).strip()
        # Strip markdown fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse model output as JSON: {err}\nOutput was: {text}")


# =====================================================================
# SECTION 3: MODEL WRAPPER IMPLEMENTATION (STANDALONE + LIVE)
# =====================================================================

class AIMessage:
    """Represents a standardized model completion message."""
    def __init__(self, content: str, model_name: str = "mock-gpt-4o"):
        self.content = content
        self.model_name = model_name

    def __repr__(self):
        return f"AIMessage(content={self.content[:60]!r}...)"


class ChatModelWrapper(Runnable):
    """
    Standardized Chat Model Wrapper conforming to LangChain's BaseChatModel interface.
    Operates seamlessly in Mock Mode or Live OpenAI API mode.
    """
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.live_client = None

        if self.api_key and not self.api_key.startswith("sk-mock"):
            try:
                from openai import OpenAI
                self.live_client = OpenAI(api_key=self.api_key)
            except Exception:
                self.live_client = None

    def invoke(self, input_prompt: str) -> AIMessage:
        if self.live_client:
            resp = self.live_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": str(input_prompt)}]
            )
            return AIMessage(content=resp.choices[0].message.content, model_name=self.model)

        # Standalone Intelligent Mock LLM Simulation
        simulated_text = self._simulate_inference(str(input_prompt))
        return AIMessage(content=simulated_text, model_name=f"mock-{self.model}")

    def stream(self, input_prompt: str) -> Generator[AIMessage, None, None]:
        full_message = self.invoke(input_prompt)
        words = full_message.content.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            time.sleep(0.02)  # Simulate network latency chunk arrival
            yield AIMessage(content=chunk, model_name=self.model)

    def _simulate_inference(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "incident" in prompt_lower or "json" in prompt_lower:
            return json.dumps({
                "service": "auth-gateway",
                "severity": "CRITICAL",
                "downtime_minutes": 14,
                "root_cause": "TLS certificate expired at 04:00 UTC",
                "recommended_action": "Renew SSL cert and rotate cluster secrets"
            }, indent=2)
        elif "translate" in prompt_lower:
            return "La base de données distribuée a atteint le consensus sur toutes les répliques."
        elif "haiku" in prompt_lower:
            return "Silent clusters hum,\nPackets dance across the wire,\nData finds its home."
        elif "summarize" in prompt_lower:
            return "Kubernetes orchestrates containerized workloads, automating deployment, scaling, and operational management."
        else:
            return f"[Simulated Response ({self.model}) for prompt length {len(prompt)}]: Completed analysis successfully."


# =====================================================================
# SECTION 4: THE REACT AGENT REASONING ENGINE
# =====================================================================

class Tool:
    """Represents an executable action capability exposed to the agent."""
    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func

    def run(self, tool_input: str) -> str:
        try:
            return str(self.func(tool_input))
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"


class ReActAgent:
    """
    Autonomous Reasoning and Acting (ReAct) Engine.
    Executes the Thought -> Action -> Observation cycle until task resolution.
    """
    def __init__(self, model: ChatModelWrapper, tools: List[Tool], max_iterations: int = 5):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations

    def run(self, user_objective: str) -> str:
        print(f"\n🎯 [Agent Goal]: {user_objective}")
        print("=" * 65)

        # Context trace of the reasoning loop
        history: List[str] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 --- Iteration {iteration} ---")

            # In production: The LLM generates the Thought and Action.
            # In our educational lab, we simulate the internal reasoning steps:
            thought, action, action_input = self._agent_reason(user_objective, history, iteration)

            print(f"💭 [Thought]: {thought}")
            
            if action == "Final Answer":
                print(f"🏁 [Final Answer]: {action_input}")
                print("=" * 65)
                return action_input

            print(f"🛠️  [Action]: Invoking tool '{action}' with input '{action_input}'")
            tool_obj = self.tools.get(action)
            if not tool_obj:
                observation = f"Error: Tool '{action}' does not exist. Available tools: {list(self.tools.keys())}"
            else:
                observation = tool_obj.run(action_input)

            print(f"👁️  [Observation]: {observation}")
            history.append(f"Thought: {thought}\nAction: {action}[{action_input}]\nObservation: {observation}")

        return "Agent reached max iterations without finding a definitive answer."

    def _agent_reason(self, goal: str, history: List[str], step: int):
        goal_lower = goal.lower()
        if "nvidia" in goal_lower or "nvda" in goal_lower:
            if step == 1:
                return (
                    "I need to check Nvidia's current market stock price first.",
                    "StockPriceLookup",
                    "NVDA"
                )
            elif step == 2:
                return (
                    "Nvidia's stock is $144.00. Now I must calculate its square root.",
                    "Calculator",
                    "sqrt(144.00)"
                )
            else:
                return (
                    "I have obtained the square root of Nvidia's price. I can provide the final answer.",
                    "Final Answer",
                    "The current price of Nvidia (NVDA) is $144.00, and its square root is exactly 12.00."
                )
        elif "cluster" in goal_lower or "server" in goal_lower:
            if step == 1:
                return (
                    "I need to inspect the operational metrics of the target Kubernetes cluster.",
                    "ClusterDiagnostics",
                    "prod-us-east-1"
                )
            else:
                return (
                    "Cluster diagnostics show 1 failed pod and 94% memory utilization.",
                    "Final Answer",
                    "Cluster 'prod-us-east-1' requires immediate remediation: 1 pod failed and memory is critical at 94%."
                )
        else:
            return (
                "Objective is straightforward; returning direct answer.",
                "Final Answer",
                f"Task resolved: {goal}"
            )


# =====================================================================
# LAB EXPERIMENTS & DEMONSTRATION SUITE
# =====================================================================

def banner(title: str):
    print("\n" + "#" * 70)
    print(f"##  {title}")
    print("#" * 70)


def experiment_1_model_wrapper():
    banner("EXPERIMENT 1: Standardized Model Wrapper Protocol")
    llm = ChatModelWrapper(model="gpt-4o", temperature=0.2)

    print("\n1. Synchronous Single Invocation (.invoke):")
    res = llm.invoke("Explain quantum entanglement in one sentence.")
    print(f"   Model Output: {res.content}")

    print("\n2. Real-Time Token Streaming (.stream):")
    print("   Streaming Chunks: ", end="")
    for chunk in llm.stream("Write a haiku about distributed systems."):
        print(chunk.content, end="", flush=True)
    print()

    print("\n3. Batch Processing (.batch):")
    prompts = [
        "Translate 'Hello world' into Spanish.",
        "Translate 'Good morning' into German.",
        "Translate 'Thank you' into Japanese."
    ]
    results = llm.batch(prompts)
    for p, r in zip(prompts, results):
        print(f"   Prompt: {p:<40} -> Output: {r.content}")


def experiment_2_lcel_pipe_syntax():
    banner("EXPERIMENT 2: LangChain Expression Language (LCEL) Unix Pipe (|)")

    prompt = PromptTemplate("Translate the following sentence into {language}: '{text}'")
    model = ChatModelWrapper(model="gpt-4o", temperature=0.0)
    parser = StrOutputParser()

    # Constructing the LCEL Pipeline using Python's overloaded pipe operator:
    # prompt | model | parser
    translation_pipeline = prompt | model | parser

    print("\nPipeline Architecture:")
    print(f"   Structure: {type(translation_pipeline).__name__}")
    print(f"   Steps: {[type(s).__name__ for s in translation_pipeline.steps]}")

    payload = {
        "language": "French",
        "text": "The distributed database reached consensus across all replicas."
    }
    print(f"\nExecuting Pipeline with Input: {payload}")
    output = translation_pipeline.invoke(payload)
    print(f"Pipeline Result: {output}")


def experiment_3_parallel_and_passthrough():
    banner("EXPERIMENT 3: Multi-Branch Composition (RunnableParallel)")

    # Simulates a RAG parallel retrieval branch
    parallel_branch = RunnableParallel({
        "original_topic": lambda x: x["topic"],
        "context_docs": lambda x: f"[Retrieved RAG Chunks for '{x['topic']}': Architecture overview, API docs, Benchmarks]",
        "timestamp": lambda _: time.strftime("%Y-%m-%d %H:%M:%S")
    })

    print("Input: {'topic': 'Vector Databases'}")
    result = parallel_branch.invoke({"topic": "Vector Databases"})
    print("Parallel Branch Output:")
    for k, v in result.items():
        print(f"   - {k}: {v}")


def experiment_4_structured_json_parsing():
    banner("EXPERIMENT 4: Structured Output Parsing (JsonOutputParser)")

    incident_prompt = PromptTemplate(
        "Analyze this server alert and format as a valid JSON object:\nAlert: {alert_text}"
    )
    model = ChatModelWrapper(model="gpt-4o", temperature=0.0)
    json_parser = JsonOutputParser()

    incident_chain = incident_prompt | model | json_parser

    alert = "ALERT: TLS certificate for auth-gateway expired at 04:00 UTC causing 14 minutes of customer downtime."
    print(f"Input Alert Text:\n   '{alert}'\n")

    structured_data = incident_chain.invoke({"alert_text": alert})
    print("Parsed & Validated Python Dictionary:")
    print(json.dumps(structured_data, indent=2))
    print(f"Type check: service='{structured_data['service']}' (Severity: {structured_data['severity']})")


def experiment_5_react_agent_reasoning_loop():
    banner("EXPERIMENT 5: Autonomous ReAct Agent Reasoning Loop")

    # Defining Tools
    def stock_lookup(ticker: str) -> str:
        stocks = {"NVDA": 144.00, "AAPL": 225.50, "MSFT": 448.20, "GOOGL": 180.10}
        price = stocks.get(ticker.upper())
        return f"${price:.2f}" if price else f"Ticker {ticker} not found."

    def math_calculator(expression: str) -> str:
        if "sqrt" in expression:
            val = float(expression.replace("sqrt(", "").replace(")", "").strip())
            return f"{math.sqrt(val):.2f}"
        return "Unknown math operation."

    def cluster_diagnostics(cluster_name: str) -> str:
        return f"Metrics for {cluster_name}: Status=DEGRADED, Memory=94%, UnhealthyPods=1."

    tools = [
        Tool("StockPriceLookup", "Looks up the current price of a stock ticker", stock_lookup),
        Tool("Calculator", "Performs mathematical computations like sqrt", math_calculator),
        Tool("ClusterDiagnostics", "Inspects Kubernetes infrastructure health", cluster_diagnostics),
    ]

    llm = ChatModelWrapper(model="gpt-4o")
    agent = ReActAgent(model=llm, tools=tools, max_iterations=4)

    # Multi-Step Goal requiring dynamic tool composition
    goal = "What is the square root of Nvidia's (NVDA) current stock price?"
    agent.run(goal)


def main():
    print("""
========================================================================
   LANGCHAIN FRAMEWORK ARCHITECTURE: WRAPPERS, CHAINS & AGENTS LAB
========================================================================
    """)
    experiment_1_model_wrapper()
    experiment_2_lcel_pipe_syntax()
    experiment_3_parallel_and_passthrough()
    experiment_4_structured_json_parsing()
    experiment_5_react_agent_reasoning_loop()
    print("\n✅ All 5 LangChain Architecture experiments completed successfully!\n")


if __name__ == "__main__":
    main()
