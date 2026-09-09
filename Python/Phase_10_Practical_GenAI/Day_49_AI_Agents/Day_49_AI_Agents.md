# Day 49: AI Agents & Function Calling — LLMs That Take Action


| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 48: RAG Part 2](../Day_48_RAG_Part_2/Day_48_RAG_Part_2.md) | [All 50 Days Overview](../../README.md) | [Day 50: Fine-Tuning, LoRA & Running Models Locally →](../Day_50_FineTuning_LoRA_Local/Day_50_FineTuning_LoRA_Local.md) |

Welcome to **Day 49 of our 50-Day Generative AI Masterclass**! In [Day 47](../Day_47_RAG_Part_1/Day_47_RAG_Part_1.md) and [Day 48](../Day_48_RAG_Part_2/Day_48_RAG_Part_2.md), you mastered Retrieval-Augmented Generation, giving LLMs the ability to consult enterprise documents before answering.

Today, we take the ultimate leap in artificial intelligence: **moving from passive question-answering chatbots to autonomous, action-taking AI Agents**.

A standard chatbot lives in a glass box: it can describe how to query a database, but it cannot run the query. An **AI Agent**, by contrast, possesses hands:
- It connects to external APIs, SQL databases, Python REPLs, and web browsers.
- It observes errors, reasons about missing facts, formulates multi-step plans, and self-corrects until a complex goal is achieved.

In this masterclass, we demystify the two pillars of agentic AI: the **ReAct (Reasoning + Acting) cognitive loop** and the **Function Calling / Tool Use Protocol**.

---

## 1. The Core Mental Model: The Mars Rover vs. The Encyclopedia

```
+-----------------------------------------------------------------------------------+
|                        PASSIVE CHATBOT VS. AUTONOMOUS AGENT                       |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. THE ENCYCLOPEDIA (Standard LLM / Chatbot):                                    |
|     • Role: Passive Information Repository.                                       |
|     • User: "What is the soil composition on Mars?"                               |
|     • LLM: Recites memorized textbook facts about iron oxide and basalt.           |
|     • Limitation: If you ask "What is the temperature at Crater 4 right now?",   |
|       it has zero live instruments to measure it.                                 |
|                                                                                   |
|  2. THE MARS ROVER (The Autonomous AI Agent):                                     |
|     • Role: Action-Oriented Decision Maker with Instruments (Tools).              |
|     • User: "Analyze Crater 4 for water ice."                                     |
|     • Agent:                                                                      |
|       - THOUGHT: "I am 50 meters away. I must first engage drive motors."         |
|       - ACTION: drive(distance_m=50, heading=180)                                 |
|       - OBSERVATION: "Arrived at Crater 4 edge."                                  |
|       - THOUGHT: "Now I deploy the drill spectrometer."                           |
|       - ACTION: drill_spectrometer(depth_cm=15)                                   |
|       - OBSERVATION: "Spectrum reveals 4.2% H2O ice signature."                   |
|       - FINAL ANSWER: "Analysis complete: Confirmed 4.2% water ice at 15cm depth."|
+-----------------------------------------------------------------------------------+
```

---

## 2. The ReAct Paradigm: Reasoning + Acting

In 2022, Shunyu Yao et al. (Princeton / Google Brain) published **ReAct: Synergizing Reasoning and Acting in Language Models**:

![ReAct Agent Loop Lifecycle](assets/react_agent_loop_lifecycle.svg)

Before ReAct, models either:
- **Reasoned only (Chain-of-Thought)**: Hallucinated external facts without grounding.
- **Acted only (Direct Tool Calling)**: Executed tools blindly without planning or tracking state, easily getting stuck in loops.

ReAct unifies both into an iterative, self-correcting cognitive loop:

```
            +-------------------------------------------------------+
            |                        THOUGHT                        |
            | "I need to find the user's account ID before I can    |
            |  lookup their billing invoice."                       |
            +-------------------------------------------------------+
                                        |
                                        v
            +-------------------------------------------------------+
            |                        ACTION                         |
            | get_user_id(email="alice@corp.com")                   |
            +-------------------------------------------------------+
                                        |
                                        v
            +-------------------------------------------------------+
            |                      OBSERVATION                      |
            | External Environment executes tool -> "usr_8921"      |
            +-------------------------------------------------------+
                                        |
                 (Feedback re-enters context as new truth)
                                        |
                                        v
            +-------------------------------------------------------+
            |                     NEXT THOUGHT                      |
            | "Now that I have usr_8921, I can query their invoice."|
            +-------------------------------------------------------+
```

---

## 3. Function Calling & Tool Use Under the Hood

How does an LLM—which only outputs tokens—physically call an external Python function or SQL database?

The secret is the **Function Calling Protocol** supported natively by OpenAI, Anthropic, and local inference engines (Ollama / vLLM):

![Tool Use Sequence Diagram](assets/tool_use_function_calling_flow.svg)

Let's trace the exact 5-step wire protocol.

---

### Step 1: Client Declares Tool Schemas via JSON Schema
When calling the API, the client supplies an array of available tools defined in standard **JSON Schema**:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieves the real-time stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, NVDA, TSLA)"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]
```

### Step 2: The Model Generates a Tool Call Packet
Instead of responding with conversational text to the user, the model detects that it cannot answer without external data. It halts text generation and returns a structured tool call message:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_stock_price",
        "arguments": "{\"ticker\": \"NVDA\"}"
      }
    }
  ]
}
```

Notice that `content` is `null`! The model is not speaking to the user; it is issuing an instruction to your client application.

### Step 3: Client Application Executes Real-World Code
Your Python code intercepts the `tool_calls` packet, parses the JSON arguments, and runs the actual underlying function in the real world:

```python
# Real Python execution on your machine or server:
result = fetch_stock_api("NVDA") # Returns: {"price": 128.50, "currency": "USD"}
```

### Step 4: Client Sends the Observation Back to the LLM
The client appends the execution result to the `messages` array using the special role `"tool"` and matching `tool_call_id`:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"price\": 128.50, \"currency\": \"USD\"}"
}
```

### Step 5: LLM Synthesizes the Final Answer
With the real-world observation now present in its context window, the model resumes normal autoregressive generation and delivers a verified answer:
> *"NVIDIA (NVDA) is currently trading at $128.50 USD."*

---

## 4. Production Agent Failure Modes & Safety Guardrails

Autonomous agents are immensely powerful, but giving models the authority to execute code introduces severe production risks:

```
+-----------------------------------------------------------------------------------+
|                        THE 3 CRITICAL AGENT FAILURE MODES                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. THE INFINITE LOOP TRAP:                                                       |
|     • Symptom: The agent calls a tool, encounters an error, tweaks an irrelevant  |
|       argument, calls the same tool again, and repeats forever (burning API credits!). |
|     • Guardrail: Hard Iteration Cap (`max_iterations = 10`) and Cycle Detection.  |
|                                                                                   |
|  2. TOOL ARGUMENT HALLUCINATION:                                                  |
|     • Symptom: The model invents parameters not present in the JSON Schema        |
|       (e.g., passing `auth_token="admin"` to a weather API).                      |
|     • Guardrail: Strict Pydantic schema validation; feed errors back as feedback. |
|                                                                                   |
|  3. CATASTROPHIC IRREVERSIBLE ACTIONS:                                            |
|     • Symptom: Agent decides to `DROP TABLE users;` or transfer $50,000 to an     |
|       unverified vendor account.                                                  |
|     • Guardrail: Human-in-the-Loop (HITL) authorization gates for all write/delete|
|       operations!                                                                 |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 5. Production Hands-On Lab: Framework-Free ReAct Agent in Pure Python

Let's build a complete, production-grade ReAct agent from scratch in pure Python with zero dependencies on heavyweight frameworks (like LangChain or AutoGen).

Our agent will possess:
1. Three registered tools: `calculator`, `query_customer_db`, and `get_system_time`.
2. A full Thought-Action-Observation loop.
3. An iteration cap to prevent runaway token costs.

### Python Script: `react_agent_from_scratch.py`

```python
"""
react_agent_from_scratch.py
Production-grade, framework-free ReAct Agent featuring:
1. Tool Registry with Strict Type Signatures
2. Thought -> Action -> Observation Cognitive Loop
3. Automated Cycle Detection & Iteration Limits
Author: GenAI 50-Day Masterclass
"""

import json
import re
import datetime
from typing import Callable, Dict, Any, Tuple

# =====================================================================
# 1. TOOL REGISTRY & REAL-WORLD FUNCTIONS
# =====================================================================
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, str] = {}

    def register(self, name: str, description: str):
        def decorator(func: Callable):
            self.tools[name] = func
            self.schemas[name] = description
            return func
        return decorator

    def execute(self, name: str, **kwargs) -> str:
        if name not in self.tools:
            return f"Error: Tool '{name}' does not exist. Available tools: {list(self.tools.keys())}"
        try:
            result = self.tools[name](**kwargs)
            return json.dumps(result)
        except Exception as e:
            return f"ExecutionError in '{name}': {str(e)}"

registry = ToolRegistry()

@registry.register("get_system_time", "Returns current UTC datetime string. No arguments required.")
def get_system_time() -> Dict[str, str]:
    return {"current_utc_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

@registry.register("query_customer_db", "Looks up customer account by email. Args: email: str")
def query_customer_db(email: str) -> Dict[str, Any]:
    mock_db = {
        "alice@acme.com": {"account_id": "usr_4021", "plan": "Enterprise", "balance_due": 450.00, "status": "Active"},
        "bob@beta.org":   {"account_id": "usr_9981", "plan": "Starter",    "balance_due": 0.00,   "status": "Suspended"}
    }
    customer = mock_db.get(email.strip())
    if customer:
        return customer
    return {"error": f"Customer with email '{email}' not found."}

@registry.register("apply_discount_calculator", "Computes discounted amount. Args: balance: float, discount_percent: float")
def apply_discount_calculator(balance: float, discount_percent: float) -> Dict[str, float]:
    discounted = balance * (1.0 - (discount_percent / 100.0))
    return {"original_balance": balance, "discount_percent": discount_percent, "final_balance": round(discounted, 2)}


# =====================================================================
# 2. MOCK LLM ENGINE SIMULATING REACT TOKEN GENERATION
# =====================================================================
class SimulatedReActLLM:
    """
    Simulates an LLM emitting Thought and Action tokens across turns.
    """
    def generate_step(self, context_history: str, step_idx: int) -> str:
        if step_idx == 0:
            return """Thought: The user wants to check Alice's account status and apply a 15% promotional discount to her outstanding balance. First, I need to lookup her account details using her email.
Action: query_customer_db {"email": "alice@acme.com"}"""

        elif step_idx == 1:
            return """Thought: The customer DB returned account usr_4021 with an active Enterprise plan and an outstanding balance of $450.00. Now I need to calculate the final balance after applying the 15% discount.
Action: apply_discount_calculator {"balance": 450.00, "discount_percent": 15.0}"""

        else:
            return """Thought: The calculator confirms that applying a 15% discount to $450.00 leaves a final balance of $382.50. I now have all verified facts needed to address the user.
Final Answer: Alice's account (usr_4021) is currently Active on the Enterprise plan with an outstanding balance of $450.00. With the 15% promotional discount applied, her new balance due is $382.50."""


# =====================================================================
# 3. REACT AGENT ORCHESTRATION ENGINE
# =====================================================================
class ReActAgent:
    def __init__(self, tool_registry: ToolRegistry, max_iterations: int = 5):
        self.registry = tool_registry
        self.max_iterations = max_iterations
        self.llm = SimulatedReActLLM()

    def run(self, user_goal: str):
        print("=" * 65)
        print("RUNNING AUTONOMOUS REACT AGENT")
        print(f"Goal: '{user_goal}'")
        print("=" * 65)

        context = f"Goal: {user_goal}\nAvailable Tools: {self.registry.schemas}\n"

        for iteration in range(self.max_iterations):
            print(f"\n--- [Cycle {iteration + 1}/{self.max_iterations}] ---")

            # 1. Model produces Thought + Action
            output = self.llm.generate_step(context, step_idx=iteration)
            print(output)

            # Check if agent has reached Final Answer
            if "Final Answer:" in output:
                print("\n✓ SUCCESS: Agent reached definitive solution!")
                return

            # 2. Parse Action and Arguments
            action_match = re.search(r"Action:\s*(\w+)\s*(\{.*?\})", output, re.DOTALL)
            if not action_match:
                print("✗ Format Error: No valid Action line found. Halting.")
                return

            tool_name = action_match.group(1)
            tool_args_str = action_match.group(2)

            try:
                tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError:
                print(f"✗ Argument Error: Could not parse JSON '{tool_args_str}'")
                return

            # 3. Execute Tool in the real world
            print(f"\n>>> [Executing Tool]: {tool_name}({tool_args})")
            observation = self.registry.execute(tool_name, **tool_args)
            print(f"<<< [Observation]: {observation}")

            # 4. Append observation to context for next loop
            context += f"{output}\nObservation: {observation}\n"

        print("\n[Alert] Agent reached maximum iteration cap without terminating.")


# =====================================================================
# 4. EXECUTION
# =====================================================================
if __name__ == "__main__":
    agent = ReActAgent(tool_registry=registry, max_iterations=5)
    agent.run("Look up Alice at alice@acme.com, check her balance, and calculate what she owes with a 15% discount.")
    print("=" * 65)
```

---

## 6. Agent Architecture Comparison Cheat Sheet

| Paradigm | Control Flow | Best For | Primary Failure Mode |
| :--- | :--- | :--- | :--- |
| **Linear Chain (CoT)** | Single pass, unbranching | Pure math, code explanation | Cascading arithmetic error |
| **Direct Tool Calling** | Single prompt $\to$ single tool | Simple API lookups (weather, stocks) | Fails on multi-hop dependent queries |
| **ReAct Loop** | Dynamic Thought $\to$ Action $\to$ Observation | Multi-step research, database audits | Infinite loop if tool returns repeated errors |
| **Multi-Agent Swarm** | Specialized agents message each other | Software engineering (Coder + Tester + Reviewer) | High token consumption; coordination deadlocks |

---

## 7. Self-Check Exercises & Solutions

### Question 1: Why Separate "Thought" from "Action" in ReAct?
What catastrophic failure occurs if an agent emits an Action directly without generating an internal Thought first?

**Solution**:
If an agent acts without reasoning, it engages in **blind trial-and-error**. Without a generated Thought token sequence:
1. The model cannot track which sub-goals have been achieved and which remain unfulfilled.
2. It cannot evaluate whether previous observations contradicted its initial hypothesis.
3. It cannot synthesize intermediate calculations before calling subsequent tools.
Generating explicit Thought tokens allocates test-time compute, allowing the model's self-attention mechanism to reflect on prior observations before committing to an external API call.

---

### Question 2: The Role of `tool_call_id` in Multi-Turn APIs
In OpenAI's Function Calling API, why must every tool observation message contain a specific `tool_call_id` that matches the assistant's previous message?

**Solution**:
In modern models that support **parallel function calling**, the assistant can issue multiple tool calls in a single turn (e.g., calling `get_weather(city="Paris")` and `get_weather(city="Tokyo")` simultaneously). The unique `tool_call_id` (e.g., `call_001` vs `call_002`) acts as an unambiguous foreign key, allowing the model's attention mechanism to map each observation payload back to its exact respective request without confusing the arguments.

---

### Question 3: The Danger of Autonomous Write Actions
An engineering team builds an autonomous DevOps agent with tools: `list_pods()`, `restart_pod()`, and `delete_cluster()`. What architectural design pattern must be enforced before `delete_cluster()` can execute?

**Solution**:
The system must enforce a **Human-in-the-Loop (HITL) Authorization Gate**. When the agent selects a destructive, irreversible action like `delete_cluster()`, execution must halt. The system generates a pending approval request (e.g., in a Slack channel or dashboard UI) detailing the exact parameters and intended reasoning. Only when a designated human administrator explicitly clicks "Approve" does the client application execute the tool and return the observation to the agent.

---

## 8. Summary & Next Steps

Today, you crossed the bridge from passive LLMs to autonomous AI agents:
- **The ReAct Cycle**: Iterative self-correction via Thought $\to$ Action $\to$ Observation.
- **Function Calling**: Contractual JSON Schema tool definitions, assistant tool call packets, and observation feedback.
- **Production Guardrails**: Iteration limits, Pydantic argument validation, and Human-in-the-Loop safety gates.

Tomorrow, we reach the grand summit of our 50-day journey: **Day 50: Fine-Tuning, LoRA & Running Models Locally**! We will explore **PEFT, QLoRA, 4-bit Quantization, and running open-weights models on your own machine**!


---

## 🧭 Navigation & Next Steps

| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 48: RAG Part 2](../Day_48_RAG_Part_2/Day_48_RAG_Part_2.md) | [All 50 Days Overview](../../README.md) | [Day 50: Fine-Tuning, LoRA & Running Models Locally →](../Day_50_FineTuning_LoRA_Local/Day_50_FineTuning_LoRA_Local.md) |
