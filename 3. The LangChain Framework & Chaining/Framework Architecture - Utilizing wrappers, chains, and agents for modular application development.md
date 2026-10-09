# 01. LangChain Framework Architecture: Wrappers, LCEL Chains, and ReAct Agents

> **Zero to Hero Gen AI Course — Module 03: The LangChain Framework & Chaining**  
> ⏱️ Estimated Reading Time: 60 minutes | 🎯 Level: Intermediate to Advanced  
> ☕ **Audience:** Java / Spring Boot Developers transitioning to Python & Generative AI

---

## 0. 🌟 Why this topic matters

When developers first build Generative AI applications, they typically write ad-hoc procedural scripts directly against vendor SDKs:

```python
# ❌ THE AD-HOC PROCEDURAL TRAP:
response = openai_client.chat.completions.create(model="gpt-4o", messages=[...])
text = response.choices[0].message.content
# Manual string slicing, fragile regex parsing, nested error handlers...
```

While this works for simple scripts, enterprise applications quickly hit three critical bottlenecks:
1. **Vendor Lock-In:** Switching from OpenAI (`gpt-4o`) to Anthropic (`claude-3-5-sonnet`) or an on-premise local open-source model (via Ollama or vLLM) requires rewriting client initializations, payload structures, streaming handlers, and exception blocks across your entire codebase.
2. **Brittle Pipeline Composition:** Real-world workflows (e.g., Query Rewrite $\to$ Document Retrieval $\to$ Summarization $\to$ Guardrail Verification $\to$ Structured JSON Extraction) turn into deeply nested procedural spaghetti code with custom glue logic at every stage.
3. **No Dynamic Autonomy:** Static code follows pre-determined `if/else` branches. It cannot autonomously reason about *which* enterprise tool to invoke (e.g., SQL Database vs. ElasticSearch vs. Internal REST API) based on real-time user intent.

**LangChain** solves these challenges by introducing a unified, composable, production-ready framework architecture:

$$\text{LangChain Framework} = \underbrace{\text{Unified Model Wrappers}}_{\text{Portability}} + \underbrace{\text{LCEL Chains}}_{\text{Deterministic Composition}} + \underbrace{\text{ReAct Agents}}_{\text{Autonomous Decision-Making}}$$

---

## 1. 🐣 Basic Level – "Explain like I'm new"

### 1.1 The 3 Core Pillars in Plain English

```
+-----------------------------------------------------------------------------------+
|                        THE THREE CORE PILLARS OF LANGCHAIN                        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. MODEL WRAPPERS              2. LCEL PIPELINES              3. REACT AGENTS    |
|   (Universal Power Adapter)     (Factory Conveyor Belt)        (Autonomous Sleuth)|
|                                                                                   |
|   ┌─────────────────┐           ┌─────────────────┐          ┌──────────────────┐ |
|   │ OpenAI / Claude │           │ Input -> Step A │          │  Thought: Plan   │ |
|   │ Ollama / Cohere │  ======>  │   │             │  ======> │  Action: Use Tool│ |
|   └────────┬────────┘           │   ▼             │          │  Obs: Read result│ |
|            │                    │ Step B -> Output│          │  Repeat / Finish │ |
|   [One Uniform Interface]       └─────────────────┘          └──────────────────┘ |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

1. **Model Wrappers:** A universal client interface. You write your code once against `BaseChatModel`, and you can swap the underlying model provider (OpenAI, Anthropic, Bedrock, Ollama) by changing a single line of configuration.
2. **Chains (LCEL):** A declarative software assembly line. You connect prompt templates, AI models, and data parsers using the Unix pipe operator `|`, giving you automatic streaming, async execution, and batch processing out of the box.
3. **Agents (ReAct):** An AI decision-making loop. Instead of following a hardcoded path, the model inspects the user's question, plans what to do, calls external tools (calculators, databases, web searches), reads the results, and iterates until the goal is achieved.

---

### 1.2 Three Real-World Mental Models & Analogies

#### 🔌 Model 1: The Universal Travel Power Adapter (Model Wrappers)
Imagine traveling through Europe, the UK, the US, and Japan. Every country has distinct wall sockets, pin shapes, and voltages. Without an adapter, you must buy four different charging cables.
- **Model Wrappers** (`ChatOpenAI`, `ChatAnthropic`, `ChatOllama`) act as the universal international power adapter.
- Your downstream application code always plugs into the exact same socket: `.invoke(prompt)`.
- The wrapper translates this call behind the scenes into the vendor's proprietary JSON schema, HTTP headers, authentication, and error formats.

---

#### 🏭 Model 2: The Factory Conveyor Belt & Unix Pipes (LCEL Chains)
In an automotive assembly plant, raw steel passes along a conveyor belt: the stamping press shapes the door, the robotic arm welds it, the sprayer paints it, and the laser sensor checks quality. Workers do not carry half-built doors across the factory floor by hand.
- **LangChain Expression Language (LCEL)** operates like this conveyor belt using Unix pipe syntax (`|`):
  $$\text{Input Dictionary} \xrightarrow{\mid} \text{Prompt Template} \xrightarrow{\mid} \text{LLM Model} \xrightarrow{\mid} \text{JSON Output Parser}$$
- Each workstation transforms the data and passes it immediately to the next stage, supporting native streaming and asynchronous execution automatically.

---

#### 🕵️ Model 3: The Detective with a Toolbag (Autonomous ReAct Agents)
A factory conveyor belt is deterministic: every car is built identically. But what if a task requires investigation?
- *"Find which regional warehouse had the highest return rate last month, find the warehouse manager's contact, and draft an escalation summary."*
- A static conveyor belt cannot predict how many database queries are needed or whether a directory lookup is required.
- An **Agent** is like an autonomous detective with a toolbag (SQL client, phone directory, calculator). The detective looks at the clue, thinks (*"I need to query returns"*), picks a tool, executes it, observes the result, and iterates until the mystery is solved!

---

### ☕ 1.3 The Java & Spring Boot Developer Bridge

How does LangChain map to enterprise concepts in Java and Spring Boot?

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Java / Spring Boot Concept            │ Python / LangChain Equivalent         │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Spring AI `ChatModel` Interface       │ LangChain `BaseChatModel`             │
│ (`OpenAiChatModel`, `OllamaChatModel`)│ (`ChatOpenAI`, `ChatOllama`)          │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Java 8+ Streams API / Apache Camel    │ LangChain Expression Language (LCEL)  │
│ `.stream().map().filter().collect()`  │ `prompt | model | output_parser`      │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Jackson `ObjectMapper.readValue(...)` │ `JsonOutputParser` / `PydanticParser` │
│ Deserializes JSON into Java POJOs     │ Parses LLM text into Pydantic models  │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Spring `@Service` bean with `@Tool`   │ LangChain `@tool` decorator           │
│ Method exposed for tool execution     │ Python function exposed to ReAct agent│
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Spring Session / Redis Session Store  │ `RunnableWithMessageHistory`          │
│ Persists state across HTTP requests   │ Persists multi-turn message history   │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

#### Code Comparison: Java Spring AI vs. Python LangChain

```java
// =========================================================================
// 1. JAVA (Spring AI) - Composing a Prompt and Model Pipeline
// =========================================================================
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.PromptTemplate;
import java.util.Map;

public class TranslationService {
    private final ChatClient chatClient;

    public TranslationService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String translate(String text, String targetLang) {
        return chatClient.prompt()
            .user(u -> u.text("Translate {text} into {language}")
                        .param("text", text)
                        .param("language", targetLang))
            .call()
            .content();
    }
}
```

```python
# =========================================================================
# 2. PYTHON (Modern LangChain v0.2+ LCEL Pipeline)
# =========================================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Declare components
prompt = ChatPromptTemplate.from_template("Translate {text} into {language}")
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Compose pipeline using Unix pipe operator |
translation_chain = prompt | model | parser

# Execute synchronously or asynchronously
result = translation_chain.invoke({"text": "Hello world", "language": "Telugu"})
print(result) # Output: "నమస్కారం ప్రపంచం" (Namaskaram Prapancham)
```

---

## 2. 🧱 Building Up – Concepts added one by one

### 2.1 The 6 Core Building Blocks of LangChain

LangChain organizes application development around six foundational modules:

![LangChain Core Architecture](assets/01_langchain_core_architecture.jpg)

| Core Module | Primary Purpose | Key Classes & Interfaces | Enterprise Role |
| :--- | :--- | :--- | :--- |
| **1. Models** | Standardized interfaces to Chat and Completion models | `ChatOpenAI`, `ChatAnthropic`, `ChatOllama` | Eliminates vendor lock-in; unifies token generation and streaming across all providers. |
| **2. Prompts** | Dynamic, parameterized templates for system and human roles | `ChatPromptTemplate`, `MessagesPlaceholder` | Ensures deterministic formatting, injection prevention, and modular prompt reusability. |
| **3. Chains (LCEL)** | Composable pipelines connecting prompts, models, and transforms | `RunnableSequence`, `RunnableParallel` | Eliminates procedural glue code; provides native async, streaming, and parallel execution. |
| **4. Memory** | State retention across stateless HTTP request cycles | `ChatMessageHistory`, `RunnableWithMessageHistory` | Injects multi-turn conversational context into prompts dynamically without blowing context limits. |
| **5. Retrievers (RAG)**| Grounding models with external private documents | `VectorStoreRetriever`, `Chroma`, `Document` | Prevents hallucinations; enables domain-specific Q&A by injecting relevant passages into prompts. |
| **6. Agents & Tools** | LLM-driven decision engines that dynamically plan and invoke tools | `create_react_agent`, `AgentExecutor`, `@tool` | Automates multi-step workflows, API calls, database lookups, and external computation. |

---

### 2.2 Model Wrappers & The Standardized `Runnable` Protocol

#### Why Abstraction Matters: Vendor Agnosticism
Without an abstraction layer, switching your LLM provider entails substantial refactoring:
```
[OpenAI API]    -> Request: {"messages": [...]}       -> Response: .choices[0].message.content
[Anthropic API] -> Request: {"system": ..., "messages"} -> Response: .content[0].text
[Ollama API]    -> Request: {"prompt": ...}           -> Response: .response
```

LangChain abstracts all models behind `BaseChatModel`. Every provider wrapper outputs a uniform `AIMessage` containing:
- `content`: The generated text or multimodal output.
- `response_metadata`: Model name, token usage breakdown, and finish reason.
- `tool_calls`: Standardized schema for function/tool invocations.

#### The Unified `Runnable` Interface
In modern LangChain (v0.1+ and v0.2+), nearly every component implements the **Runnable Protocol**:

```
                   +-----------------------------------------------+
                   |            THE RUNNABLE PROTOCOL              |
                   +-----------------------------------------------+
                   |                                               |
                   |   Synchronous                 Asynchronous    |
                   |   -------------               -------------   |
                   |   .invoke(input)    <=====>   .ainvoke(input) |
                   |   .batch([inputs])  <=====>   .abatch([inputs])|
                   |   .stream(input)    <=====>   .astream(input) |
                   |                                               |
                   +-----------------------------------------------+
```

This protocol guarantees that any component can receive an input, transform it, and yield an output under four execution modes:
1. **`invoke(input)`**: Synchronous call with a single input; blocks until full output is returned.
2. **`ainvoke(input)`**: Non-blocking asynchronous call utilizing Python's `asyncio` event loop.
3. **`stream(input)`**: Synchronous generator yielding output tokens or chunks as they arrive over the wire.
4. **`astream(input)`**: Asynchronous generator yielding chunks without blocking concurrent tasks.
5. **`batch([inputs])`**: Executes multiple inputs concurrently using automatic internal thread pools.
6. **`abatch([inputs])`**: Executes multiple inputs concurrently using native `asyncio.gather`.

---

### 2.3 LangChain Expression Language (LCEL) & Pipeline Orchestration

#### The Mathematical Formulation of LCEL
In mathematics, **function composition** is defined as:

$$(g \circ f)(x) = g(f(x))$$

For an arbitrary sequence of $N$ operations:

$$(\mathcal{R}_n \circ \dots \circ \mathcal{R}_2 \circ \mathcal{R}_1)(x) = \mathcal{R}_n(\dots \mathcal{R}_2(\mathcal{R}_1(x))\dots)$$

LangChain implements this functional composition directly in Python by overloading the bitwise OR operator `__or__` across all `Runnable` objects:

$$\text{Chain} = \mathcal{R}_{\text{Prompt}} \mid \mathcal{R}_{\text{Model}} \mid \mathcal{R}_{\text{Parser}}$$

![LCEL Execution Architecture](assets/02_lcel_chain_execution.jpg)

When Python evaluates `a | b`, where `a` and `b` inherit from `Runnable`, it instantiates a `RunnableSequence(first=a, middle=[], last=b)`.

#### Essential LCEL Primitives: Passthrough, Parallel, and Lambda
To construct non-trivial, multi-branch architectures (such as RAG pipelines), LCEL provides foundational composition primitives:

```
               ┌──► [Retriever] ─────► "context" ──┐
               │                                   │
User Query ────┼                                   ├──► [Prompt] ──► [LLM]
               │                                   │
               └──► [Passthrough] ───► "question" ─┘
```

1. **`RunnablePassthrough`**: Passes the incoming input unchanged into the next stage, or appends additional keys to an input dictionary.
2. **`RunnableParallel`**: Executes multiple runnables concurrently on the same input, packaging the outputs into a dictionary.
3. **`RunnableLambda`**: Wraps any standard Python function or lambda into a `Runnable`, granting it `.invoke()`, `.stream()`, and `.batch()` capabilities automatically.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Building a multi-branch RAG pipeline:
retrieval_chain = RunnableParallel({
    "context": (lambda x: f"Verified documentation for {x['topic']}"),
    "question": (lambda x: x["topic"])
})

prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
rag_pipeline = retrieval_chain | prompt | model | StrOutputParser()
```

#### Output Parsers: Extracting Deterministic Typed Payloads
LLMs generate unstructured text strings by default. Enterprise systems require strongly typed structured data:

| Parser | Input Type | Output Type | Best Used For |
| :--- | :--- | :--- | :--- |
| `StrOutputParser` | `AIMessage` | `str` | Pure text pipelines, chatbot dialogues, summaries. |
| `JsonOutputParser` | `AIMessage` | `dict` | Generating arbitrary structured JSON payloads. |
| `PydanticOutputParser` | `AIMessage` | `BaseModel` | Production systems requiring strict schema validation and typing. |

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class IncidentReport(BaseModel):
    service_name: str = Field(description="Name of affected microservice")
    severity: str = Field(description="Severity: LOW, MEDIUM, CRITICAL")
    downtime_minutes: int = Field(description="Total observed downtime in minutes")
    root_cause: str = Field(description="Description of failure mechanism")

parser = PydanticOutputParser(pydantic_object=IncidentReport)
prompt = ChatPromptTemplate.from_template(
    "Parse the incident log into a structured report.\n{format_instructions}\nLog: {log}"
).partial(format_instructions=parser.get_format_instructions())

incident_chain = prompt | model | parser
```

---

### 2.4 Agent Architecture: The ReAct Reasoning Paradigm

#### Why Fixed Chains Fall Short
A fixed LCEL chain executes a strictly deterministic directed acyclic graph (DAG):

$$\text{Step } 1 \longrightarrow \text{Step } 2 \longrightarrow \text{Step } 3$$

If Step 2 encounters unexpected data, an edge-case failure, or requires dynamic branching based on runtime evidence, the chain halts.

**Agents** invert control: rather than a hardcoded sequence of calls, an LLM serves as a **reasoning engine** that inspects user goals, plans actions, interacts with the environment, evaluates feedback, and determines termination dynamically.

#### The ReAct Loop: Thought $\to$ Action $\to$ Action Input $\to$ Observation
Formulated by *Yao et al. (2022)*, the **ReAct** (Reasoning + Acting) paradigm combines chain-of-thought prompting with tool execution:

![ReAct Agent Reasoning Loop](assets/03_react_agent_reasoning_loop.jpg)

The loop executes cyclically:

$$\text{Observation}_{t-1} \xrightarrow{} \text{Thought}_t \xrightarrow{} \text{Action}_t(\text{Tool}, \text{Input}) \xrightarrow{} \text{Environment Execution} \xrightarrow{} \text{Observation}_t$$

```
+-----------------------------------------------------------------------------+
|                         THE REACT REASONING TRACE                           |
+-----------------------------------------------------------------------------+
|                                                                             |
| User: "What is the square root of Nvidia's current stock price?"            |
|                                                                             |
| Thought: I need to find Nvidia's current stock price first.                 |
| Action: StockPriceLookup                                                    |
| Action Input: "NVDA"                                                        |
| Observation: $128.50                                                        |
|                                                                             |
| Thought: Now I need to calculate the square root of 128.50.                 |
| Action: Calculator                                                          |
| Action Input: "sqrt(128.50)"                                                |
| Observation: 11.33579                                                       |
|                                                                             |
| Thought: I have all required information to answer the user request.        |
| Final Answer: The square root of Nvidia's current stock price ($128.50)     |
|               is approximately 11.34.                                       |
|                                                                             |
+-----------------------------------------------------------------------------+
```

#### Tool Definition with `@tool` and Parameter Binding
In LangChain, tools are defined using the `@tool` decorator with type annotations and docstrings:

```python
from langchain_core.tools import tool

@tool
def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculates compound interest given principal, annual rate (decimal), and years."""
    return round(principal * ((1 + rate) ** years), 2)

@tool
def check_server_health(cluster_id: str) -> str:
    """Checks the operational status of a Kubernetes cluster by its identifier."""
    return f"Cluster {cluster_id}: STATUS=HEALTHY, CPU_UTILIZATION=42%, PODS=18/18"
```

#### Execution Guardrails & Stopping Conditions
Unbounded agents can fall into infinite loops or burn excessive tokens. Enterprise agent executors enforce strict operational guardrails:
1. **`max_iterations`**: Caps the maximum number of thought-action cycles (e.g., 5 or 10 iterations).
2. **`max_execution_time`**: Prevents hanging requests by terminating after a timeout threshold (e.g., 30 seconds).
3. **`early_stopping_method`**: Generates a best-effort response if iteration limits are reached before finding a complete answer.
4. **Tool Error Handling**: Catches exceptions in tool logic and injects error messages back into the observation channel so the LLM can self-correct.

---

### 2.5 Memory & Conversational State Management

#### The Statelessness Problem
Foundation models are strictly stateless. If a user asks:
- **Turn 1:** *"My name is Maya, and I manage the infrastructure team."*
- **Turn 2:** *"What team do I run?"*

Without memory, the model evaluates Turn 2 in isolation and responds: *"I don't know who you are or what team you manage."*

#### Memory Topologies Comparison

```
+-------------------------------------------------------------------------------+
|                        MEMORY TOPOLOGIES COMPARISON                           |
+-------------------------------------------------------------------------------+
|                                                                               |
|  1. CONVERSATION BUFFER MEMORY:                                               |
|     Stores 100% of raw chat history.                                          |
|     [T1] -> [T2] -> [T3] -> [T4] -> [T5] ... (Warning: Context Explosion!)   |
|                                                                               |
|  2. WINDOW BUFFER MEMORY (K=2):                                               |
|     Keeps only the most recent K interactions.                                |
|     (Evicted: [T1], [T2], [T3]) ----> Keeps: [T4] -> [T5]                     |
|                                                                               |
|  3. CONVERSATION SUMMARY MEMORY:                                              |
|     LLM continuously compresses older dialogue into a compact running narrative|
|     Summary: "User is Maya, infra manager." + Active Turn: [T5]               |
|                                                                               |
+-------------------------------------------------------------------------------+
```

| Memory Strategy | How It Works | Strengths | Trade-offs |
| :--- | :--- | :--- | :--- |
| **Buffer Memory** | Appends every raw message to the prompt history. | 100% fidelity, zero data loss. | Rapidly explodes context windows and token costs. |
| **Window Memory (`K`)** | Retains only the last $K$ turns; evicts older messages. | Constant token budget, zero extra LLM calls. | Loses long-term facts established early in conversations. |
| **Summary Memory** | Uses a background LLM call to summarize history continuously. | Preserves core facts in small token space. | Incurs extra LLM token latency and cost per turn. |
| **Vector Store Memory** | Embeds turns and retrieves only semantically relevant past messages. | Scalable to thousands of historical turns. | Higher architectural complexity and vector retrieval latency. |

#### Modern State Management with `RunnableWithMessageHistory`
In modern LangChain, legacy memory classes are superseded by `RunnableWithMessageHistory`, which wraps any LCEL chain and persists message histories per session ID:

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful software architecture assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

base_chain = qa_prompt | model | StrOutputParser()

conversational_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

---

### 2.6 Complete Architecture Visualized

```mermaid
graph TD
    User([User Request / Query]) --> Router{Execution Paradigm?}
    
    %% Deterministic LCEL Pipeline
    Router -->|Known Deterministic DAG| LCEL[LCEL Pipeline: RunnableSequence]
    LCEL --> Prompt[ChatPromptTemplate]
    Prompt --> Passthrough[RunnableParallel / Passthrough]
    Passthrough --> ModelWrapper[Model Wrapper: ChatOpenAI / ChatOllama]
    ModelWrapper --> Parser[Output Parser: Pydantic / Json / Str]
    Parser --> ValidatedOutput([Structured Output])

    %% Dynamic ReAct Agent Loop
    Router -->|Dynamic Reasoning Required| Agent[ReAct Agent Engine]
    Agent --> Thought[1. Thought: Analyze Goal & State]
    Thought --> ActionCheck{Final Answer Ready?}
    
    ActionCheck -->|No| ToolSelection[2. Action: Select Tool & Format Args]
    ToolSelection --> ToolExec[3. Execute Tool: SQL, API, Python]
    ToolExec --> Observation[4. Observation: Ingest Tool Result]
    Observation --> Thought
    
    ActionCheck -->|Yes| FinalAnswer([Return Final Answer to User])

    %% Memory Subsystem
    HistoryStore[(Session History Store)] <-->|Inject & Save State| LCEL
    HistoryStore <-->|Inject & Save State| Agent
```

---

## 3. 🧪 Hands-On Lab & Practice Exercises

### 3.1 Standalone Python Lab: LCEL & ReAct Agent Simulator

You can execute the official lab script directly from your terminal:
```bash
python "3. The LangChain Framework & Chaining/code/langchain_architecture_lab.py"
```

Here is a foundational standalone implementation demonstrating how the Runnable pipe operator `|` and `RunnableSequence` are built from scratch in pure Python:

```python
"""
Hands-On Pure-Python Implementation of the Runnable Protocol & LCEL Pipe Operator
"""
from typing import Any, Callable, List

class Runnable:
    def invoke(self, input_data: Any) -> Any:
        raise NotImplementedError

    def __or__(self, other: Any) -> "RunnableSequence":
        if isinstance(other, Runnable):
            return RunnableSequence(self, other)
        elif callable(other):
            return RunnableSequence(self, RunnableLambda(other))
        raise TypeError(f"Cannot pipe {type(self)} with {type(other)}")

class RunnableSequence(Runnable):
    def __init__(self, first: Runnable, second: Runnable):
        self.steps = []
        for step in (first, second):
            if isinstance(step, RunnableSequence):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def invoke(self, input_data: Any) -> Any:
        current = input_data
        for step in self.steps:
            current = step.invoke(current)
        return current

class RunnableLambda(Runnable):
    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)

# Test execution:
step1 = RunnableLambda(lambda x: x * 2)
step2 = RunnableLambda(lambda x: f"Value is: {x}")
pipeline = step1 | step2

print(pipeline.invoke(21)) # Output: "Value is: 42"
```

---

### 3.2 Practice Exercises (Beginner to Advanced)

#### 🟢 Exercise 1 (Easy): 3-Stage LCEL Chain with Formatting & Validation
**Problem:** Construct an LCEL pipeline using `ChatPromptTemplate`, a model wrapper, and `StrOutputParser` that translates input text to German and wraps the result in an uppercase validation check using a `RunnableLambda`.

<details>
<summary><b>View Complete Solution</b></summary>

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# 1. Components
prompt = ChatPromptTemplate.from_template("Translate the following into German:\n{text}")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
parser = StrOutputParser()

# 2. Custom Validator Lambda
def format_validator(text: str) -> dict:
    return {
        "translated_text": text.strip(),
        "uppercase_preview": text.strip().upper(),
        "char_count": len(text.strip())
    }

# 3. LCEL Pipeline Composition
translation_pipeline = prompt | model | parser | RunnableLambda(format_validator)

# 4. Execution
result = translation_pipeline.invoke({"text": "Microservices communicate via gRPC."})
print(result)
```
</details>

---

#### 🟡 Exercise 2 (Intermediate): Parallel RAG Chain with `RunnableParallel`
**Problem:** Construct an LCEL chain that takes a topic string `{"topic": "Kubernetes"}` and executes two branches in parallel:
- Branch 1: Calls a simulated retriever lambda returning documentation context.
- Branch 2: Uses `RunnablePassthrough` to preserve the original topic.
Then feeds both into a prompt that answers the question.

<details>
<summary><b>View Complete Solution</b></summary>

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def mock_retriever(inputs: dict) -> str:
    topic = inputs.get("topic", "")
    return f"Retrieved KB Article: {topic} uses Raft/etcd for distributed cluster consensus."

# 1. Parallel Branching Setup
prep_chain = RunnableParallel({
    "context": RunnableLambda(mock_retriever),
    "topic": RunnablePassthrough()
})

# 2. Downstream Prompt & Model
rag_prompt = ChatPromptTemplate.from_template(
    "Reference Context: {context}\n\n"
    "Explain the architecture of {topic} based strictly on the context."
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# 3. Full Assembled Chain
full_rag_chain = prep_chain | rag_prompt | model | StrOutputParser()

# 4. Execution
output = full_rag_chain.invoke({"topic": "Kubernetes"})
print(output)
```
</details>

---

#### 🟠 Exercise 3 (Intermediate/Hard): Engineering a ReAct Tool with Pydantic Parameter Validation
**Problem:** Build an enterprise `@tool` named `query_customer_db` that takes a Pydantic schema enforcing:
- `customer_id`: Must match pattern `^CUST-\d{5}$`
- `query_type`: Must be `"INVOICE"` or `"PROFILE"`
Simulate database lookup and error handling.

<details>
<summary><b>View Complete Solution</b></summary>

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CustomerQuerySchema(BaseModel):
    customer_id: str = Field(..., pattern=r"^CUST-\d{5}$", description="Customer ID matching CUST-XXXXX")
    query_type: str = Field(..., pattern=r"^(INVOICE|PROFILE)$", description="Type of record to lookup")

@tool(args_schema=CustomerQuerySchema)
def query_customer_db(customer_id: str, query_type: str) -> str:
    """Queries the internal enterprise customer database for invoices or profiles."""
    mock_db = {
        "CUST-10492": {
            "PROFILE": {"name": "Srinivas R.", "tier": "ENTERPRISE", "status": "ACTIVE"},
            "INVOICE": {"latest_invoice": "INV-9901", "amount": "$450.00", "due_date": "2026-11-01"}
        }
    }
    
    if customer_id not in mock_db:
        return f"Error: Customer ID {customer_id} not found in database."
        
    return str(mock_db[customer_id].get(query_type, "No record found."))

# Verification
print("Tool Name:", query_customer_db.name)
print("Tool JSON Schema:\n", json.dumps(query_customer_db.args, indent=2))
```
</details>

---

#### 🔴 Exercise 4 (Advanced): Pure-Python Autonomous ReAct Agent Loop
**Problem:** Implement an autonomous ReAct reasoning loop from scratch in pure Python without third-party frameworks. The agent must support registered tools, generate simulated thoughts, execute actions, capture observations, and stop when a final answer is determined.

<details>
<summary><b>View Complete Solution</b></summary>

```python
import re

class MinimalReActAgent:
    def __init__(self, tools: dict):
        self.tools = tools

    def run(self, query: str, max_iterations: int = 5) -> str:
        trace = []
        print(f"User Goal: {query}\n" + "-" * 50)
        
        # Hardcoded simulation of model thought/action cycle for demonstration
        if "stock price" in query.lower() and "square root" in query.lower():
            steps = [
                {"thought": "I need to lookup Nvidia stock price.", "action": "StockLookup", "arg": "NVDA"},
                {"thought": "Now I need to calculate the square root of 128.50.", "action": "Calculator", "arg": "sqrt(128.50)"},
                {"thought": "I have the answer.", "action": "FINISH", "arg": "The square root is approximately 11.34"}
            ]
        else:
            return "Task completed directly."

        for i, step in enumerate(steps, 1):
            thought = step["thought"]
            action = step["action"]
            arg = step["arg"]
            
            print(f"[Turn {i}] Thought: {thought}")
            if action == "FINISH":
                print(f"Final Answer: {arg}")
                return arg
                
            print(f"[Turn {i}] Action: {action}('{arg}')")
            tool_fn = self.tools.get(action)
            observation = tool_fn(arg) if tool_fn else "Error: Tool not found."
            print(f"[Turn {i}] Observation: {observation}\n")

tools = {
    "StockLookup": lambda ticker: "$128.50",
    "Calculator": lambda expr: "11.33579"
}

agent = MinimalReActAgent(tools)
result = agent.run("What is the square root of Nvidia's current stock price?")
```
</details>

---

## 4. ⚙️ Pro Level – Internals & Interview Q&A

### 4.1 Advanced Internals

#### 1. AST Generation and Streaming Propagation in `RunnableSequence`
When you pipe multiple `Runnable` objects (`prompt | model | parser`), LangChain doesn't just execute them sequentially. It builds an internal **Execution Graph**:
- In streaming mode (`.stream()`), intermediate runnables yield chunks immediately.
- If an upstream model emits a chunk `AIMessageChunk(content="Hel")`, the downstream `StrOutputParser` yields `"Hel"` immediately over the generator stream without waiting for the entire completion to finish!

#### 2. LangSmith Tracing & Observability
Every `Runnable` automatically carries an internal `callbacks` manager. When connected to LangSmith:
- Every node records latency, prompt token counts, completion token counts, and input/output payloads.
- Nested ReAct loops log each Thought, Tool Call, and Observation as hierarchical execution spans, providing full enterprise observability.

---

### 4.2 High-Frequency Technical Interview Questions & Answers

#### Q1: What is the primary advantage of the LCEL pipe operator `|` over legacy nested function calls?
<details>
<summary><b>View Detailed Answer</b></summary>
<b>Explanation:</b><br>
LCEL's pipe operator <code>|</code> builds a unified <code>RunnableSequence</code> where every component conforms to the standardized <code>Runnable</code> interface. This provides four crucial architectural capabilities out of the box without manual glue code:
1. <b>Unified Streaming:</b> Tokens stream automatically from model to parser to client over Server-Sent Events (SSE).
2. <b>Native Asynchronous Execution:</b> Calling <code>.ainvoke()</code> or <code>.astream()</code> automatically schedules tasks on the Python <code>asyncio</code> event loop without blocking threads.
3. <b>Optimized Batching:</b> Calling <code>.batch()</code> leverages concurrent thread pools or asynchronous task gathering.
4. <b>Built-in Observability & Tracing:</b> Every node automatically logs inputs, outputs, latencies, and token counts to tracing frameworks like LangSmith.
</details>

#### Q2: Why does an agent need both "Reasoning" and "Acting" (ReAct) rather than just acting alone?
<details>
<summary><b>View Detailed Answer</b></summary>
<b>Explanation:</b><br>
Acting alone without reasoning (pure tool-calling) forces the model to guess which function to trigger based purely on pattern matching. In multi-step or ambiguous tasks, this causes incorrect parameters, premature execution, and compounding errors. 

By enforcing an explicit <b>Thought</b> step before every <b>Action</b>, the model generates an internal Chain-of-Thought (CoT) scratchpad. It tracks what sub-goals have been accomplished, what evidence is missing, which tool is appropriate, and how to format arguments. Furthermore, reasoning over the subsequent <b>Observation</b> allows the model to handle errors, adjust hypotheses, and verify whether the user's objective is satisfied before emitting a final answer.
</details>

#### Q3: When should you use a deterministic LCEL Chain versus an autonomous ReAct Agent?
<details>
<summary><b>View Detailed Answer</b></summary>
<b>Explanation:</b><br>
- <b>Use an LCEL Chain</b> when the workflow is predictable and follows a well-defined directed acyclic graph (DAG). Examples include standard document summarization, fixed RAG pipelines (Retrieve $\to$ Augment $\to$ Generate), sentiment classification, and structured data extraction. Chains are faster, cheaper, deterministic, and easier to debug.
- <b>Use a ReAct Agent</b> when the path to the solution cannot be determined ahead of time and depends on runtime feedback from external tools. Examples include customer support troubleshooting, dynamic SQL queries requiring exploratory schema checks, multi-step math/financial calculations, and autonomous web research.
</details>

#### Q4: What failure mode occurs if you use `ConversationBufferMemory` in a long-running customer service bot?
<details>
<summary><b>View Detailed Answer</b></summary>
<b>Explanation:</b><br>
`ConversationBufferMemory` stores 100% of raw conversation history. As dialogues exceed 20, 50, or 100 turns:
1. <b>Context Window Exhaustion:</b> The accumulated tokens exceed the model's maximum context limit, causing API requests to fail with HTTP 400 Context Length Exceeded errors.
2. <b>Quadratic Cost Explosion:</b> Because every turn re-submits the entire preceding transcript as input prompt tokens, costs escalate quadratically relative to conversation length.
3. <b>Attention Degradation ("Lost in the Middle"):</b> LLMs struggle to attend accurately to relevant facts buried inside massive message histories.

<b>Remedy:</b> Use a sliding window (`ConversationTokenBufferMemory`), running LLM summarization (`ConversationSummaryMemory`), or vector-retrieved semantic memory.
</details>

#### Q5: How does LangChain ensure that LLM tool calls conform strictly to function parameter types?
<details>
<summary><b>View Detailed Answer</b></summary>
<b>Explanation:</b><br>
LangChain extracts function signatures and type hints via <b>Pydantic</b> (`pydantic.BaseModel`) schemas. When tools are bound to a model (`model.bind_tools(tools)`), LangChain automatically converts the Pydantic model into the OpenAPI/JSON Schema required by frontier models (e.g., OpenAI Function Calling). The LLM is trained to emit JSON arguments matching this exact schema. Upon receiving the tool call, LangChain validates the arguments against the Pydantic schema before executing the underlying Python function, rejecting malformed calls before execution occurs.
</details>

#### Q6: How do you persist conversational state across distributed microservice instances using `RunnableWithMessageHistory`?
<details>
<summary><b>View Detailed Answer</b></summary>
<b>Explanation:</b><br>
In stateless, auto-scaling production microservices, session history cannot be stored in local in-memory dictionaries. 

Instead, configure `RunnableWithMessageHistory` with a connection factory that queries a centralized caching layer such as <b>Redis</b> (`RedisChatMessageHistory`) or a relational database via <b>PostgreSQL</b> (`PostgresChatMessageHistory`). Each incoming HTTP request includes a `session_id` header, which the history provider uses to fetch and append conversation turns from the centralized database, ensuring state continuity regardless of which pod handles the request.
</details>

---

## 5. ⚡ Quick Revision (Cheat-Sheet)

```
========================================================================================
                          LANGCHAIN ARCHITECTURE REVISION CHEAT SHEET
========================================================================================

1. THE THREE PILLARS:
   • MODEL WRAPPERS: Standardized interface (.invoke, .stream, .batch). Swappable providers.
   • LCEL CHAINS:    Composable Unix-style pipelines (prompt | model | parser). Fast, deterministic DAGs.
   • REACT AGENTS:   Autonomous reasoning loops (Thought -> Action -> Observation). Dynamic problem-solving.

2. RUNNABLE EXECUTION MODES:
   • .invoke(input):    Synchronous blocking execution.
   • .ainvoke(input):   Asynchronous non-blocking (asyncio).
   • .stream(input):    Token-by-token streaming generator.
   • .batch([inputs]):  Concurrent batch execution via thread pools.

3. LCEL COMPOSITION PRIMITIVES:
   • Pipe Operator (|):      Chains Runnables into a RunnableSequence.
   • RunnablePassthrough:    Passes input through unchanged or appends dictionary keys.
   • RunnableParallel:       Runs multiple Runnables concurrently on identical input.
   • RunnableLambda:         Converts any Python function into a Runnable.

4. MEMORY STRATEGIES:
   • Buffer Memory:   Stores 100% of tokens (Risk: Context explosion!).
   • Window Memory:   Keeps last K turns (Fixed budget, drops distant history).
   • Summary Memory:  LLM summarizes past turns into a narrative (Compacts context).
   • Modern Pattern:  RunnableWithMessageHistory wraps LCEL with Redis / Postgres backend.

5. JAVA / SPRING BOOT DEVELOPER EQUIVALENTS:
   • Spring AI ChatModel       ===> LangChain BaseChatModel
   • Java Streams / Camel      ===> LCEL Pipe Operator (|)
   • Jackson ObjectMapper      ===> JsonOutputParser / PydanticOutputParser
   • Spring @Service with @Tool===> LangChain @tool decorator
   • Spring Session (Redis)    ===> RunnableWithMessageHistory
========================================================================================
```

---

## 6. 🎬 References & Visual Learning Videos

### 6.1 🇮🇳 Telugu Tech Video References
For native Telugu speakers, these curated video tutorials explain LangChain, chains, and agents step-by-step:

| # | Topic / Video Title | Channel / Creator | Search Query | Highlights |
|---|---|---|---|---|
| 1 | **LangChain Complete Tutorial in Telugu** | **Python Life Telugu** | `Python Life Telugu LangChain Generative AI` | Comprehensive introduction to LangChain components, models, and chains in Telugu. |
| 2 | **Building AI Apps with LangChain in Telugu** | **Vamsi Bhavani** | `Vamsi Bhavani LangChain Gen AI` | Practical walkthrough of connecting OpenAI models, building prompts, and using tools in Telugu. |
| 3 | **LangChain Agents & Chains in Telugu** | **Telugu Tech Tutorials** | `Telugu Tech LangChain Agents Tutorial` | Step-by-step guide to building ReAct agents and tool execution in Telugu. |

---

### 6.2 🎥 3D Animated & World-Class Visual Deep Dives

| # | Topic / Video Title | Channel / Creator | Search Query | Visual & Technical Highlights |
|---|---|---|---|---|
| 1 | **How LangChain Works & System Design** | **ByteByteGo** | `ByteByteGo LangChain Architecture` | Visual animations explaining model wrappers, LCEL pipelines, and agent loops from a system design perspective. |
| 2 | **LangChain Crash Course for Beginners** | **freeCodeCamp.org** | `freeCodeCamp LangChain Crash Course` | Comprehensive hands-on tutorial covering models, prompts, LCEL, and vector stores. |
| 3 | **LangChain Chains & Agents Clearly Explained!** | **StatQuest with Josh Starmer** | `StatQuest LangChain Clearly Explained` | Step-by-step visual breakdown of chains, memory, and ReAct agent loops with zero jargon. |
| 4 | **Learn RAG From Scratch (LangChain & LCEL)** | **freeCodeCamp.org (Lance Martin)** | `freeCodeCamp Learn RAG From Scratch Lance Martin` | Deep-dive architectural walkthrough of LCEL routing, vector search, and document retrieval. |
| 5 | **State of GPT & Autonomous Tool Use** | **Andrej Karpathy** | `Andrej Karpathy State of GPT Microsoft Build` | The definitive masterclass on system prompts, function calling, and agent reasoning traces. |

---

### 6.3 📚 Foundational Research Papers & Framework Docs
1. **Yao, S., et al. (2022).** *"ReAct: Synergizing Reasoning and Acting in Language Models."* ICLR 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
2. **LangChain Core Documentation:** [python.langchain.com](https://python.langchain.com/)
3. **LangChain Expression Language (LCEL) Concept Guide:** [python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)
