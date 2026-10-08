# 03. Functions, Arguments, Keyword Arguments (*args, **kwargs), and Lambdas (for Java Developers)

---

## 0. Why this topic matters

In Java, every piece of business logic must live inside a `class`. You cannot have standalone functions; you define methods, declare strict parameter types, overload methods for optional parameters, or use the Builder Pattern for complex configurations.

In modern Python and Generative AI frameworks (such as PyTorch, Hugging Face `transformers`, and LangChain), functions are **first-class citizens**. You pass functions as arguments (e.g., custom tokenizers, prompt formatters, tool call handlers). Furthermore, almost every AI API uses `*args` and `**kwargs` to dynamically forward hyperparameter dictionaries (`temperature`, `top_p`, `model_kwargs`) through multiple architectural layers without modifying intermediate function signatures.

Mastering Python's function mechanics and understanding how they map to Java concepts (like `Method`, `varargs`, and `java.util.function.*`) eliminates confusion when reading and writing modern AI libraries.

---

## 1. Basic Level – "Explain like I'm new"

### 1.1 Python Functions vs. Java Methods

- In Java, a function cannot exist in isolation; it must belong to a class and specify an explicit return type and access modifier (`public static int calculateTokens(...)`).
- In Python, a function is declared at the module level using the `def` keyword, does not require an enclosing class, and returns whatever value it wants dynamically (or returns `None` if no return statement is reached).

```python
# Python: Standalone function
def calculate_cost(tokens: int, rate: float = 0.00002) -> float:
    """Calculates dollar cost for a given token count."""
    return tokens * rate

cost = calculate_cost(1500)
print(f"Total cost: ${cost:.5f}")
# Output: Total cost: $0.03000
```

```java
// Java: Requires class, method visibility, explicit typing
public class PricingCalculator {
    public static double calculateCost(int tokens, double rate) {
        return tokens * rate;
    }
    // Method overloading needed for default arguments in Java:
    public static double calculateCost(int tokens) {
        return calculateCost(tokens, 0.00002);
    }
}
```

### 1.2 Default Arguments vs. Java Method Overloading

In Java, if you want an optional parameter, you must overload the method:
```java
public void generateText(String prompt) {
    generateText(prompt, 0.7, 512);
}
public void generateText(String prompt, double temperature, int maxTokens) {
    // implementation
}
```

In Python, you specify default values directly in the function signature:
```python
def generate_text(prompt: str, temperature: float = 0.7, max_tokens: int = 512):
    print(f"Prompt: {prompt} | Temp: {temperature} | MaxTokens: {max_tokens}")

# Call with just the required argument:
generate_text("Summarize quarterly revenue")
# Output: Prompt: Summarize quarterly revenue | Temp: 0.7 | MaxTokens: 512

# Call with positional overrides:
generate_text("Write poem", 0.9, 100)

# Call with named keyword arguments (order does not matter!):
generate_text("Classify sentiment", max_tokens=256, temperature=0.1)
```

---

## 2. Building Up – Concepts Added One by One

### 2.1 Keyword Arguments (Named Parameters)

In Java, all method invocations are positional: `send(to, from, subject, body)`. If arguments have the same type, passing them in the wrong order causes silent bugs.

Python supports **Named Keyword Arguments**:
```python
def configure_retriever(collection_name: str, top_k: int = 5, score_threshold: float = 0.75, rerank: bool = False):
    return f"Retriever[{collection_name}]: top_{top_k}, threshold_{score_threshold}, rerank={rerank}"

# You can pass arguments by name in ANY order:
retriever = configure_retriever(
    top_k=10, 
    collection_name="clinical_guidelines", 
    rerank=True
)
print(retriever)
# Output: Retriever[clinical_guidelines]: top_10, threshold_0.75, rerank=True
```

---

### 2.2 Variable Positional Arguments: `*args` vs. Java `...` (Varargs)

In Java, you accept a variable number of positional arguments using the ellipsis syntax `Type... args`:
```java
public static int sumTokens(int... tokens) {
    int sum = 0;
    for (int t : tokens) sum += t;
    return sum;
}
```

In Python, the `*args` syntax packs arbitrary extra positional arguments into an **immutable `tuple`**:

```python
def aggregate_token_usage(*token_batches):
    # token_batches is a tuple of all positional arguments passed
    print(f"Received {len(token_batches)} batches: {token_batches}")
    return sum(token_batches)

total = aggregate_token_usage(140, 250, 89, 512)
print("Total tokens:", total)
# Output:
# Received 4 batches: (140, 250, 89, 512)
# Total tokens: 991
```

---

### 2.3 Variable Keyword Arguments: `**kwargs` (The Java Developer's Swiss Army Knife)

Java has **no direct language equivalent** for `**kwargs`. To pass arbitrary named key-value options in Java, you pass a `Map<String, Object>` or use a configuration POJO/Builder pattern.

In Python, `**kwargs` (short for *keyword arguments*) captures any unspecified named arguments into a **dictionary `dict`**:

```python
def call_llm(prompt: str, model: str = "gpt-4o", **generation_kwargs):
    print(f"Invoking {model} with prompt: '{prompt}'")
    print(f"Additional hyperparameter flags: {generation_kwargs}")
    
    # You can inspect kwargs dynamically using dict methods:
    temperature = generation_kwargs.get("temperature", 0.7)
    seed = generation_kwargs.get("seed", 42)
    print(f"Applied temp={temperature}, seed={seed}")

# Call with arbitrary extra parameters:
call_llm(
    "Explain quantum superposition",
    model="claude-3-5-sonnet",
    temperature=0.3,
    top_p=0.95,
    presence_penalty=0.5,
    custom_tag="production_v1"
)
```

#### Output
```
Invoking claude-3-5-sonnet with prompt: 'Explain quantum superposition'
Additional hyperparameter flags: {'temperature': 0.3, 'top_p': 0.95, 'presence_penalty': 0.5, 'custom_tag': 'production_v1'}
Applied temp=0.3, seed=42
```

#### Why This Is Everywhere in AI Frameworks:
When LangChain or Hugging Face wraps an LLM, their wrapper function accepts `**kwargs` and transparently forwards unknown parameters to the underlying provider API (OpenAI, Anthropic, Bedrock, Ollama) without needing to update their class definition every time an API provider adds a new parameter!

```python
def wrapper_pipeline(input_query: str, **kwargs):
    # Unpack the dictionary into the downstream call using **
    return downstream_llm_call(prompt=input_query, **kwargs)
```

---

### 2.4 First-Class Functions vs. Java Functional Interfaces

In Java 8+, functions are objects wrapped inside functional interfaces (`Function<T, R>`, `Predicate<T>`, `Consumer<T>`, `Supplier<T>`).

In Python, **functions are ordinary objects** (`PyFunctionObject`):
- You can assign a function to a variable.
- You can store functions in lists or dictionaries.
- You can pass functions into other functions as arguments (Higher-Order Functions).
- You can return functions from inside other functions (Closures & Decorators).

```python
# 1. Store functions in a dictionary (Strategy Pattern without classes!)
def format_chat(prompt: str) -> str:
    return f"<|user|>\n{prompt}\n<|assistant|>"

def format_qa(prompt: str) -> str:
    return f"Question: {prompt}\nAnswer:"

# Dictionary of functions
template_strategies = {
    "chat": format_chat,
    "qa": format_qa
}

# Invoke dynamically
chosen_strategy = template_strategies["chat"]
print(chosen_strategy("How does backpropagation work?"))
```

#### Output
```
<|user|>
How does backpropagation work?
<|assistant|>
```

---

### 2.5 Lambdas vs. Java Lambda Expressions

Both Java and Python support anonymous functions (lambdas). However, Python lambdas are intentionally restricted to a **single inline expression** (they cannot contain statements like `if/else` blocks, loops, or assignments).

#### Java Lambda
```java
Function<String, Integer> tokenCounter = s -> s.split("\\s+").length;
int count = tokenCounter.apply("Hello world from Java");
```

#### Python Lambda
```python
# Syntax: lambda <arguments>: <expression>
token_counter = lambda s: len(s.split())
count = token_counter("Hello world from Python")
print("Tokens:", count)  # Output: Tokens: 4

# Most common practical usage: Sorting keys
candidates = [
    {"doc": "doc_a", "score": 0.82},
    {"doc": "doc_b", "score": 0.95},
    {"doc": "doc_c", "score": 0.74}
]

# Sort candidates by score descending using a lambda
sorted_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
for c in sorted_candidates:
    print(f"{c['doc']} -> {c['score']}")
```

#### Output
```
doc_b -> 0.95
doc_a -> 0.82
doc_c -> 0.74
```

---

## 3. Practice Exercises (Easy to Hard)

### Exercise 1: Flexible Temperature Validator (Easy)
**Task**: Write a function `validate_temperature(temp: float = 0.7) -> bool` that verifies if a temperature is within the valid range `[0.0, 2.0]`. If no value is provided, it should evaluate the default `0.7`.

```python
# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
def validate_temperature(temp: float = 0.7) -> bool:
    """Validates if the sampling temperature is bounded between 0.0 and 2.0."""
    return 0.0 <= temp <= 2.0

print(validate_temperature())        # True (default 0.7)
print(validate_temperature(1.5))     # True
print(validate_temperature(2.5))     # False
print(validate_temperature(-0.1))    # False
```
</details>

---

### Exercise 2: Dynamic LLM Request Payload Builder with `**kwargs` (Easy)
**Task**: Write a function `build_payload(prompt: str, model: str = "gpt-4o", **options) -> dict` that returns a dictionary payload for an HTTP request. The dictionary must contain `"model"`, `"messages"` (formatted as a single user message), and all additional options passed through `**options`.

```python
# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
def build_payload(prompt: str, model: str = "gpt-4o", **options) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    # Merge additional options (temperature, top_p, stream, etc.)
    payload.update(options)
    return payload

req = build_payload(
    "Explain attention heads", 
    temperature=0.2, 
    max_tokens=256, 
    stop=["\n\n"]
)
print(req)

# Expected Output:
# {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': 'Explain attention heads'}], 'temperature': 0.2, 'max_tokens': 256, 'stop': ['\n\n']}
```
</details>

---

### Exercise 3: Prompt Pipeline with Function Composition (Medium)
**Task**: In LLM workflows, text passes through multiple preprocessing steps (sanitizing, injecting system prefixes, truncating).
Write a function `apply_pipeline(text: str, *transforms) -> str` that accepts an initial text string and an arbitrary sequence of transformation functions (`*transforms`), passing the output of each function as input to the next.

```python
def clean_whitespace(s: str) -> str:
    return " ".join(s.split())

def add_system_prefix(s: str) -> str:
    return f"[SYSTEM]: {s}"

def enforce_uppercase(s: str) -> str:
    return s.upper()

# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
def apply_pipeline(text: str, *transforms) -> str:
    result = text
    for transform_fn in transforms:
        result = transform_fn(result)
    return result

raw_input = "   Mastering    transformer     architectures   "
final_prompt = apply_pipeline(raw_input, clean_whitespace, add_system_prefix, enforce_uppercase)

print("Processed Prompt:", final_prompt)

# Expected Output:
# Processed Prompt: [SYSTEM]: MASTERING TRANSFORMER ARCHITECTURES
```
</details>

---

### Exercise 4: Keyword-Only Arguments and Strict Parameter Enforcement (Medium)
**Task**: Write a function `deploy_model(model_name: str, *, environment: str = "staging", gpu_enabled: bool = False, max_replicas: int = 1)` where `environment`, `gpu_enabled`, and `max_replicas` **must** be passed as keyword arguments (cannot be passed positionally). Verify that calling `deploy_model("llama-3", "production")` raises a `TypeError`.

```python
# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
def deploy_model(model_name: str, *, environment: str = "staging", gpu_enabled: bool = False, max_replicas: int = 1):
    """The bare asterisk (*) forces all subsequent parameters to be keyword-only."""
    return {
        "model": model_name,
        "env": environment,
        "gpu": gpu_enabled,
        "replicas": max_replicas
    }

# Valid call using named keyword arguments:
config = deploy_model("llama-3-8b", environment="production", gpu_enabled=True, max_replicas=4)
print("Config:", config)

# Testing positional enforcement:
try:
    deploy_model("llama-3-8b", "production")  # ❌ Attempting to pass environment positionally
except TypeError as e:
    print("Caught expected TypeError:", e)

# Expected Output:
# Config: {'model': 'llama-3-8b', 'env': 'production', 'gpu': True, 'replicas': 4}
# Caught expected TypeError: deploy_model() takes 1 positional argument but 2 were given
```
</details>

---

### Exercise 5: Building a Retry Decorator with `*args` and `**kwargs` (Hard)
**Task**: In LLM engineering, API calls fail intermittently due to network blips or rate limits (`429 Too Many Requests`).
Write a higher-order function (closure/decorator) `with_retry(max_attempts=3)` that wraps any API client function, executes it, and retries up to `max_attempts` if an exception occurs before re-raising the error.

```python
# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
import time

def with_retry(max_attempts: int = 3):
    """A higher-order function returning a wrapper decorator."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    print(f"[Warning] Attempt {attempt}/{max_attempts} failed: {e}")
                    last_exception = e
                    time.sleep(0.1)  # Brief backoff
            print("[Error] All retry attempts exhausted.")
            raise last_exception
        return wrapper
    return decorator

# Mock flaky API function
call_count = 0

@with_retry(max_attempts=3)
def simulate_flaky_llm_call(prompt: str, temperature: float = 0.7):
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError("API connection timeout")
    return f"Success response for prompt '{prompt}' (temperature={temperature})"

# Test invocation:
response = simulate_flaky_llm_call("Explain backpropagation", temperature=0.5)
print("Result:", response)

# Expected Output:
# [Warning] Attempt 1/3 failed: API connection timeout
# [Warning] Attempt 2/3 failed: API connection timeout
# Result: Success response for prompt 'Explain backpropagation' (temperature=0.5)
```
</details>

---

## 4. Pro Level – Internals, Edge Cases & Interview Q&A

### 4.1 The Classic Python Trap: Mutable Default Arguments
In Java, default values in overloaded methods are evaluated every time the method is invoked:
```java
public void record(String item) {
    List<String> list = new ArrayList<>(); // Fresh instance created on every call
    list.add(item);
}
```

In Python, **default arguments are evaluated ONCE at function definition time**, when the `.py` file is parsed! If you use a mutable object (`list`, `dict`, `set`) as a default argument, **all invocations share the exact same instance in memory**:

```python
# ⚠️ BUG: The list is instantiated once at function creation time!
def append_to_history(message: str, history: list = []):
    history.append(message)
    return history

print(append_to_history("Hello"))   # ['Hello']
print(append_to_history("World"))   # ['Hello', 'World'] -> ❌ BUG: Leaked from previous call!
```

#### The Idiomatic Python Solution: Use `None` as Sentinel
```python
# ✅ Safe, standard Python pattern:
def append_to_history(message: str, history: list = None):
    if history is None:
        history = []  # Fresh instance allocated per invocation
    history.append(message)
    return history

print(append_to_history("Hello"))   # ['Hello']
print(append_to_history("World"))   # ['World'] -> ✅ Clean!
```

---

### 4.2 Parameter Unpacking Mechanics (`*` and `**`)

The asterisks have two distinct meanings based on context:
1. **In a Function Definition**: Packs incoming parameters into a tuple (`*args`) or dictionary (`**kwargs`).
2. **In a Function Call**: Unpacks an existing iterable or dictionary into arguments!

```python
# Unpacking a dictionary into keyword arguments
llm_options = {
    "temperature": 0.2,
    "max_tokens": 512,
    "top_p": 0.95
}

def generate(prompt: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 1.0):
    return f"Prompt='{prompt}', T={temperature}, MaxT={max_tokens}, TopP={top_p}"

# Unpack dictionary directly:
result = generate("Summarize document", **llm_options)
print(result)
# Output: Prompt='Summarize document', T=0.2, MaxT=512, TopP=0.95
```

---

### 4.3 Top Technical Interview Questions & Answers

#### Q1: What is the exact execution order of arguments in Python function signatures?
**Answer**:
A Python function signature must strictly follow this syntactic order:
1. Standard positional arguments: `(a, b)`
2. Positional arguments with defaults: `(c=10)`
3. Variable positional arguments: `*args`
4. Keyword-only arguments: `(d, e=20)`
5. Variable keyword arguments: `**kwargs`

```python
def full_signature(a, b=2, *args, kw_only, kw_default=10, **kwargs):
    pass
```

#### Q2: How does Python's LEGB Scope Rule work, and how does it differ from Java scoping?
**Answer**:
In Java, variables are strictly block-scoped within `{ ... }`.
In Python, variables are resolved via the **LEGB** lookup hierarchy:
1. **L**ocal: Defined inside the current function.
2. **E**nclosing: Defined inside an outer/enclosing function (closures).
3. **G**lobal: Defined at the top-level module file.
4. **B**uilt-in: Pre-loaded Python names (`len`, `range`, `print`).

If a variable is read, Python checks L ➔ E ➔ G ➔ B. If you assign to a variable inside a function without the `global` or `nonlocal` keyword, Python creates a new local variable shadowing any outer variable.

---

## 5. Quick Revision (Cheat-Sheet)

### 5.1 Java vs. Python Function Equivalents

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ JAVA CONSTRUCT                  │ PYTHON CONSTRUCT                          │
├─────────────────────────────────┼───────────────────────────────────────────┤
│ Method inside class             │ Standalone function with `def`            │
│ Method overloading (for defaults│ Default parameter values in signature     │
│ Varargs: `int... numbers`       │ Positional packing: `*args` (tuple)       │
│ Map<String, Object> configMap   │ Keyword packing: `**kwargs` (dict)        │
│ No direct equivalent            │ Named keyword calling: `fn(top_k=5)`      │
│ No direct equivalent            │ Keyword-only barrier: `def fn(a, *, b):`  │
│ Function<T, R> / Predicate<T>   │ First-class functions (passed directly)   │
│ `(x) -> x * 2`                  │ `lambda x: x * 2`                         │
│ Method reference: `String::trim`│ Function identifier without parens: `trim`│
│ Checked exceptions in signature │ No checked exceptions; raises dynamically │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 5.2 Key Takeaways
1. **No Overloading**: You cannot define two functions with the same name and different arguments in Python; the second definition will overwrite the first. Use default arguments instead.
2. **Never Mutable Defaults**: Always use `def fn(items=None): if items is None: items = []`. Never use `items=[]`.
3. **`*args` vs `**kwargs`**: `*args` captures unmapped positional inputs as a `tuple`; `**kwargs` captures unmapped keyword inputs as a `dict`.
4. **Unpacking**: Use `*` to unpack lists/tuples into positional parameters, and `**` to unpack dictionaries into named parameters.

---

## 6. References & Recommended Videos

To master Python functions, scoping, and closures:

| Topic | Channel / Creator | Exact Search Phrase | Why Watch |
|---|---|---|---|
| **Python Functions & Parameter Passing** | Corey Schafer | `Corey Schafer Python Tutorial: Functions` | The definitive breakdown of positional vs. keyword arguments, return values, and scopes. |
| **`*args` and `**kwargs` Explained** | ArjanCodes | `ArjanCodes args and kwargs in Python` | Architectural perspective on when to use dynamic arguments and when they make code unmaintainable. |
| **The Mutable Default Argument Gotcha** | mCoding | `mCoding Python's Default Arguments are Dangerous` | A short, visual demonstration of the CPython function object memory trap. |
| **First-Class Functions & Closures** | Corey Schafer | `Corey Schafer First-Class Functions` | Bridges the conceptual gap from Java interfaces to Python higher-order functions. |
| **Decorators Masterclass** | ArjanCodes | `ArjanCodes Decorators in Python` | Explains how Python decorators (`@with_retry`, `@app.route`) leverage closures, `*args`, and `**kwargs`. |
