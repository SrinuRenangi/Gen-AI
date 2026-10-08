# 01. Python Syntax, Variables, and Data Types (for Java Developers)

---

## 0. Why this topic matters

Every modern Generative AI framework—from PyTorch and Hugging Face to LangChain, LlamaIndex, and vLLM—is built natively in Python. As a Java/Spring Boot developer, your existing understanding of object-oriented architecture, memory, and clean code is a massive superpower; mastering Python's dynamic typing, clean syntax, and execution model is the exact bridge that unlocks building production-grade AI systems.

---

## 1. Basic Level – "Explain like I'm new"

### 1.1 What is Python?
In Java, you write source code (`.java`), compile it with `javac` into bytecode (`.class`), and execute it on the Java Virtual Machine (JVM).

Python is an **interpreted, dynamically typed language**. You write `.py` files and hand them directly to the Python interpreter (`python main.py`). The interpreter reads your code line-by-line, compiles it in-memory to bytecode, and executes it immediately.

```
Java:    [Code .java] ──(javac)──> [Bytecode .class] ──(JVM)──> [Machine Execution]
Python:  [Code .py]   ───────────> [Python Interpreter]  ─────> [Machine Execution]
```

### 1.2 The Indentation Rule (No More Semicolons or Curly Braces `{}`)
In Java, blocks of code belong inside curly braces `{ ... }`, and statements end with semicolons `;`. You can format your Java code onto one ugly continuous line and the compiler still runs it.

In Python, **whitespace and indentation are syntax**:
- There are no semicolons `;` at the end of lines.
- There are no curly braces `{}` to define code blocks.
- A block of code (like inside an `if` statement or method) is defined strictly by **4 spaces of indentation**.

```python
# Python defines code blocks with a colon (:) and 4 spaces
if temperature > 0.7:
    print("Creative mode active")
    print("Generating diverse tokens")
print("This line is outside the if-block")
```

```java
// Java equivalent
if (temperature > 0.7) {
    System.out.println("Creative mode active");
    System.out.println("Generating diverse tokens");
}
System.out.println("This line is outside the if-block");
```

### 1.3 Variables Do Not Have Type Declarations
In Java, you must declare variable types before using them:
```java
int maxTokens = 512;
String modelName = "gpt-4o";
double temperature = 0.7;
boolean isStreaming = true;
```

In Python, you do **not** declare types. You simply assign a value with `=`. Python automatically detects the type at runtime (**Duck Typing**):
```python
max_tokens = 512            # int
model_name = "gpt-4o"       # str (string)
temperature = 0.7           # float
is_streaming = True         # bool (Note capital T)
```

---

## 2. Building Up – Concepts Added One by One

### 2.1 The Primitive Data Types: Java vs. Python

| Conceptual Type | Java Type | Python Type | Python Example | Key Difference |
|---|---|---|---|---|
| **Integer** | `byte`, `short`, `int`, `long` | `int` | `batch_size = 32` | Python's `int` has **unlimited precision**. It never overflows into negative numbers! |
| **Decimal** | `float`, `double` | `float` | `learning_rate = 0.0002` | Python's `float` is equivalent to Java's 64-bit `double`. |
| **Text** | `char`, `String` | `str` | `prompt = "Hello AI"` | Python has no `char` type; single characters are just length-1 strings. Both `'text'` and `"text"` are identical. |
| **Boolean** | `boolean`, `Boolean` | `bool` | `is_ready = True` | Must be capitalized: `True` and `False` (not `true`/`false`). |
| **Absence of Value** | `null` | `None` | `response = None` | Represented by the singleton object `None` (type `NoneType`). |

---

### 2.2 Deep-Dive: Strings and Modern String Interpolation (f-strings)

#### Definition
In Java, you concatenate strings with `+`, `String.format()`, or `StringBuilder`. In modern Python (3.6+), you use **f-strings** (formatted string literals) prefixed with `f"..."`. Expressions inside `{}` are evaluated directly.

#### Code Example
```python
model = "Meta-Llama-3-8B"
tokens = 1024
cost_per_token = 0.000015

# Modern Python f-string
message = f"Model: {model} | Total Cost: ${tokens * cost_per_token:.4f}"
print(message)
```

#### Output
```
Model: Meta-Llama-3-8B | Total Cost: $0.0154
```

#### Java Comparison
```java
// Java String.format equivalent
String model = "Meta-Llama-3-8B";
int tokens = 1024;
double costPerToken = 0.000015;

String message = String.format("Model: %s | Total Cost: $%.4f", model, tokens * costPerToken);
System.out.println(message);
```

#### Common Mistake
> Forgetting the leading `f` before the quote: `"Model: {model}"` will print the literal characters `{model}` instead of the variable's value!

---

### 2.3 Type Inspection and Dynamic Re-assignment

#### Definition
Because Python is dynamically typed, a variable name is just a pointer or label referencing an object in memory. You can inspect any variable's type using `type()`.

#### Code Example
```python
x = 42
print(f"Value: {x}, Type: {type(x)}")

# Re-assigning to a string (Allowed in Python, would fail compilation in Java!)
x = "Now I am a string"
print(f"Value: {x}, Type: {type(x)}")
```

#### Output
```
Value: 42, Type: <class 'int'>
Value: Now I am a string, Type: <class 'str'>
```

#### Java Comparison
```java
// In Java, this fails at compile time:
int x = 42;
x = "Now I am a string"; // COMPILE ERROR: incompatible types: String cannot be converted to int
```

#### Common Mistake
> While Python allows changing a variable's type on the fly, doing so arbitrarily creates subtle bugs in production code. Maintain single-type consistency for variables.

---

### 2.4 Truthiness & Falsiness (Implicit Boolean Conversion)

#### Definition
In Java, `if (condition)` strictly requires a `boolean` expression (`true` or `false`). Passing an integer or object reference (`if (count)`) triggers a compilation error.

In Python, **every object has an innate truth value**:
- **Falsy values**: `False`, `None`, `0`, `0.0`, empty string `""`, empty list `[]`, empty dict `{}`, empty set `set()`.
- **Truthy values**: Everything else! Any non-zero number, any non-empty string, or any collection with items.

#### Code Example
```python
user_input = ""       # Empty string is Falsy
chat_history = []     # Empty list is Falsy
active_users = 5      # Non-zero integer is Truthy

if not user_input:
    print("User prompt cannot be empty!")

if not chat_history:
    print("Starting a fresh conversation.")

if active_users:
    print(f"Server active with {active_users} users.")
```

#### Output
```
User prompt cannot be empty!
Starting a fresh conversation.
Server active with 5 users.
```

#### Java Comparison
```java
// In Java, you must write explicit checks:
String userInput = "";
List<String> chatHistory = new ArrayList<>();
int activeUsers = 5;

if (userInput.isEmpty()) {
    System.out.println("User prompt cannot be empty!");
}
if (chatHistory.isEmpty()) {
    System.out.println("Starting a fresh conversation.");
}
if (activeUsers > 0) {
    System.out.println("Server active with " + activeUsers + " users.");
}
```

---

### 2.5 `None` vs. `null`: Safe Identity Checks

#### Definition
Python's `None` is an object representing the absence of a value (identical conceptually to Java's `null`). However, in Python, checking if something is `None` should **always** use the identity operator `is`, not the equality operator `==`.

#### Code Example
```python
api_key = None

# Correct Pythonic check (checks memory identity)
if api_key is None:
    print("Warning: API key is not configured!")

# Check if value exists
api_key = "sk-live-12345"
if api_key is not None:
    print("API key loaded successfully.")
```

#### Java Comparison
```java
// Java comparison:
String apiKey = null;

if (apiKey == null) {
    System.out.println("Warning: API key is not configured!");
}
```

#### Common Mistake
> Writing `if api_key == None:`. While it often works, custom classes can overload `__eq__`, causing unexpected bugs. Always use `is None` or `is not None`.

---

## 3. Practice Exercises

Try writing the code for each problem before reading the solution at the end of this section!

### Exercise 1 (Easy): Temperature Validator
Write a script that takes a variable `temperature = 0.85`. If the temperature is greater than `1.0` or less than `0.0`, print `"Invalid temperature"`. Otherwise, print `"Valid temperature: <value>"` formatted to 2 decimal places using an f-string.

### Exercise 2 (Easy): Truthiness Guard
Create a variable `retrieved_chunks = []`. Using Python's implicit truthiness (without using `len()`), write an `if`/`else` check that prints `"No documents retrieved from vector database"` if the list is empty, and `"Documents found"` if it contains items.

### Exercise 3 (Medium): Cost Calculator
Given `input_tokens = 450`, `output_tokens = 125`, `input_cost_per_million = 2.50`, and `output_cost_per_million = 10.00`, calculate the total request cost in dollars and print it in the format: `Total Cost: $0.002375`.

### Exercise 4 (Medium): Safe Type Coercion
You receive user inputs as strings from an HTTP form: `raw_temperature = "0.7"` and `raw_max_tokens = "256"`. Convert them into their proper Python float and int types, multiply them (`temperature * max_tokens`), and print the result and its type.

### Exercise 5 (Hard): String Template Sanitizer
Given an input prompt `raw_prompt = "   tell me about quantum computing   "`, strip all leading/trailing whitespace, convert it to Title Case, and print it enclosed in XML tags: `<query>Tell Me About Quantum Computing</query>`.

---

### Solutions

```python
# --- Solution 1 ---
temperature = 0.85
if temperature < 0.0 or temperature > 1.0:
    print("Invalid temperature")
else:
    print(f"Valid temperature: {temperature:.2f}")

# --- Solution 2 ---
retrieved_chunks = []
if not retrieved_chunks:
    print("No documents retrieved from vector database")
else:
    print("Documents found")

# --- Solution 3 ---
input_tokens = 450
output_tokens = 125
input_cost_per_million = 2.50
output_cost_per_million = 10.00

total_cost = ((input_tokens / 1_000_000) * input_cost_per_million) + \
             ((output_tokens / 1_000_000) * output_cost_per_million)
print(f"Total Cost: ${total_cost:.6f}")

# --- Solution 4 ---
raw_temperature = "0.7"
raw_max_tokens = "256"

temperature = float(raw_temperature)
max_tokens = int(raw_max_tokens)
result = temperature * max_tokens
print(f"Result: {result}, Type: {type(result)}")

# --- Solution 5 ---
raw_prompt = "   tell me about quantum computing   "
cleaned = raw_prompt.strip().title()
print(f"<query>{cleaned}</query>")
```

---

## 4. Pro Level: Internals, Performance & Interview Scenarios

### 4.1 Python's Memory Model: "Pass-by-Assignment"
Java passes everything by value. For primitive types, it passes the raw value; for objects, it passes a copy of the reference.

In Python, **everything is an object**, and variables are references pointing to memory addresses.
- **Immutable Types** (`int`, `float`, `str`, `tuple`, `bool`): Modifying them creates a *new* object in memory.
- **Mutable Types** (`list`, `dict`, `set`): Modifying them mutates the *existing* object in place.

```python
# Demonstrating Python object IDs (memory addresses)
a = 10
b = a
print(f"Address of a: {id(a)}, Address of b: {id(b)}")  # Same memory address!

a = a + 1  # Rebinds 'a' to a new integer object
print(f"Address of a: {id(a)}, Address of b: {id(b)}")  # 'a' has a new address; 'b' is still 10!
```

### 4.2 Type Hinting in Python (`typing` Module)
If you miss Java's compile-time safety, modern Python supports **Type Hints** (PEP 484). While Python will not prevent execution at runtime if types mismatch, static analysis tools (like `mypy`) check your code just like the Java compiler:

```python
from typing import Optional

def calculate_token_cost(
    prompt: str,
    max_tokens: int,
    cost_per_token: float = 0.00002,
    model_alias: Optional[str] = None
) -> float:
    estimated_tokens: int = len(prompt.split()) + max_tokens
    return estimated_tokens * cost_per_token
```

### 4.3 Interview Scenario: Integer Caching in Python (The Small Integer Pool)
**Question**: What does the following code print in Python, and why?
```python
a = 256
b = 256
print(a is b)

x = 257
y = 257
print(x is y)
```
**Answer**:
```
True
False (in interactive repl / separate bytecode blocks)
```
**Explanation**: CPython pre-allocates an array of small integer objects in memory for numbers in the range **$-5$ to $256$** when the interpreter starts up. Any integer in this range reuses the exact same singleton memory address (`a is b` evaluates to `True`). For integers $\ge 257$, distinct objects are allocated. This is similar to Java's `Integer.valueOf()` caching integers between $-128$ and $127$!

---

## 5. Quick Revision & Cheat Sheet

```
+----------------------------------------------------------------------------------------------------+
|                                PYTHON SYNTAX CHEAT SHEET FOR JAVA DEVS                             |
+----------------------------------------------------------------------------------------------------+
|  Concept              | Java Syntax                         | Python Syntax                        |
+----------------------------------------------------------------------------------------------------+
|  Print to console     | System.out.println("Hello");        | print("Hello")                       |
|  String formatting    | String.format("Val: %d", x);        | f"Val: {x}"                          |
|  Constant / Final     | final int MAX = 100;                | MAX = 100 (convention: UPPERCASE)   |
|  Logical AND          | &&                                  | and                                  |
|  Logical OR           | ||                                  | or                                   |
|  Logical NOT          | !flag                               | not flag                             |
|  Null check           | obj == null                         | obj is None                          |
|  Block structure      | { ... }                             | : followed by 4-space indent         |
|  Line comment         | // comment                          | # comment                            |
|  Multi-line string    | """ text """ (Java 15+)             | """ text """                         |
|  Type conversion      | Integer.parseInt("10")              | int("10")                            |
+----------------------------------------------------------------------------------------------------+
```

---

## 6. References & Curated Video Resources

| Topic | Channel | Video Title & Search Phrase | Why Watch |
|---|---|---|---|
| **Python for Java Developers** | Programming with Mosh | `Python Tutorial for Beginners [Full Course]` | Crisp overview of Python syntax with zero fluff; ideal for existing programmers. |
| **Python Variables & Memory Model** | Corey Schafer | `Python Tutorial: Variable Scope and Memory Internals` | Master how Python handles object references, mutability, and assignment. |
| **f-strings Deep Dive** | mCoding | `Python f-strings Can Do WHAT?! (Advanced Formatting)` | Learn advanced formatting syntax, alignment, and debugging tricks inside f-strings. |
| **Python Truth Value Testing** | ArjanCodes | `Python Truth Value Testing (Truthy and Falsy Explained)` | Clear architectural guide on how Python evaluates conditions and expressions cleanly. |
