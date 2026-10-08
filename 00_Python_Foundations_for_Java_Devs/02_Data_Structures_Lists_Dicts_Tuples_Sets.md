# 02. Data Structures: Lists, Dictionaries, Tuples, and Sets (for Java Developers)

---

## 0. Why this topic matters

In Generative AI and Machine Learning, virtually all data pipelines handle text tokens as arrays, prompt templates as key-value pairs, embeddings as numerical vectors, and vocabularies as distinct sets. 

In Java, data manipulation requires importing `java.util.*`, choosing specific interface implementations (`ArrayList`, `HashMap`, `HashSet`), handling generic types (`Map<String, List<Integer>>`), and writing multi-line boilerplate or verbose Streams pipelines (`.stream().filter(...).map(...).collect(...)`).

Python makes data structures first-class citizens of the language with native literal syntax (`[]`, `{}`, `()`), powerful slicing syntax (`tokens[:5]`), and concise comprehensions. Understanding how Python's built-in collections map directly to Java collections—along with their runtime behaviors and memory footprints—is essential for writing clean, performant AI pipelines.

---

## 1. Basic Level – "Explain like I'm new"

### 1.1 The Four Core Built-in Collections

Python provides four foundational collection types built directly into the language syntax without requiring any imports:

| Python Collection | Syntax Literal | Java Equivalent | Mutability | Ordering | Duplicates Allowed? |
|---|---|---|---|---|---|
| **`list`** | `[1, 2, 3]` | `ArrayList<Object>` | **Mutable** (Can add/remove/change) | Ordered (Indexed 0, 1, 2...) | **Yes** |
| **`dict`** | `{"key": "val"}` | `LinkedHashMap<K, V>` | **Mutable** | Ordered (Preserves insertion order since 3.7) | Keys: **No**, Values: **Yes** |
| **`tuple`** | `(1, 2, 3)` | Immutable Record / Array `Object[]` | **Immutable** (Cannot modify once created) | Ordered (Indexed 0, 1, 2...) | **Yes** |
| **`set`** | `{1, 2, 3}` | `HashSet<E>` | **Mutable** | Unordered (Hash-based) | **No** (Strictly unique) |

```
                       ┌──────────────────────────┐
                       │   PYTHON COLLECTIONS     │
                       └─────────────┬────────────┘
                                     │
         ┌───────────────────┬───────┴───────────┬────────────────────┐
         ▼                   ▼                   ▼                    ▼
     [ list ]            { dict }            ( tuple )             { set }
  Mutable Array       Key-Value Map       Immutable Array       Unique Elements
  Java: ArrayList     Java: HashMap       Java: Record / Tuple  Java: HashSet
```

### 1.2 Quick Syntax Comparison

```python
# 1. List (dynamic array)
model_names = ["gpt-4o", "claude-3-5-sonnet", "llama-3-8b"]
model_names.append("mistral-7b")

# 2. Dictionary (key-value hash map)
token_counts = {"prompt": 142, "completion": 65}
token_counts["total"] = token_counts["prompt"] + token_counts["completion"]

# 3. Tuple (immutable fixed sequence)
embedding_shape = (1536, "float32")
# embedding_shape[0] = 768  # ❌ TypeError! Tuples cannot be modified

# 4. Set (unique distinct elements)
stop_words = {"the", "is", "at", "which", "on", "the"}  # duplicate "the" discarded automatically
```

```java
// Equivalent Java Setup
// 1. List
List<String> modelNames = new ArrayList<>(List.of("gpt-4o", "claude-3-5-sonnet", "llama-3-8b"));
modelNames.add("mistral-7b");

// 2. Map
Map<String, Integer> tokenCounts = new LinkedHashMap<>();
tokenCounts.put("prompt", 142);
tokenCounts.put("completion", 65);
tokenCounts.put("total", tokenCounts.get("prompt") + tokenCounts.get("completion"));

// 3. Tuple / Record
record EmbeddingShape(int dimension, String dtype) {}
EmbeddingShape embeddingShape = new EmbeddingShape(1536, "float32");

// 4. Set
Set<String> stopWords = new HashSet<>(Set.of("the", "is", "at", "which", "on"));
```

---

## 2. Building Up – Concepts Added One by One

### 2.1 Python `list` vs. Java `ArrayList`

#### Key Characteristics
- **Dynamic Resizing**: Just like Java's `ArrayList`, Python's `list` grows and shrinks dynamically.
- **Heterogeneous Elements**: While Java collections enforce generic types (e.g., `ArrayList<String>`), Python lists can hold mixed types simultaneously (e.g., `["gpt-4", 8192, 0.7, True]`).
- **Negative Indexing**: Access elements from the end using negative integers (`[-1]` is the last element, `[-2]` is second to last). Java requires `list.get(list.size() - 1)`.

#### Slicing: `list[start:stop:step]`
Python has built-in slicing syntax that extracts sub-lists without looping. Java requires `list.subList(from, to)`.
- `start`: Inclusive starting index (defaults to `0`).
- `stop`: Exclusive ending index (defaults to end of list).
- `step`: Interval stride (defaults to `1`, negative steps reverse the list).

```python
tokens = ["User:", "Summarize", "this", "quarterly", "financial", "earnings", "report"]

# 1. Basic indexing (Positive & Negative)
first_token = tokens[0]      # "User:"
last_token = tokens[-1]      # "report" (Java: tokens.get(tokens.size() - 1))

# 2. Slicing [start:stop] -> stop index is EXCLUSIVE
head = tokens[0:3]           # ['User:', 'Summarize', 'this']
first_four = tokens[:4]      # ['User:', 'Summarize', 'this', 'quarterly']
tail = tokens[3:]            # ['quarterly', 'financial', 'earnings', 'report']

# 3. Step stride [start:stop:step]
every_other = tokens[::2]    # ['User:', 'this', 'financial', 'report']
reversed_tokens = tokens[::-1] # Reverses the entire list!
```

#### Common Methods & Java Equivalents

| Operation | Python Method | Java Equivalent | Time Complexity |
|---|---|---|---|
| Append to end | `lst.append(x)` | `list.add(x)` | $O(1)$ amortized |
| Insert at index | `lst.insert(0, x)` | `list.add(0, x)` | $O(n)$ |
| Remove by value | `lst.remove(x)` | `list.remove(x)` | $O(n)$ |
| Pop and return | `lst.pop()` / `lst.pop(i)` | `list.remove(list.size() - 1)` | $O(1)$ end / $O(n)$ index |
| Length | `len(lst)` | `list.size()` | $O(1)$ |
| Check existence | `"item" in lst` | `list.contains("item")` | $O(n)$ |
| Sort in-place | `lst.sort()` | `Collections.sort(list)` | $O(n \log n)$ (Timsort) |
| Return new sorted | `sorted(lst)` | `list.stream().sorted().toList()` | $O(n \log n)$ |

---

### 2.2 Python `dict` vs. Java `HashMap`

A Python `dict` stores key-value associations. Since Python 3.7, standard dictionaries are **guaranteed to preserve insertion order** (similar to Java's `LinkedHashMap`).

#### Safe Access with `.get()`
In Java:
```java
// If key is missing, map.get("temp") returns null
Double temp = config.getOrDefault("temperature", 0.7);
```

In Python:
```python
model_config = {
    "model": "gpt-4o",
    "max_tokens": 2048
}

# Direct key lookup throws KeyError if missing:
# temp = model_config["temperature"]  # ❌ KeyError: 'temperature'

# Safe lookup with default fallback:
temperature = model_config.get("temperature", 0.7)  # Returns 0.7 without error
model_name = model_config.get("model", "default-model")  # Returns "gpt-4o"
```

#### Iterating Dictionaries
```python
rag_metrics = {
    "precision": 0.94,
    "recall": 0.88,
    "latency_ms": 145.2
}

# 1. Iterate keys
for metric in rag_metrics:
    print(metric)

# 2. Iterate key-value pairs (like Java's Map.Entry)
for key, value in rag_metrics.items():
    print(f"{key} -> {value}")

# 3. Check membership (O(1) average lookup)
if "latency_ms" in rag_metrics:
    print(f"Latency recorded: {rag_metrics['latency_ms']} ms")
```

---

### 2.3 Python `tuple` vs. Java Records / Immutable Arrays

A `tuple` is an ordered, **immutable** sequence defined with parentheses `()`.

#### Why Tuples Matter
1. **Safety**: Once created, elements cannot be added, removed, or reassigned.
2. **Hashable**: Because tuples are immutable, they can be used as keys in dictionaries or elements in sets (lists cannot!).
3. **Multiple Return Values**: Python functions frequently return tuples to return multiple values cleanly.

```python
# Function returning multiple values via tuple
def get_model_specs(model_id: str):
    # Returns (context_window, cost_per_1k, supports_vision)
    return 128000, 0.005, True

# Tuple unpacking (destructuring)
context_window, cost, supports_vision = get_model_specs("gpt-4o")

print(f"Context: {context_window} tokens | Vision: {supports_vision}")
# Output: Context: 128000 tokens | Vision: True

# Tuple as a dictionary key (e.g., coordinates or composite keys)
# In Java, you would need a custom class with equals() and hashCode()
cache = {}
prompt_version = ("rag_v2", "english")
cache[prompt_version] = "System prompt text content..."
print(cache[("rag_v2", "english")])
```

---

### 2.4 Python `set` vs. Java `HashSet`

A `set` stores distinct, unordered, hashable items. Duplicate additions are silently ignored. Lookups (`x in my_set`) take $O(1)$ time.

#### Set Operations (Mathematical Operators)
Python sets feature dedicated operators for common mathematical set operations:

```python
vocabulary_a = {"token", "attention", "embedding", "transformer", "encoder"}
vocabulary_b = {"transformer", "decoder", "latent", "attention", "diffusion"}

# 1. Union (all unique items across both sets): |
all_tokens = vocabulary_a | vocabulary_b
# Java: Set<String> u = new HashSet<>(a); u.addAll(b);

# 2. Intersection (items present in BOTH sets): &
common_tokens = vocabulary_a & vocabulary_b
# Java: Set<String> i = new HashSet<>(a); i.retainAll(b);

# 3. Difference (items in A but NOT in B): -
unique_to_a = vocabulary_a - vocabulary_b
# Java: Set<String> d = new HashSet<>(a); d.removeAll(b);

# 4. Symmetric Difference (items in A or B, but NOT in both): ^
exclusive_tokens = vocabulary_a ^ vocabulary_b

print("Intersection:", common_tokens)
# Output: Intersection: {'attention', 'transformer'}
```

---

### 2.5 Comprehensions vs. Java Streams API

One of Python's most beloved idioms is **comprehensions**. Instead of writing imperative `for` loops or verbose Java Streams chains, comprehensions build transformed collections in a single, readable line.

#### Formula
```python
[ <expression> for <item> in <iterable> if <condition> ]
```

#### Comparison: Filtering and Transforming Words
Suppose we want to take a list of query terms, filter out those shorter than 4 characters, and convert them to uppercase.

**Java Streams Implementation:**
```java
List<String> rawTokens = List.of("ai", "neural", "llm", "transformer", "rag");

List<String> processed = rawTokens.stream()
    .filter(t -> t.length() > 3)
    .map(String::toUpperCase)
    .collect(Collectors.toList());
// Result: ["NEURAL", "TRANSFORMER"]
```

**Python List Comprehension:**
```python
raw_tokens = ["ai", "neural", "llm", "transformer", "rag"]

# Concise, elegant, native
processed = [t.upper() for t in raw_tokens if len(t) > 3]
print(processed)
# Output: ['NEURAL', 'TRANSFORMER']
```

#### Dictionary Comprehensions
Transforming or building a dictionary in a single line:

```python
models = ["llama-3-8b", "llama-3-70b", "mistral-7b"]

# Map model name to its character length
name_lengths = {model: len(model) for model in models}
print(name_lengths)
# Output: {'llama-3-8b': 10, 'llama-3-70b': 11, 'mistral-7b': 10}

# Invert a dictionary (swap keys and values)
status_codes = {"OK": 200, "UNAUTHORIZED": 401, "SERVER_ERROR": 500}
code_to_status = {code: name for name, code in status_codes.items()}
print(code_to_status[200])
# Output: OK
```

---

## 3. Practice Exercises (Easy to Hard)

### Exercise 1: Extract Head, Middle, and Tail (Easy)
**Task**: Given a list of 8 generated token IDs, extract:
1. The first 2 tokens (Prompt prefix)
2. The last 2 tokens (Stop sequence)
3. The remaining middle 4 tokens (Generated payload)
Use list slicing.

```python
# Input
token_ids = [101, 2054, 2003, 1037, 3231, 1012, 102, 0]

# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
token_ids = [101, 2054, 2003, 1037, 3231, 1012, 102, 0]

prefix = token_ids[:2]
suffix = token_ids[-2:]
middle = token_ids[2:-2]

print("Prefix:", prefix)
print("Middle:", middle)
print("Suffix:", suffix)

# Expected Output:
# Prefix: [101, 2054]
# Middle: [2003, 1037, 3231, 1012]
# Suffix: [102, 0]
```
</details>

---

### Exercise 2: Deduplicating Document Chunks (Easy)
**Task**: In a Retrieval-Augmented Generation (RAG) system, multiple search queries returned identical document chunk IDs. Write a function that takes a list of IDs and returns the count of unique chunks and a sorted list of unique IDs.

```python
chunk_ids = ["doc_42", "doc_07", "doc_15", "doc_42", "doc_99", "doc_07", "doc_01"]

# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
chunk_ids = ["doc_42", "doc_07", "doc_15", "doc_42", "doc_99", "doc_07", "doc_01"]

# 1. Convert to set for O(1) deduplication
unique_set = set(chunk_ids)

# 2. Convert back to list and sort
sorted_unique = sorted(unique_set)

print(f"Total Unique: {len(sorted_unique)}")
print("Sorted Chunks:", sorted_unique)

# Expected Output:
# Total Unique: 5
# Sorted Chunks: ['doc_01', 'doc_07', 'doc_15', 'doc_42', 'doc_99']
```
</details>

---

### Exercise 3: Prompt Template Parameter Substitution (Medium)
**Task**: Given a prompt template dictionary with missing keys and default configurations, create a merged configuration where user overrides take precedence over defaults without mutating the original dictionary.

```python
default_params = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.9,
    "stream": False
}

user_overrides = {
    "temperature": 0.2,
    "stream": True
}

# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
default_params = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.9,
    "stream": False
}

user_overrides = {
    "temperature": 0.2,
    "stream": True
}

# Python 3.9+ Dictionary Merge Operator (|)
# Right-hand side overrides left-hand side
merged_config = default_params | user_overrides

print("Merged Config:", merged_config)

# Alternative (Pre-Python 3.9 dictionary unpacking):
# merged_config = {**default_params, **user_overrides}

# Expected Output:
# Merged Config: {'temperature': 0.2, 'max_tokens': 1024, 'top_p': 0.9, 'stream': True}
```
</details>

---

### Exercise 4: Filtering and Normalizing Vector Scores with Comprehensions (Medium)
**Task**: A vector database returns a list of candidate documents with their cosine similarity scores: `[("doc_a", 0.89), ("doc_b", 0.62), ("doc_c", 0.95), ("doc_d", 0.45)]`. 
Using a single dictionary comprehension, filter out any document with a score below `0.70`, and round the remaining scores to 1 decimal place.

```python
candidates = [("doc_a", 0.89), ("doc_b", 0.62), ("doc_c", 0.95), ("doc_d", 0.45)]

# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
candidates = [("doc_a", 0.89), ("doc_b", 0.62), ("doc_c", 0.95), ("doc_d", 0.45)]

# Dictionary comprehension with tuple unpacking and filtering
filtered_docs = {doc_id: round(score, 1) for doc_id, score in candidates if score >= 0.70}

print("Qualified Documents:", filtered_docs)

# Expected Output:
# Qualified Documents: {'doc_a': 0.9, 'doc_c': 1.0}
```
</details>

---

### Exercise 5: Batching Batches for LLM Inference (Hard)
**Task**: LLM API endpoints require requests to be grouped into batches (e.g., maximum batch size of 3). Write a reusable generator or function `create_batches(prompts, batch_size)` that slices any input list into chunks of size `batch_size`.

```python
prompts = [
    "Explain transformers",
    "Define attention mechanism",
    "What is gradient descent?",
    "Explain RLHF",
    "Compare BERT and GPT",
    "What is temperature sampling?",
    "Explain LoRA fine-tuning"
]

# --- Write your solution here ---
```

<details>
<summary>👉 View Solution</summary>

```python
def create_batches(items: list, batch_size: int) -> list:
    """Chunks an input list into sub-lists of length batch_size using range slicing."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

prompts = [
    "Explain transformers",
    "Define attention mechanism",
    "What is gradient descent?",
    "Explain RLHF",
    "Compare BERT and GPT",
    "What is temperature sampling?",
    "Explain LoRA fine-tuning"
]

batches = create_batches(prompts, batch_size=3)

for idx, batch in enumerate(batches, start=1):
    print(f"Batch {idx} (size {len(batch)}): {batch}")

# Expected Output:
# Batch 1 (size 3): ['Explain transformers', 'Define attention mechanism', 'What is gradient descent?']
# Batch 2 (size 3): ['Explain RLHF', 'Compare BERT and GPT', 'What is temperature sampling?']
# Batch 3 (size 1): ['Explain LoRA fine-tuning']
```
</details>

---

## 4. Pro Level – Internals, Edge Cases & Interview Q&A

### 4.1 CPython Internals: How `list` Works Under the Hood
In Java, an `ArrayList<Object>` is a contiguous array of references (`Object[] elementData`). When it fills up, it allocates a new array of size `oldCapacity + (oldCapacity >> 1)` (1.5x) and copies elements via `System.arraycopy()`.

CPython implements `list` similarly as a dynamic array of pointers (`PyObject** ob_item`):
- **Over-allocation Formula**: CPython uses an over-allocation factor of approximately $1.125 \times$ plus 3 to 6 slots. This guarantees $O(1)$ amortized append operations while minimizing memory waste.
- **Cache Locality**: A Python `list` stores pointers (`PyObject*`), not contiguous raw bytes. While sequential access is fast, it incurs a pointer dereference for every element access, unlike a C primitive array or a NumPy `ndarray`.

```
CPython PyListObject:
┌──────────────┬──────────────┬───────────────────────────────┐
│  ob_refcnt   │   ob_type    │  ob_size = 3 | allocated = 6  │
└──────────────┴──────────────┴───────┬───────────────────────┘
                                      │ ob_item (array of pointers)
                     ┌────────────────┴───────────────┐
                     ▼                                ▼
              [ Ptr 0 | Ptr 1 | Ptr 2 | NULL | NULL | NULL ]
                 │       │       │
                 ▼       ▼       ▼
              "gpt-4"   2048    True
```

### 4.2 CPython `dict` Internals: Compact Hash Tables
Prior to Python 3.6, Python dictionaries used a sparse hash table array where each row stored `hash, key_ptr, value_ptr`. This wasted up to 60% of memory in empty hash buckets.

Since Python 3.6+ (standardized in Python 3.7), PyDict uses a **two-table compact representation**:
1. **Indices Array**: A sparse array of small integer indices (e.g., `[-1, 0, -1, 1, 2]`).
2. **Entries Array**: A dense, contiguous array storing entries in exact insertion order:
   `[ {"hash": ..., "key": "k1", "value": "v1"}, {"hash": ..., "key": "k2", "value": "v2"} ]`.

**Consequences:**
- Insertion order is strictly preserved with zero runtime overhead.
- Memory usage decreased by 20% to 25%.
- Iterating over keys or values is a continuous cache-friendly sweep across the dense entries array.

### 4.3 Shallow Copy vs. Deep Copy Pitfall
In Java, modifying a sub-object inside a shallow-copied list mutates the original reference. The exact same hazard exists in Python:

```python
import copy

original_pipeline = [
    {"stage": "retrieval", "top_k": 5},
    {"stage": "rerank", "top_k": 3}
]

# Shallow copy (slice [:] or .copy())
shallow = original_pipeline.copy()
shallow[0]["top_k"] = 10  # ⚠️ Mutates the dictionary INSIDE original_pipeline!

print(original_pipeline[0]["top_k"])  # Prints 10! (Bug!)

# Deep copy (completely independent object tree)
deep = copy.deepcopy(original_pipeline)
deep[0]["top_k"] = 99

print(original_pipeline[0]["top_k"])  # Still 10 (Protected)
print(deep[0]["top_k"])               # 99
```

### 4.4 Top Technical Interview Questions & Answers

#### Q1: What makes an object "hashable" in Python, and why can't a `list` be used as a dictionary key?
**Answer**: An object is hashable if it has a hash value that never changes during its lifetime (implements `__hash__()`) and can be compared to other objects for equality (implements `__eq__()`). 
Mutable containers like `list`, `set`, and `dict` are unhashable because their contents can change after insertion; if mutated, their hash would change, making them impossible to find in a hash bucket. An immutable `tuple` containing only hashable elements is hashable and can safely serve as a dictionary key.

#### Q2: What is the difference between `==` and `is` when comparing collections?
**Answer**:
- `==` checks **value equality** (similar to Java's `obj1.equals(obj2)`). It compares whether two collections contain identical elements in identical order.
- `is` checks **reference identity** (identical to Java's `obj1 == obj2`). It returns `True` only if both variables point to the exact same memory address (`id(a) == id(b)`).

```python
list_a = [1, 2, 3]
list_b = [1, 2, 3]

print(list_a == list_b)  # True (Same values)
print(list_a is list_b)  # False (Different memory locations)
```

---

## 5. Quick Revision (Cheat-Sheet)

### 5.1 Collection Syntax Comparison: Java vs. Python

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ JAVA                            │ PYTHON EQUIVALENT                         │
├─────────────────────────────────┼───────────────────────────────────────────┤
│ List<String> l = new ArrayList<>()│ l = []                                  │
│ l.add("item");                  │ l.append("item")                          │
│ l.get(0);                       │ l[0]                                      │
│ l.get(l.size() - 1);            │ l[-1]                                     │
│ l.subList(0, 3);                │ l[:3]                                     │
│ Collections.reverse(l);         │ l.reverse()  OR  l[::-1]                  │
├─────────────────────────────────┼───────────────────────────────────────────┤
│ Map<String, Object> m = new Map()│ m = {}                                   │
│ m.put("k", "v");                │ m["k"] = "v"                              │
│ m.getOrDefault("k", "default"); │ m.get("k", "default")                     │
│ m.containsKey("k");             │ "k" in m                                  │
│ m.keySet();                     │ m.keys()                                  │
│ m.entrySet();                   │ m.items()                                 │
├─────────────────────────────────┼───────────────────────────────────────────┤
│ Set<String> s = new HashSet<>();│ s = set()                                 │
│ s.add("item");                  │ s.add("item")                             │
│ s.contains("item");             │ "item" in s                               │
│ s1.addAll(s2);                  │ s1 | s2                                   │
│ s1.retainAll(s2);               │ s1 & s2                                   │
├─────────────────────────────────┼───────────────────────────────────────────┤
│ stream().filter().map().toList()│ [f(x) for x in items if condition(x)]     │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 5.2 Key Takeaways for Java Engineers
1. **Literal Syntax**: Avoid verbose constructors; use `[]` for lists, `{}` for dicts, `()` for tuples, and `{1, 2}` for sets.
2. **Slicing**: Remember `items[start:stop]` excludes `stop`. Negative indices count backward from the end.
3. **Safe Lookups**: Prefer `dict.get(key, default)` over direct indexing `dict[key]` to avoid unhandled `KeyError` exceptions.
4. **Comprehensions**: Replace multi-line loops and Java Streams with list and dict comprehensions for idiomatic, readable code.
5. **Tuples for Immutability**: Use tuples when a collection must remain unmutated or be used as a dictionary key.

---

## 6. References & Recommended Videos

To reinforce your understanding of Python collections and internals, search for these authoritative tutorials:

| Topic | Channel / Creator | Exact Search Phrase | Why Watch |
|---|---|---|---|
| **Lists, Tuples, and Sets** | Corey Schafer | `Corey Schafer Python Lists, Tuples, and Sets` | The gold standard beginner-to-intermediate walkthrough of Python's sequential data structures. |
| **Dictionaries & Key-Value Operations** | Corey Schafer | `Corey Schafer Python Dictionaries: Working with Key-Value Pairs` | Clear breakdown of dictionary methods, `.get()`, key deletion, and iteration techniques. |
| **List Comprehensions** | mCoding | `mCoding List Comprehensions in Python Are Awesome` | Concise, practical demonstration of comprehensions, nested comprehensions, and performance comparisons. |
| **Dictionary Internals & Memory** | PyCon / Raymond Hettinger | `Raymond Hettinger Modern Python Dictionaries A confluence of a dozen great ideas` | Deep dive by Python core developer Raymond Hettinger explaining the compact dict architecture and hash internals. |
| **Memory & Deep Copy Gotchas** | ArjanCodes | `ArjanCodes Stop using copy in Python unless you know this` | Explains shallow vs. deep copy risks in real-world object architectures. |
