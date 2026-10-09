# ⚙️ Development Workflow: Managing Dependencies, Version Control with Git & GitHub, and Building Front-End Interfaces with Streamlit

> **Zero to Hero Gen AI Course — Module 06: End-to-End Development & MLOps**
>
> 📅 **Module 6: End-to-End Development & MLOps** | ⏱️ **Estimated Reading Time:** 80 minutes | 🎯 **Level:** Intermediate to Advanced
>
> **Core Objective:** Master the engineering discipline required to transition Generative AI experiments from fragile local Jupyter notebooks into robust, collaborative, and deployable production software. Deconstruct modern Python dependency management (`venv`, `poetry`, `pyproject.toml`, pinned lockfiles, and CUDA/PyTorch wheel caching). Implement professional Git and GitHub practices for AI projects: repository hygiene, Git LFS for neural weights, conventional commits, pre-commit hooks (Ruff, Black, Gitleaks for API key shielding), and automated GitHub Actions CI/CD workflows. Architect reactive conversational user interfaces using **Streamlit**: master the reactive script re-run execution model, persistent multi-turn chat via `st.session_state`, real-time token streaming via `st.chat_message`, memory caching (`st.cache_resource`), and production secrets management.

---

## 📑 Comprehensive Syllabus & Table of Contents

- [Part 1: Core Concept & Architecture Overview 🌟 🐣 💡](#part-1-core-concept--architecture-overview----)
  - [1.1 The Software Engineering Chasm in AI: From Notebook to Production](#11-the-software-engineering-chasm-in-ai-from-notebook-to-production)
  - [1.2 The Three Pillars of Engineering Hygiene](#12-the-three-pillars-of-engineering-hygiene)
  - [1.3 Intuitive Mental Models & Analogies](#13-intuitive-mental-models--analogies)
  - [1.4 Modern Python Dependency Management: venv, Poetry & pyproject.toml](#14-modern-python-dependency-management-venv-poetry--pyprojecttoml)
  - [1.5 Professional Version Control: Git LFS, Pre-Commit Hooks & GitHub Actions](#15-professional-version-control-git-lfs-pre-commit-hooks--github-actions)
  - [1.6 Rapid Front-End Prototyping with Streamlit: The Reactive Re-Run Engine](#16-rapid-front-end-prototyping-with-streamlit-the-reactive-re-run-engine)
  - [1.7 End-to-End Development Workflow Visualized](#17-end-to-end-development-workflow-visualized)
- [Part 2: Mathematical Foundations & Algorithms 🧱](#part-2-mathematical-foundations--algorithms-)
  - [2.1 Reactive Directed Acyclic Graph (DAG) State Re-evaluation Complexity](#21-reactive-directed-acyclic-graph-dag-state-re-evaluation-complexity)
  - [2.2 Amdahl's Law and Cache Latency Speedup Ratio](#22-amdahls-law-and-cache-latency-speedup-ratio)
  - [2.3 Content-Addressable Storage (CAS) Merkle Tree Mathematics in Git](#23-content-addressable-storage-cas-merkle-tree-mathematics-in-git)
  - [2.4 Git LFS Pointer Storage Economics: O(1) vs O(Data) Cloning](#24-git-lfs-pointer-storage-economics-o1-vs-odata-cloning)
- [Part 3: Java & Spring Boot Developer Bridge ☕](#part-3-java--spring-boot-developer-bridge-)
  - [3.1 Conceptual Mapping: Python MLOps vs Spring Boot / JVM Ecosystem](#31-conceptual-mapping-python-mlops-vs-spring-boot--jvm-ecosystem)
  - [3.2 Dependency & Build Systems: Maven/Gradle vs Poetry/pyproject.toml](#32-dependency--build-systems-mavengradle-vs-poetrypyprojecttoml)
  - [3.3 Streamlit Reactive Model vs Spring MVC, Vaadin & Thymeleaf](#33-streamlit-reactive-model-vs-spring-mvc-vaadin--thymeleaf)
  - [3.4 State & Cache Management: Spring @SessionScope & @Cacheable vs Streamlit](#34-state--cache-management-spring-sessionscope--cacheable-vs-streamlit)
- [Part 4: Hands-On Implementation & Practice Exercises 🧪](#part-4-hands-on-implementation--practice-exercises-)
  - [Exercise 1 (Beginner): Deterministic Dependency Manifest & Lockfile Validator](#exercise-1-beginner-deterministic-dependency-manifest--lockfile-validator)
  - [Exercise 2 (Intermediate): Pure-Python Git Merkle Tree & Content Hasher](#exercise-2-intermediate-pure-python-git-merkle-tree--content-hasher)
  - [Exercise 3 (Advanced): Reactive Streamlit State Machine & Multi-Turn Chat Simulator](#exercise-3-advanced-reactive-streamlit-state-machine--multi-turn-chat-simulator)
  - [Exercise 4 (Expert): Production Two-Tier Caching Decorator with LRU & Singleton Management](#exercise-4-expert-production-two-tier-caching-decorator-with-lru--singleton-management)
- [Part 5: Production Engineering, Edge Cases & Failure Modes ⚙️ ⚡](#part-5-production-engineering-edge-cases--failure-modes-️-)
  - [5.1 Leaking API Keys to Public GitHub Repositories (The GitLeaks Defense)](#51-leaking-api-keys-to-public-github-repositories-the-gitleaks-defense)
  - [5.2 The Re-Run Trap: Expensive Model Reloads on Every Widget Click](#52-the-re-run-trap-expensive-model-reloads-on-every-widget-click)
  - [5.3 Streamlit Concurrency Limitations: Single-Process GIL vs FastAPI Decoupling](#53-streamlit-concurrency-limitations-single-process-gil-vs-fastapi-decoupling)
  - [5.4 Secrets Management: .streamlit/secrets.toml vs Environment Variables](#54-secrets-management-streamlitsecretstoml-vs-environment-variables)
  - [5.5 Enterprise Case Study: Building an Internal Legal Contract Review Copilot](#55-enterprise-case-study-building-an-internal-legal-contract-review-copilot)
- [Part 6: Video Masterclasses, Lab Suites & Review Questions 🎬](#part-6-video-masterclasses-lab-suites--review-questions-)
  - [6.1 Telugu Tech Masterclasses & Global Visual 3D Animations](#61-telugu-tech-masterclasses--global-visual-3d-animations)
  - [6.2 Complete Hands-On Lab Walkthrough](#62-complete-hands-on-lab-walkthrough)
  - [6.3 Comprehensive Self-Assessment & Review Questions](#63-comprehensive-self-assessment--review-questions)
  - [6.4 Key Takeaways & Architectural Checklist](#64-key-takeaways--architectural-checklist)

---

## Part 1: Core Concept & Architecture Overview 🌟 🐣 💡

### 1.1 The Software Engineering Chasm in AI: From Notebook to Production

In exploratory research and quick hacks, machine learning practitioners spend most of their time in **Jupyter Notebooks**. Notebooks excel at interactive data exploration, chart plotting, and testing isolated prompts.

However, moving an AI system from a notebook into an enterprise software product creates severe failure modes:
- **Hidden Execution State:** Notebook cells can run out of order. A variable assigned in Cell 14 and mutated in Cell 3 creates an invisible, un-reproducible Python runtime state that fails the moment the kernel restarts.
- **Unpinned Transitive Dependencies:** Running `pip install langchain` today installs different sub-dependencies than it did six months ago. Without exact lockfiles, deployment pods crash with broken imports or deprecated function arguments.
- **Leaked API Keys:** Developers accidentally commit OpenAI, Anthropic, or Hugging Face API keys directly into notebook outputs or git history, leading to automated bot scraping and thousands of dollars in unauthorized cloud bills within minutes.

```
+---------------------------------------------------------------------------------------------------+
|                                 THE PROTOTYPE vs PRODUCTION CHASM                                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   FRAGILE RESEARCH PROTOTYPE (Jupyter):              ENTERPRISE AI SOFTWARE PRODUCT:              |
|   - Out-of-order cell execution state.               - Modular, tested Python package architecture|
|   - Global pip environment (dependency collisions).  - Isolated venv / Poetry with pinned lockfile|
|   - Hardcoded API keys in plaintext cells.           - Centralized secrets management (.env)       |
|   - Untracked 5 GB model weights in git.             - Git LFS / Hugging Face model registries     |
|   - No automated tests or linting checks.            - Pre-commit hooks & GitHub Actions CI/CD     |
|   - Raw print() statements for output.               - Interactive, reactive Streamlit web UI     |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

### 1.2 The Three Pillars of Engineering Hygiene

To cross this chasm, professional Generative AI engineers rely on **Three Structural Pillars**:
1. **Dependency & Environment Hygiene:** Strict virtual environment isolation, deterministic dependency resolution, and locked package versions (`pyproject.toml` + `poetry.lock`).
2. **Version Control & Collaboration:** Structured Git workflows, branch protection rules, automated pre-commit scanning (Gitleaks, Ruff), Git LFS for neural weights, and CI/CD validation gates.
3. **Interactive User Interfaces:** Rapid, reactive front-end development using **Streamlit**, enabling business stakeholders to test and validate AI models through a polished web interface without writing React or JavaScript.

---

### 1.3 Intuitive Mental Models & Analogies

```
+---------------------------------------------------------------------------------------------------+
|                                  DEVELOPMENT WORKFLOW ANALOGIES                                   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. THE CLEANROOM vs CONTAMINATED WORKSHOP     2. THE TIME-TRAVELING TREE OF HISTORY              |
|                                                                                                   |
|      Global Python Environment:                     Git Version Control:                          |
|      * A messy open workshop where woodworking,     * A magical branching tree where every        |
|        spray painting, and chemistry occur on         atomic change is an indelible snapshot.    |
|        the same table. Wood shavings contaminate    * If a branch catches fire, you can prune     |
|        the beaker!                                    it and step back to a pristine timeline.    |
|                                                                                                   |
|      Isolated Virtual Environment (venv):          3. THE SELF-UPDATING CHALKBOARD                |
|      * A sterile laboratory cleanroom with an       * Streamlit does not require manual event     |
|        airlock. Only the exact chemicals required     listeners. When a user moves a slider,      |
|        for this specific experiment are admitted.     a robot instantly wipes the board clean     |
|        Zero cross-contamination.                      and re-executes the script top-to-bottom!   |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

- **The Laboratory Cleanroom (Virtual Environments) vs The Contaminated Workshop:** Imagine a chemist conducting a delicate pharmaceutical synthesis on a workbench covered in sawdust from a previous carpentry project and motor oil from a motorcycle repair. Installing conflicting packages (e.g. PyTorch 1.13 for an old project and PyTorch 2.4 for a new LLM) globally leads to catastrophic library collisions. A virtual environment is a cleanroom built specifically for one experiment.
- **The Time-Traveling Tree of History (Git & GitHub):** Without Git, developers save files named `bot_v1.py`, `bot_v2_final.py`, `bot_v2_final_FINAL_edit.py`. With Git, every commit is an atomic logical mutation with a cryptographic signature. Feature branches allow teams to build vector search, UI widgets, and prompt templates simultaneously without colliding.
- **The Self-Updating Chalkboard (The Streamlit Reactive Rerun Loop):** In traditional web development, building an interface requires HTML, CSS, JavaScript event listeners, REST APIs, and client-server state synchronization. Streamlit reimagines this as a self-updating chalkboard: you write pure Python sequentially, and whenever the user clicks a widget, Streamlit wipes the chalkboard clean and re-runs the script from top to bottom!

---

### 1.4 Modern Python Dependency Management: venv, Poetry & pyproject.toml

What actually happens when you create and activate a virtual environment?

```bash
# 1. Create a virtual environment directory named .venv
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

#### Under-the-Hood Mechanics:
1. **Isolated Directory Creation:** Python creates `.venv` containing local copies (or symlinks) of `python.exe`, `pip`, and an empty `Lib/site-packages/`.
2. **`PATH` Prepending:** Activating modifies your shell environment variable:
   $$\text{PATH} = \text{C:\Users\...\.venv\Scripts}; \ \$ \text{PATH}_{\text{system}}$$
3. **Resolution Redirection:** Executing `python` or `pip install` resolves to the local `.venv` binary first, installing packages strictly into `.venv/Lib/site-packages/`.

#### Declarative Manifests: `requirements.txt` vs Modern `pyproject.toml` (PEP 518 / PEP 621)

```toml
[project]
name = "clinical-medical-chatbot"
version = "1.0.0"
description = "Enterprise Clinical Decision Support Chatbot with Hybrid RAG"
readme = "README.md"
requires-python = ">=3.10,<3.13"
dependencies = [
    "langchain>=0.2.14",
    "langchain-community>=0.2.12",
    "chromadb>=0.5.5",
    "pydantic>=2.8.2",
    "streamlit>=1.38.0",
    "requests>=2.32.3"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.2",
    "ruff>=0.6.1",
    "black>=24.8.0",
    "pre-commit>=3.8.0"
]
```

> [!IMPORTANT]
> **The Production Golden Rule:**
> Always commit your lockfile (`poetry.lock` or `requirements.lock`) to Git! When deploying to Docker or Kubernetes, install exclusively from the lockfile:
> ```bash
> pip install --no-deps -r requirements.lock
> ```
> This guarantees 100% byte-for-byte reproducibility across every developer workstation and deployment pod.

---

### 1.5 Professional Version Control: Git LFS, Pre-Commit Hooks & GitHub Actions

```
+---------------------------------------------------------------------------------------------------+
|                                 THE ESSENTIAL AI .gitignore CHECKLIST                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   1. Virtual Environments        -> .venv/, venv/, env/                                           |
|   2. Secret Keys & Environment   -> .env, *.env, .streamlit/secrets.toml                          |
|   3. Local Vector Databases      -> chroma_db/, .chroma/, faiss_index/, *.index                   |
|   4. Cached Models & Embeddings  -> models/, checkpoints/, *.bin, *.safetensors, *.gguf           |
|   5. Python Cache Directories    -> __pycache__/, *.pyc, .pytest_cache/, .ruff_cache/             |
|   6. OS Artifacts                -> .DS_Store, Thumbs.db, desktop.ini                             |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

#### Git LFS for Large Neural Weights:
Git blocks individual file pushes exceeding 100 MB. Git Large File Storage replaces multi-gigabyte models with lightweight text pointers in Git, storing the raw bytes on object storage:

```bash
git lfs install
git lfs track "*.safetensors"
git lfs track "*.gguf"
git lfs track "*.parquet"
git add .gitattributes
git commit -m "chore: track large model binaries with Git LFS"
```

---

### 1.6 Rapid Front-End Prototyping with Streamlit: The Reactive Re-Run Engine

Streamlit simplifies UI development by executing scripts as **reactive state machines**:

```
+---------------------------------------------------------------------------------------------------+
|                               THE STREAMLIT REACTIVE RE-RUN MACHINE                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   1. USER LOADS PAGE -> Python script executes from Line 1 to Line 100.                           |
|   2. WIDGET RENDERING -> UI renders buttons, text inputs, chat boxes.                             |
|   3. USER INTERACTION -> User types message in st.chat_input("Ask question").                     |
|   4. EVENT TRIGGER    -> Streamlit INTERRUPTS and RE-RUNS the entire script from Line 1!          |
|                                                                                                   |
|   HOW STATE PERSISTS:                                                                             |
|   - st.session_state stores multi-turn conversation messages across re-runs.                      |
|   - @st.cache_resource keeps heavy models (ChromaDB, LLM pipelines) loaded in memory as singletons!|
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

### 1.7 End-to-End Development Workflow Visualized

```
+---------------------------------------------------------------------------------------------------+
|                        MODERN GENERATIVE AI ENGINEERING DEVELOPMENT WORKFLOW                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Developer Workstation]                                                                          |
|        |                                                                                          |
|        +---> Environment: Isolated .venv / pyproject.toml + requirements.lock                     |
|        +---> Code Formatting: Ruff / Black                                                        |
|        +---> Secret Shielding: Pre-Commit Hooks (Gitleaks blocks API key leaks)                  |
|        |                                                                                          |
|        v [git commit -m "feat: ..."]                                                              |
|  [Git Version Control & Git LFS]                                                                  |
|        |                                                                                          |
|        +---> Source Code -> Git Commit Tree                                                      |
|        +---> Model Weights (*.safetensors, *.gguf) -> Git LFS Object Storage                     |
|        |                                                                                          |
|        v [git push origin feat/branch]                                                            |
|  [GitHub Repository & CI/CD Actions]                                                              |
|        |                                                                                          |
|        +---> Automated Unit Tests (pytest tests/)                                                 |
|        +---> Security Secret Scans & Ruff Linter                                                  |
|        +---> Pull Request Code Review Gate                                                        |
|        |                                                                                          |
|        v [Merge to main]                                                                          |
|  [Streamlit Reactive Web Application]                                                             |
|        |                                                                                          |
|        +---> UI Widgets: st.sidebar, st.selectbox, st.slider                                      |
|        +---> State Management: st.session_state (Chat history persistence)                        |
|        +---> Performance Caching: @st.cache_resource (Vector DB & LLM singletons)                 |
|        +---> Real-Time Output: st.write_stream (Dynamic token streaming)                          |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

#### Verified System Architecture Blueprint

![Development Workflow and Streamlit Architecture](assets/04_dev_workflow_git_streamlit_pipeline.jpg)

---

## Part 2: Mathematical Foundations & Algorithms 🧱

### 2.1 Reactive Directed Acyclic Graph (DAG) State Re-evaluation Complexity

Streamlit scripts represent an implicit **Directed Acyclic Graph (DAG)** of computational nodes $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
- $\mathcal{V}$: UI widgets, state variables, and computational functions.
- $\mathcal{E}$: Data dependencies connecting inputs to rendered outputs.

When a user triggers widget $v_k \in \mathcal{V}$:
The re-run engine performs a topological traversal over the dependency subgraph:

$$\text{Time Complexity} = O(|\mathcal{V}| + |\mathcal{E}|)$$

Without caching, if node $v_{\text{model}} \in \mathcal{V}$ performs heavy model weight loading ($T_{\text{load}} \approx 2.5\text{s}$), the total re-run latency is dominated by:

$$T_{\text{total}} = T_{\text{load}} + \sum_{v_i \in \text{Path}(v_k)} T_{\text{eval}}(v_i)$$

Wrapping $v_{\text{model}}$ in `@st.cache_resource` reduces $T_{\text{load}}$ from $2.5\text{s}$ to $O(1)$ memory pointer dereference ($<0.01\text{ms}$).

---

### 2.2 Amdahl's Law and Cache Latency Speedup Ratio

The latency speedup $S_{\text{latency}}$ obtained by caching expensive model and vector store initializations follows **Amdahl's Law**:

$$S(h) = \frac{1}{(1 - h) + \frac{h}{S_{\text{resource}}}}$$

Where:
- $h \in [0, 1]$: Cache hit rate across user interactions.
- $S_{\text{resource}} = \frac{T_{\text{cold}}}{T_{\text{warm}}}$: Speedup ratio of the cached component.

For an application where model initialization takes $T_{\text{cold}} = 3000\text{ms}$ and cached lookup takes $T_{\text{warm}} = 0.05\text{ms}$, $S_{\text{resource}} \approx 60,000$. With a cache hit rate of $h = 0.95$, the effective end-to-end interactive speedup is:

$$S(0.95) \approx \frac{1}{(1 - 0.95) + \frac{0.95}{60000}} = \frac{1}{0.05} = 20\times$$

---

### 2.3 Content-Addressable Storage (CAS) Merkle Tree Mathematics in Git

Git is mathematically a **Directed Acyclic Graph of Content-Addressable Objects** using cryptographic SHA-1 / SHA-256 hashes.

Every entity is hashed using a standardized payload prefix:

1. **Blob Object (File Content):**
   $$\text{Hash}_{\text{blob}} = \text{SHA-1}\left( \text{"blob "} + \text{len}(\text{content}) + \backslash 0 + \text{content} \right)$$
2. **Tree Object (Directory Listing):**
   $$\text{Hash}_{\text{tree}} = \text{SHA-1}\left( \text{"tree "} + \text{len}(\text{entries}) + \backslash 0 + \sum_{i} (\text{mode}_i + \text{" "} + \text{name}_i + \backslash 0 + \text{hash}_i) \right)$$
3. **Commit Object (Atomic Snapshot):**
   $$\text{Hash}_{\text{commit}} = \text{SHA-1}\left( \text{"commit "} + \text{len}(\dots) + \backslash 0 + \text{tree\_hash} + \text{parent\_hash} + \text{author} + \text{msg} \right)$$

Because hashes are content-addressable, identical file contents share identical hashes, enabling instant deduplication.

---

### 2.4 Git LFS Pointer Storage Economics: O(1) vs O(Data) Cloning

When committing a 4 GB model weight into standard Git:
- Every modification commits a full binary delta into `.git/objects/`.
- Repository cloning size grows with history:
  $$\text{Repo Size}_{\text{standard}} = \sum_{t=1}^T \text{Size}(\text{Model}_t) = O(T \cdot \text{Size})$$

With **Git LFS**, Git stores a tiny 130-byte pointer file:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:4d87b32a76ef48231c62981db8948194cf3519c7a6e709a321948ef1891bca72
size 4294967296
```

The Git repository size remains strictly **$O(1)$** in Git metadata space, downloading the 4 GB payload over HTTP only when explicitly checked out.

---

## Part 3: Java & Spring Boot Developer Bridge ☕

### 3.1 Conceptual Mapping: Python MLOps vs Spring Boot / JVM Ecosystem

| Python AI Development Pattern | Java / Spring Boot Equivalent | Architectural Difference |
| :--- | :--- | :--- |
| `venv` / `poetry` / `site-packages` | Maven `pom.xml` / Gradle `build.gradle.kts` + `~/.m2` | Maven isolates dependencies per artifact repository; Python traditionally isolates per project virtual environment directory. |
| `poetry.lock` / `requirements.lock` | Gradle `gradle.lockfile` / Maven Dependency Verification | Both produce cryptographic SHA-256 checksums to ensure reproducible build resolution. |
| Streamlit reactive UI | Vaadin (server-driven UI) / Thymeleaf / Spring MVC + React | Streamlit re-runs the full script on event triggers; Vaadin executes event listeners over a stateful WebSocket connection. |
| `st.session_state` | Spring `@SessionScope` bean / `HttpSession` | Spring scopes beans to HTTP sessions via servlet containers; Streamlit manages browser session state in Python memory. |
| `@st.cache_resource` | Spring `@Bean` (Default Singleton Scope) | Spring manages singletons in the ApplicationContext IoC container; Streamlit caches via function signature hashing. |
| `@st.cache_data` | Spring `@Cacheable` (with Caffeine / Redis) | Spring provides declarative cache eviction via annotations and cache managers. |

---

### 3.2 Dependency & Build Systems: Maven/Gradle vs Poetry/pyproject.toml

In Java, dependencies and build steps are declared in `pom.xml`:

```xml
<!-- Maven pom.xml Equivalent -->
<dependencies>
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
        <version>1.0.0-M1</version>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-validation</artifactId>
    </dependency>
</dependencies>
```

In Python, `pyproject.toml` handles dependencies, virtual environment configuration, and tool settings (Ruff, Pytest) in a single unified manifest.

---

### 3.3 Streamlit Reactive Model vs Spring MVC, Vaadin & Thymeleaf

- **Java Vaadin:** Vaadin is the closest JVM equivalent to Streamlit. You write pure Java UI components (`Button`, `Grid`, `TextField`), and Vaadin communicates with the browser over WebSockets. However, Vaadin uses standard event listeners:
  ```java
  // Java Vaadin Event-Driven Model
  Button sendBtn = new Button("Send", e -> {
      chatHistory.add(new Message(inputField.getValue()));
  });
  ```
- **Streamlit:** Completely eliminates event listener boilerplate. It re-runs the entire Python script sequentially, checking `if prompt := st.chat_input():` on each pass.

---

### 3.4 State & Cache Management: Spring @SessionScope & @Cacheable vs Streamlit

In Spring Boot, session persistence and model caching are managed via IoC annotations:

```java
// Spring Boot Stateful Session & Model Singleton
@Component
@SessionScope
public class UserConversationSession {
    private final List<ChatMessage> history = new ArrayList<>();
    // Persists across requests for this specific browser session
}

@Configuration
public class AiModelConfig {
    @Bean
    @Scope("singleton") // Analogous to @st.cache_resource
    public VectorStore vectorStore(EmbeddingModel embeddingModel) {
        return new SimpleVectorStore(embeddingModel);
    }
}
```

---

## Part 4: Hands-On Implementation & Practice Exercises 🧪

### Exercise 1 (Beginner): Deterministic Dependency Manifest & Lockfile Validator

Build a pure-Python dependency auditor that parses dependency declarations, validates semantic version constraints, and audits cryptographic SHA-256 checksums to detect tampered wheels.

```python
"""
Exercise 1: Deterministic Dependency Manifest & Lockfile Validator
Level: Beginner
Objective: Parse dependency ranges and audit cryptographic SHA-256 lockfile checksums.
"""
import hashlib
import re
from typing import Dict, List, Tuple

class DependencyValidator:
    def __init__(self):
        # Simulated verified package registry with cryptographic hashes
        self.registry = {
            "langchain": {"version": "0.2.14", "sha256": "4a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b"},
            "chromadb": {"version": "0.5.5", "sha256": "9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e"},
            "streamlit": {"version": "1.38.0", "sha256": "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"}
        }

    def verify_lockfile_integrity(self, lockfile_records: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
        """Audits package records against expected cryptographic checksums."""
        violations = []
        for pkg in lockfile_records:
            name = pkg.get("name")
            version = pkg.get("version")
            provided_hash = pkg.get("sha256")

            if name not in self.registry:
                violations.append(f"Unregistered package detected: {name}")
                continue

            expected = self.registry[name]
            if expected["version"] != version:
                violations.append(f"Version mismatch for {name}: expected {expected['version']}, got {version}")
            if expected["sha256"] != provided_hash:
                violations.append(f"TAMPER WARNING: SHA-256 checksum mismatch for {name}!")

        is_valid = len(violations) == 0
        return is_valid, violations

# Demonstration
if __name__ == "__main__":
    validator = DependencyValidator()
    
    mock_lockfile = [
        {"name": "langchain", "version": "0.2.14", "sha256": "4a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b"},
        {"name": "chromadb", "version": "0.5.5", "sha256": "CORRUPTED_TAMPERED_HASH_HERE"},
        {"name": "streamlit", "version": "1.38.0", "sha256": "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"}
    ]

    is_secure, issues = validator.verify_lockfile_integrity(mock_lockfile)
    print("=== DEPENDENCY INTEGRITY AUDIT ===")
    print(f"Lockfile Verified: {is_secure}")
    if not is_secure:
        print("Security Violations Flagged:")
        for issue in issues:
            print(f"  ❌ {issue}")
```

---

### Exercise 2 (Intermediate): Pure-Python Git Merkle Tree & Content Hasher

Demystify Git internals by implementing pure-Python SHA-1 Content-Addressable Storage (CAS) for Git Blobs, Tree directories, and Commit objects from scratch.

```python
"""
Exercise 2: Pure-Python Git Merkle Tree & Content Hasher
Level: Intermediate
Objective: Compute exact Git SHA-1 hashes for blobs, trees, and commits.
"""
import hashlib
from typing import List, Tuple

class GitObjectEngine:
    @staticmethod
    def hash_blob(content: str) -> Tuple[str, bytes]:
        """Calculates exact Git SHA-1 hash for a file blob: sha1('blob ' + size + '\0' + content)."""
        content_bytes = content.encode("utf-8")
        header = f"blob {len(content_bytes)}\0".encode("utf-8")
        payload = header + content_bytes
        sha1 = hashlib.sha1(payload).hexdigest()
        return sha1, payload

    @staticmethod
    def hash_tree(entries: List[Tuple[str, str, str]]) -> Tuple[str, bytes]:
        """
        Calculates Git Tree SHA-1 hash from entries: (mode, filename, sha1_hex).
        Format: 'tree ' + size + '\0' + [mode filename\0binary_sha1]...
        """
        body = bytearray()
        for mode, name, hex_sha1 in entries:
            body.extend(f"{mode} {name}\0".encode("utf-8"))
            body.extend(bytes.fromhex(hex_sha1))
            
        header = f"tree {len(body)}\0".encode("utf-8")
        payload = header + body
        return hashlib.sha1(payload).hexdigest(), payload

    @staticmethod
    def hash_commit(tree_sha: str, parent_sha: str, author: str, message: str) -> str:
        """Calculates Git Commit object hash."""
        body = (
            f"tree {tree_sha}\n"
            f"parent {parent_sha}\n"
            f"author {author} 1728468000 +0000\n"
            f"committer {author} 1728468000 +0000\n\n"
            f"{message}\n"
        ).encode("utf-8")
        header = f"commit {len(body)}\0".encode("utf-8")
        return hashlib.sha1(header + body).hexdigest()

# Demonstration
if __name__ == "__main__":
    engine = GitObjectEngine()
    
    # 1. Hash Python script file
    blob_sha, _ = engine.hash_blob("import streamlit as st\nst.title('AI App')")
    print(f"Git Blob SHA-1  : {blob_sha}")

    # 2. Hash Directory Tree
    tree_sha, _ = engine.hash_tree([("100644", "app.py", blob_sha)])
    print(f"Git Tree SHA-1  : {tree_sha}")

    # 3. Hash Commit
    commit_sha = engine.hash_commit(
        tree_sha=tree_sha,
        parent_sha="0000000000000000000000000000000000000000",
        author="Lead AI Engineer <engineer@enterprise.ai>",
        message="feat: initialize production Streamlit app"
    )
    print(f"Git Commit SHA-1: {commit_sha}")
```

---

### Exercise 3 (Advanced): Reactive Streamlit State Machine & Multi-Turn Chat Simulator

Simulate Streamlit's reactive re-run machine in pure Python, demonstrating how user widget interactions trigger full script re-runs while `session_state` preserves conversation memory and streams responses.

```python
"""
Exercise 3: Reactive Streamlit State Machine & Multi-Turn Chat Simulator
Level: Advanced
Objective: Emulate Streamlit's script re-run execution loop and session_state persistence.
"""
import time
from typing import Dict, Any, List, Generator

class SimulatedStreamlitSession:
    def __init__(self):
        self.session_state: Dict[str, Any] = {}
        self.rerun_count = 0

    def initialize_state(self):
        if "messages" not in self.session_state:
            self.session_state["messages"] = [
                {"role": "assistant", "content": "System initialized. How can I assist you?"}
            ]

    def script_execution_pass(self, user_input: str = None) -> List[Dict[str, str]]:
        """Simulates top-to-bottom execution of app.py on a user interaction event."""
        self.rerun_count += 1
        self.initialize_state()

        if user_input:
            # 1. Append user prompt
            self.session_state["messages"].append({"role": "user", "content": user_input})
            
            # 2. Generate simulated assistant response tokens
            assistant_reply = f"Processed query '{user_input}' via Clinical Knowledge Graph."
            self.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

        return self.session_state["messages"]

# Demonstration
if __name__ == "__main__":
    st_app = SimulatedStreamlitSession()

    print("--- User Action 1: Initial Page Load ---")
    history_pass1 = st_app.script_execution_pass()
    print(f"Re-Run Pass #{st_app.rerun_count} | Messages Count: {len(history_pass1)}")
    print(f"Active History: {history_pass1}\n")

    print("--- User Action 2: User types 'Check drug interactions' ---")
    history_pass2 = st_app.script_execution_pass(user_input="Check drug interactions")
    print(f"Re-Run Pass #{st_app.rerun_count} | Messages Count: {len(history_pass2)}")
    for idx, msg in enumerate(history_pass2, 1):
        print(f"  [{idx}] {msg['role'].upper()}: {msg['content']}")
```

---

### Exercise 4 (Expert): Production Two-Tier Caching Decorator with LRU & Singleton Management

Implement custom decorators mimicking Streamlit's `@st.cache_data` (deep-copy data serialization with TTL) and `@st.cache_resource` (singleton memory pointer persistence with thread safety).

```python
"""
Exercise 4: Production Two-Tier Caching Decorator with LRU & Singleton Management
Level: Expert
Objective: Implement cache_data and cache_resource equivalents with TTL and thread safety.
"""
import copy
import functools
import threading
import time
from typing import Dict, Tuple, Any, Callable

def cache_resource(func: Callable) -> Callable:
    """Emulates @st.cache_resource: Singleton pointer persistence without copying."""
    singleton_store: Dict[str, Any] = {}
    lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = func.__name__
        with lock:
            if key not in singleton_store:
                singleton_store[key] = func(*args, **kwargs)
            return singleton_store[key]

    return wrapper

def cache_data(ttl_seconds: int = 60):
    """Emulates @st.cache_data: Deep-copy serialization with Time-To-Live (TTL)."""
    data_store: Dict[Tuple, Tuple[float, Any]] = {}
    lock = threading.Lock()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = (func.__name__, args, frozenset(kwargs.items()))
            now = time.time()

            with lock:
                if cache_key in data_store:
                    expiry, data = data_store[cache_key]
                    if now < expiry:
                        return copy.deepcopy(data)  # Return independent copy
                    del data_store[cache_key]

                # Compute fresh
                fresh_result = func(*args, **kwargs)
                data_store[cache_key] = (now + ttl_seconds, fresh_result)
                return copy.deepcopy(fresh_result)

        return wrapper
    return decorator

# Demonstration
@cache_resource
def load_massive_vector_store():
    time.sleep(0.5)  # Simulate heavy disk I/O
    return {"engine": "ChromaDB", "vectors_count": 100000}

@cache_data(ttl_seconds=2)
def query_clinical_terms(term: str):
    return {"search_term": term, "timestamp": time.time()}

if __name__ == "__main__":
    print("=== CACHE_RESOURCE (SINGLETON) BENCHMARK ===")
    t0 = time.time()
    db1 = load_massive_vector_store()
    t_cold = time.time() - t0
    print(f"Cold Call: {t_cold:.4f}s")

    t1 = time.time()
    db2 = load_massive_vector_store()
    t_warm = time.time() - t1
    print(f"Warm Call: {t_warm:.6f}s (Identical Object: {db1 is db2})")

    print("\n=== CACHE_DATA (SERIALIZED COPY WITH TTL) BENCHMARK ===")
    res1 = query_clinical_terms("hypertension")
    res2 = query_clinical_terms("hypertension")
    print(f"Deep Copy Check: {res1 is not res2} (Separate memory copies)")
```

---

## Part 5: Production Engineering, Edge Cases & Failure Modes ⚙️ ⚡

### 5.1 Leaking API Keys to Public GitHub Repositories (The GitLeaks Defense)

- **The Threat:** Automated scraper bots scan GitHub commits within 30 seconds of push. A leaked API key can result in thousands of dollars in fine-tuning abuse or unauthorized token usage.
- **Defensive Safeguards:**
  1. Add `.env` and `.streamlit/secrets.toml` to `.gitignore`.
  2. Install `gitleaks` as a mandatory pre-commit hook (`gitleaks protect --staged`).
  3. Enable **GitHub Secret Scanning & Push Protection** in repository security settings to reject pushes containing high-entropy keys.

---

### 5.2 The Re-Run Trap: Expensive Model Reloads on Every Widget Click

If a developer places model or vector store initializations in the top-level script without caching:
- Every keystroke in a text input or click on a button triggers a full script re-run, reloading hundreds of megabytes of embeddings into RAM.
- **Fix:** Always isolate stateful objects inside `@st.cache_resource`.

---

### 5.3 Streamlit Concurrency Limitations: Single-Process GIL vs FastAPI Decoupling

- **Limitation:** Streamlit runs as a single Python process. High-concurrency enterprise traffic (e.g. 500 simultaneous users) saturates Python's Global Interpreter Lock (GIL).
- **Production Architecture:**
  - Run Streamlit purely as a lightweight front-end UI.
  - Offload heavy LLM reasoning and vector searches to a scalable **FastAPI / vLLM backend cluster** running behind an NGINX load balancer on Kubernetes.

---

### 5.4 Secrets Management: `.streamlit/secrets.toml` vs Environment Variables

- **Local Development:** Store local credentials in `.streamlit/secrets.toml` (never committed to git) or `.env`.
- **Cloud & Container Production:** Inject credentials via standard container environment variables (`OPENAI_API_KEY`). Streamlit seamlessly resolves `st.secrets["KEY"]` to matching OS environment variables.

---

### 5.5 Enterprise Case Study: Building an Internal Legal Contract Review Copilot

- **Business Scenario:** A corporate legal department handles 2,000 vendor agreements per month. Attorneys spend 6 hours per contract manually searching for indemnity clauses and non-compete liabilities.
- **System Architecture:**
  1. **Dependencies:** `pyproject.toml` with pinned lockfile (`langchain`, `chromadb`, `streamlit`, `pdfplumber`).
  2. **Version Control:** Enforced conventional commits (`feat: add clause extraction`) and Git LFS for contract datasets.
  3. **Streamlit UI:** 3-column layout featuring PDF upload with real-time rendering, risk radar charts, and conversational SBAR chat.
  4. **Outcome:** Turnaround dropped from 6 hours to 20 minutes, with zero dependency drift across the 15-person engineering team.

---

## Part 6: Video Masterclasses, Lab Suites & Review Questions 🎬

### 6.1 Telugu Tech Masterclasses & Global Visual 3D Animations

To solidify your intuitive and architectural grasp of modern development workflows, Git collaboration, and Streamlit front-ends, study these curated video resources:

```
+---------------------------------------------------------------------------------------------------+
|                               CURATED MASTERCLASSES & BENCHMARKS                                  |
+---------------------------------------------------------------------------------------------------+
```

#### 🌟 Telugu Tech Masterclasses (Local Language Foundation)
- **Python Life Telugu — Git & GitHub Complete Tutorial in Telugu:** Master Git version control, branching, merge conflicts, and GitHub repository management in Telugu. (Search: `Python Life Telugu Git GitHub Tutorial`).
- **Vamsi Bhavani — Streamlit Full Course in Telugu:** Step-by-step guide to building interactive web applications and AI dashboards using Streamlit in Telugu. (Search: `Vamsi Bhavani Streamlit Full Course`).
- **Telugu Tech Tutorials — Virtual Environments & Package Management:** Practical walkthrough of Python `venv`, `pip`, and project directory organization. (Search: `Telugu Tech Tutorials Python Virtual Environment`).

#### 🎨 Global Visual 3D Animations & Deep-Dive Lectures
- **freeCodeCamp.org — AI Agents For Beginners:** Modular engineering architecture, tools, and user interface integration. [Watch on YouTube](https://www.youtube.com/watch?v=xM7E_Of1J80)
- **freeCodeCamp.org — LangChain Crash Course for Beginners:** Building end-to-end applications, connecting chains to front-ends, and dependency management. [Watch on YouTube](https://www.youtube.com/watch?v=kYRB-v9z610)
- **Andrej Karpathy — State of GPT:** LLM engineering lifecycle, System 1 vs System 2 thinking, and production deployment considerations. [Watch on YouTube](https://www.youtube.com/watch?v=bZQun8Y4L2A)
- **ByteByteGo — How Git Works Under the Hood (Blobs, Trees, Commits):** 3D animated architectural explanation of Git's content-addressable storage, SHA-1 Merkle trees, and pointer mechanics. (Search: `ByteByteGo How Git Works Under the Hood`).

---

### 6.2 Complete Hands-On Lab Walkthrough

The companion production lab script [`code/development_workflow_streamlit_lab.py`](file:///c:/Users/sriva/OneDrive/Desktop/GEN%20AI%20COURSE/6.%20End-to-End%20Development%20&%20MLOps/code/development_workflow_streamlit_lab.py) contains a full, standalone, battle-tested implementation with 5 comprehensive experiments:

```
6. End-to-End Development & MLOps/
├── assets/
│   ├── 01_medical_chatbot_architecture.jpg
│   ├── 02_clinical_rag_pipeline.jpg
│   ├── 03_docker_mlops_deployment.jpg
│   └── 04_dev_workflow_git_streamlit_pipeline.jpg
├── code/
│   ├── medical_chatbot_architecture_lab.py     <-- Lab 01 (Clinical Architecture & MLOps)
│   └── development_workflow_streamlit_lab.py   <-- Lab 02 (Dev Workflow & Streamlit State Lab)
├── Application Architecture - Building complex systems, such as the Medical Chatbot, from concept to implementation.md
├── Deployment - Strategies for testing, deploying, and operationalizing Generative AI applications for production use.md
└── Development Workflow - Managing dependencies, version control with Git and GitHub, and building front-end interfaces with Streamlit.md
```

#### Overview of the 5 Lab Experiments:
1. **Experiment 1: Declarative Dependency Manifest & Lockfile Cryptographic Validator** — Parses `pyproject.toml` schemas, verifies dependency ranges, and audits lockfile SHA-256 integrity hashes to guarantee reproducible builds.
2. **Experiment 2: Git Repository Hygiene & Pre-Commit Secret Shielding Simulator** — Simulates an automated pre-commit hook that scans code files for leaked API keys (OpenAI, Anthropic, Hugging Face) and validates conventional commit messages.
3. **Experiment 3: Streamlit Reactive Re-Run Engine & Session State Simulator** — Implements a pure Python simulation of Streamlit's reactive execution loop, demonstrating persistent multi-turn chat memory across full script re-runs.
4. **Experiment 4: Resource Caching Benchmark (`@st.cache_resource` Simulation)** — Measures latency differences between un-cached model instantiations (2.5s delay) versus singleton cached resource access ($<0.01\text{ms}$), proving re-run optimization.
5. **Experiment 5: End-to-End Conversational UI Pipeline with Streaming Generator** — Simulates real-time token streaming (`st.write_stream`), message rendering, and structured clinical disclaimer injection into active session state.

---

### 6.3 Comprehensive Self-Assessment & Review Questions

Test your architectural understanding of development workflows, Git/GitHub, and Streamlit. Click each question to expand the comprehensive explanation.

<details>
<summary><b>Q1: Why is committing an exact lockfile (`poetry.lock` or `requirements.lock`) mandatory in enterprise Generative AI projects, rather than relying solely on a loose `requirements.txt`?</b></summary>
<br>

**Answer:**
1. **Transitive Dependency Drift:** Modern LLM libraries (like LangChain, LlamaIndex, or Hugging Face Transformers) depend on dozens of sub-libraries (e.g. `pydantic`, `aiohttp`, `tiktoken`, `tokenizers`). A loose specification like `langchain>=0.2.0` allows pip to install whatever sub-dependency version is latest at the moment of build. If a sub-dependency releases a minor breaking change, your deployment pod will fail to build unexpectedly.
2. **Cryptographic Integrity & Supply Chain Security:** Lockfiles store SHA-256 checksums of every downloaded wheel package. This ensures that the exact binary wheels verified by your security team in development are identical to those running in production, preventing supply-chain package tampering.
3. **Build Determinism:** A lockfile guarantees that every engineer on the team and every Docker container across AWS/GCP builds an identical, byte-for-byte virtual environment.
</details>

<br>

<details>
<summary><b>Q2: What is the fundamental operational difference between `@st.cache_data` and `@st.cache_resource` in Streamlit applications?</b></summary>
<br>

**Answer:**
- **`@st.cache_data`:** Designed for **serializable data objects** (such as pandas DataFrames, SQL query results, raw text, and JSON dictionaries). When cached, Streamlit serializes the output. On cache hits, it creates a clean, independent copy of the data to prevent accidental state mutation.
- **`@st.cache_resource`:** Designed for **non-serializable, stateful resources** (such as PyTorch models, LangChain LLM pipelines, ChromaDB vector stores, database connection pools, and thread locks). It maintains a single global shared pointer (singleton). It never copies the object, allowing heavy 10 GB model weights or open socket connections to persist efficiently across script re-runs and multiple browser sessions.
</details>

<br>

<details>
<summary><b>Q3: How does Streamlit's reactive script re-run model work, and why does an un-cached LLM pipeline cause severe performance degradation?</b></summary>
<br>

**Answer:**
- **Execution Model:** Streamlit executes Python scripts strictly from top to bottom. Whenever an interactive widget event occurs (a user types in a chat box, clicks a button, or toggles a slider), Streamlit interrupts the process and re-executes the entire script from line 1.
- **The Degradation Trap:** If an LLM pipeline or vector store is initialized directly in the top-level script without caching, Streamlit will re-download model weights, re-parse configuration files, and re-instantiate embedding models on **every single user keystroke or click**. This introduces 2 to 5 seconds of unnecessary latency per interaction. Using `@st.cache_resource` ensures the model is instantiated exactly once on server startup.
</details>

<br>

<details>
<summary><b>Q4: Why should large neural network weights and vector database indexes never be committed directly to standard Git, and how does Git LFS solve this?</b></summary>
<br>

**Answer:**
1. **Git Repository Bloat:** Git is an append-only version control system designed for delta-compressed source code. When binary files (like 4 GB `.safetensors` model weights or 500 MB ChromaDB SQLite files) are committed, every modification stores a full, uncompressed copy in the hidden `.git/objects/` folder. The repository size quickly balloons to tens of gigabytes, making `git clone` impossibly slow for teammates.
2. **GitHub Push Ceilings:** GitHub strictly blocks any individual file push exceeding 100 MB.
3. **Git LFS Mechanism:** Git Large File Storage replaces the massive binary inside the Git tree with a tiny text pointer containing the file's SHA-256 hash and size. The actual heavy binaries are uploaded to dedicated object storage servers, and are only downloaded on demand when the developer checks out that specific commit.
</details>

<br>

<details>
<summary><b>Q5: What are Pre-Commit Hooks, and how do tools like Gitleaks protect enterprise AI development teams?</b></summary>
<br>

**Answer:**
- **Pre-Commit Hooks:** Client-side scripts triggered automatically when a developer executes `git commit`. The commit is blocked from finalizing if any hook check fails.
- **The Gitleaks Defense:** Gitleaks scans staged code changes using high-entropy heuristic analysis and regex patterns matching known cloud and AI provider API keys (`sk-proj-...`, `ghp_...`, `hf_...`).
- **Why Pre-Commit is Critical:** If a secret is committed locally—even if it is deleted in the very next commit—it remains permanently embedded in Git's historical commit graph. If pushed to GitHub, automated scraper bots extract the key in seconds. Pre-commit hooks intercept the secret **before it is ever recorded in Git history**, ensuring zero accidental credentials leakage.
</details>

---

### 6.4 Key Takeaways & Architectural Checklist

| Architectural Check | Implementation Standard | Status |
| :--- | :--- | :--- |
| **Deterministic Dependencies** | Declare in `pyproject.toml`; lock cryptographically with lockfiles | ✅ Verified |
| **Secret Shielding** | Block API key leaks via pre-commit hooks (Gitleaks) & push protection | ✅ Verified |
| **Binary Tracking** | Track `.safetensors`, `.gguf`, and large data files via Git LFS | ✅ Verified |
| **Streamlit State Management** | Preserve chat history in `st.session_state`; stream tokens with `st.write_stream` | ✅ Verified |
| **Performance Caching** | Wrap stateful models in `@st.cache_resource`; data in `@st.cache_data` | ✅ Verified |
| **Production Decoupling** | Use Streamlit for presentation; offload inference to FastAPI / vLLM backends | ✅ Verified |

---

*Continue to the companion lab in [`code/development_workflow_streamlit_lab.py`](file:///c:/Users/sriva/OneDrive/Desktop/GEN%20AI%20COURSE/6.%20End-to-End%20Development%20&%20MLOps/code/development_workflow_streamlit_lab.py) to run all 5 interactive experiments.*
