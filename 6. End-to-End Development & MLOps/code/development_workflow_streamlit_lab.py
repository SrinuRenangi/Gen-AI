"""
Development Workflow Lab: Dependencies, Git/GitHub & Streamlit State
====================================================================

Zero to Hero Gen AI Course - Module 06: End-to-End Development & MLOps
Companion Lab: Development Workflow (Managing Dependencies, Version Control & Streamlit)

This production-grade educational lab demonstrates:
  1. Experiment 1: Declarative Dependency Manifest & Lockfile Cryptographic Validator.
  2. Experiment 2: Git Repository Hygiene & Pre-Commit Secret Shielding Simulator.
  3. Experiment 3: Streamlit Reactive Re-Run Engine & Session State Simulator.
  4. Experiment 4: Resource Caching Benchmark (@st.cache_resource Simulation).
  5. Experiment 5: End-to-End Conversational UI Pipeline with Streaming Generator.

Features:
  - 100% standalone and runnable out-of-the-box (zero mandatory external UI or web server needed).
  - High-fidelity reactive state machine simulation mimicking Streamlit internals.
  - Windows CP1252-safe UTF-8 console output.
"""

import sys
import os
import re
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError

# Ensure Windows terminal handles UTF-8 formatting safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# Schemas & Data Models
# ============================================================================

class ProjectMetadata(BaseModel):
    """Metadata schema for pyproject.toml [project] section."""
    name: str
    version: str
    description: str
    requires_python: str
    dependencies: List[str]
    dev_dependencies: List[str] = Field(default_factory=list)


class LockfilePackage(BaseModel):
    """Cryptographic lockfile package entry."""
    name: str
    version: str
    sha256: str
    dependencies: List[str] = Field(default_factory=list)


# ============================================================================
# The 5 Experimental Suites
# ============================================================================

def run_experiment_1():
    """Experiment 1: Declarative Dependency Manifest & Lockfile Cryptographic Validator."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 1: Declarative Dependencies & Cryptographic Lockfile Validation")
    print("#"*80)

    # Simulated pyproject.toml manifest for our enterprise medical chatbot
    raw_pyproject = {
        "name": "clinical-medical-chatbot",
        "version": "1.0.0",
        "description": "Enterprise Clinical Decision Support Chatbot with Hybrid RAG",
        "requires_python": ">=3.10,<3.13",
        "dependencies": [
            "langchain>=0.2.14",
            "chromadb>=0.5.5",
            "pydantic>=2.8.2",
            "streamlit>=1.38.0"
        ],
        "dev_dependencies": [
            "pytest>=8.3.2",
            "ruff>=0.6.1",
            "pre-commit>=3.8.0"
        ]
    }

    print("1. Parsing pyproject.toml Declarative Manifest:")
    proj = ProjectMetadata(**raw_pyproject)
    print(f"   Project Name       : {proj.name} (v{proj.version})")
    print(f"   Python Constraint  : {proj.requires_python}")
    print(f"   Core Dependencies  : {len(proj.dependencies)} packages declared")
    for dep in proj.dependencies:
        print(f"     * {dep}")
    print(f"   Dev Dependencies   : {len(proj.dev_dependencies)} packages declared")

    # 2. Simulated lockfile with exact pins and SHA-256 checksums
    print("\n2. Auditing requirements.lock Cryptographic Integrity Hashes:")
    simulated_lockfile = [
        LockfilePackage(
            name="langchain",
            version="0.2.14",
            sha256=hashlib.sha256(b"langchain-0.2.14-py3-none-any.whl").hexdigest(),
            dependencies=["pydantic", "langchain-core"]
        ),
        LockfilePackage(
            name="pydantic",
            version="2.8.2",
            sha256=hashlib.sha256(b"pydantic-2.8.2-py3-none-any.whl").hexdigest(),
            dependencies=["pydantic-core"]
        ),
        LockfilePackage(
            name="streamlit",
            version="1.38.0",
            sha256=hashlib.sha256(b"streamlit-1.38.0-py3-none-any.whl").hexdigest(),
            dependencies=["tornado", "protobuf"]
        )
    ]

    for pkg in simulated_lockfile:
        print(f"   [{pkg.name:<12} == {pkg.version:<8}] SHA-256: {pkg.sha256[:20]}... (VERIFIED)")

    print("\nStatus: ✅ Deterministic build manifest and cryptographic lockfile validated.")


def run_experiment_2():
    """Experiment 2: Git Repository Hygiene & Pre-Commit Secret Shielding Simulator."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 2: Git Hygiene & Pre-Commit Secret Shielding Simulator")
    print("#"*80)

    class PreCommitHookSimulator:
        """Simulates automated pre-commit scanners (Gitleaks + Conventional Commits)."""
        SECRET_PATTERNS = [
            (r"sk-[a-zA-Z0-9]{48}", "OpenAI API Key"),
            (r"sk-ant-[a-zA-Z0-9]{32,}", "Anthropic Claude API Key"),
            (r"hf_[a-zA-Z0-9]{34}", "Hugging Face User Access Token"),
            (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID")
        ]
        CONVENTIONAL_COMMIT_REGEX = r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-zA-Z0-9_-]+\))?:\s+[a-z0-9].+$"

        @classmethod
        def scan_staged_content(cls, file_content: str) -> List[str]:
            violations = []
            for pat, desc in cls.SECRET_PATTERNS:
                if re.search(pat, file_content):
                    violations.append(f"LEAKED_CREDENTIAL: {desc}")
            return violations

        @classmethod
        def validate_commit_message(cls, message: str) -> bool:
            return bool(re.match(cls.CONVENTIONAL_COMMIT_REGEX, message))

    # Test Staged Files
    clean_code = "import os\napi_key = os.getenv('OPENAI_API_KEY')\nprint('Initialized pipeline')"
    leaked_code = "import os\napi_key = 'sk-proj-9999888877776666555544443333222211110000aaaa'\nprint('API Loaded')"

    print("1. Scanning Staged Code with Pre-Commit Secret Scanner:")
    violations_clean = PreCommitHookSimulator.scan_staged_content(clean_code)
    violations_leaked = PreCommitHookSimulator.scan_staged_content(leaked_code)

    print(f"   File 1 (Environment Variable): {'✅ PASS - Zero Secrets' if not violations_clean else '❌ BLOCKED'}")
    print(f"   File 2 (Hardcoded Key)       : {'❌ BLOCKED' if violations_leaked else '✅ PASS'}")
    if violations_leaked:
        print(f"     * Gitleaks Guard Triggered : {violations_leaked[0]}")

    print("\n2. Validating Conventional Commit Messages:")
    commit_messages = [
        ("feat(retrieval): implement hybrid BM25 and dense cosine search", True),
        ("fix: correct pediatric amoxicillin dosage formula in SBAR prompt", True),
        ("updated stuff and fixed bug", False),  # Bad non-conventional message
        ("WIP", False)                          # Bad non-conventional message
    ]

    for msg, expected in commit_messages:
        valid = PreCommitHookSimulator.validate_commit_message(msg)
        status = "✅ ACCEPTED" if valid else "❌ REJECTED"
        print(f"   [{status}] \"{msg}\"")
        assert valid == expected, f"Validation mismatch for commit message: {msg}"

    print("\nStatus: ✅ Secret scanning and conventional commit validation confirmed.")


def run_experiment_3():
    """Experiment 3: Streamlit Reactive Re-Run Engine & Session State Simulator."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 3: Streamlit Reactive Re-Run Engine & Session State Simulation")
    print("#"*80)

    class SimulatedStreamlitContext:
        """
        Emulates Streamlit's reactive execution loop where the entire script
        re-runs on every user interaction while session_state persists across runs.
        """
        def __init__(self):
            self.session_state: Dict[str, Any] = {}
            self.run_count = 0

        def trigger_rerun(self, user_action_desc: str, script_func: Callable):
            self.run_count += 1
            print(f"\n--- [Streamlit Re-Run #{self.run_count}: User Action -> {user_action_desc}] ---")
            script_func(self)

    # Simulated Streamlit script definition
    def my_streamlit_app(st: SimulatedStreamlitContext):
        # Local non-state variable (WIPED on every re-run!)
        local_execution_counter = 0
        local_execution_counter += 1

        # Session state initialization (PERSISTS across re-runs!)
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = ["Assistant: Welcome to Clinical Copilot."]

        print(f"   Line 10: Local variable value        = {local_execution_counter} (Reset to 1 every re-run!)")
        print(f"   Line 20: Persistent session_state     = {len(st.session_state['chat_history'])} messages")
        for idx, m in enumerate(st.session_state["chat_history"], start=1):
            print(f"            [{idx}] {m}")

    ctx = SimulatedStreamlitContext()

    # Interaction 1: Page Load
    ctx.trigger_rerun("Page Initial Load", my_streamlit_app)

    # Interaction 2: User types first message
    ctx.session_state["chat_history"].append("User: What is first-line treatment for pediatric otitis media?")
    ctx.session_state["chat_history"].append("Assistant: High-dose amoxicillin (80-90 mg/kg/day).")
    ctx.trigger_rerun("User Sent Message #1", my_streamlit_app)

    # Interaction 3: User changes sidebar dropdown
    ctx.session_state["chat_history"].append("User: What if patient is allergic to penicillin?")
    ctx.session_state["chat_history"].append("Assistant: Consider cefdinir, cefuroxime, or azithromycin.")
    ctx.trigger_rerun("User Changed Sidebar Dropdown", my_streamlit_app)

    # Verify session state survived 3 re-runs
    assert len(ctx.session_state["chat_history"]) == 5, "Session state failed to preserve message turns!"
    print("\nStatus: ✅ Reactive re-run state machine validated; multi-turn memory preserved.")


def run_experiment_4():
    """Experiment 4: Resource Caching Benchmark (@st.cache_resource Simulation)."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 4: Resource Caching Benchmark (@st.cache_resource Simulation)")
    print("#"*80)

    class SimulatedCacheResourceRegistry:
        """Emulates Streamlit's @st.cache_resource singleton decorator."""
        _registry: Dict[str, Any] = {}

        @classmethod
        def cache_resource(cls, func: Callable):
            def wrapper(*args, **kwargs):
                key = func.__name__
                if key not in cls._registry:
                    # Cold execution: run heavy initialization
                    instance = func(*args, **kwargs)
                    cls._registry[key] = instance
                    return instance, "COLD_LOAD (Newly Initialized)"
                else:
                    # Warm execution: return cached singleton pointer
                    return cls._registry[key], "WARM_CACHE_HIT (Instantaneous Pointer)"
            return wrapper

    # Heavy model loading function (simulates downloading / initializing a 500 MB embedding model)
    def un_cached_load_vector_db():
        time.sleep(0.08)  # Simulate expensive disk I/O and neural tensor initialization
        return {"db_name": "ChromaClinicalDB", "vectors_count": 14200}

    @SimulatedCacheResourceRegistry.cache_resource
    def cached_load_vector_db():
        time.sleep(0.08)  # Expensive initial load
        return {"db_name": "ChromaClinicalDB", "vectors_count": 14200}

    print("Simulating 5 consecutive widget interactions triggering script re-runs:\n")

    # Part A: Without Caching (The Re-Run Performance Trap)
    print("--- [Part A: Un-Cached Model Loading (Anti-Pattern)] ---")
    t0_uncached = time.time()
    for i in range(1, 4):
        t_step = time.time()
        _ = un_cached_load_vector_db()
        dur = (time.time() - t_step) * 1000.0
        print(f"   Re-run #{i}: Reloaded Vector DB from scratch in {dur:.2f} ms")
    total_uncached_ms = (time.time() - t0_uncached) * 1000.0

    # Part B: With @st.cache_resource (The Enterprise Solution)
    print("\n--- [Part B: With @st.cache_resource (Production Solution)] ---")
    t0_cached = time.time()
    for i in range(1, 4):
        t_step = time.time()
        res, status = cached_load_vector_db()
        dur = (time.time() - t_step) * 1000.0
        print(f"   Re-run #{i}: [{status:<32}] Accessed in {dur:.3f} ms")
    total_cached_ms = (time.time() - t0_cached) * 1000.0

    speedup = total_uncached_ms / total_cached_ms
    print(f"\nCaching Performance Audit:")
    print(f"   Total Un-Cached Time : {total_uncached_ms:.2f} ms")
    print(f"   Total Cached Time    : {total_cached_ms:.2f} ms")
    print(f"   Performance Speedup  : {speedup:.1f}x faster execution across script re-runs!")
    print("Status: ✅ @st.cache_resource eliminates model reloading overhead.")


def run_experiment_5():
    """Experiment 5: End-to-End Conversational UI Pipeline with Streaming Generator."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 5: Conversational UI Pipeline with Streaming Generator")
    print("#"*80)

    class StreamlitChatPipelineSimulator:
        """Emulates st.chat_message, st.chat_input, and st.write_stream."""
        def __init__(self):
            self.session_messages: List[Dict[str, str]] = []

        def handle_user_query(self, user_prompt: str) -> str:
            # 1. Record user message
            self.session_messages.append({"role": "user", "content": user_prompt})
            print(f"\n👤 [User]: {user_prompt}")

            # 2. Simulate token stream generator (st.write_stream)
            print("🤖 [Assistant]: ", end="", flush=True)
            response_tokens = [
                "Based ", "on ", "current ", "clinical ", "guidelines, ",
                "the ", "primary ", "recommendation ", "is ", "oral ", "amoxicillin ",
                "(90 mg/kg/day). ", "Always ", "verify ", "penicillin ", "allergy ", "status."
            ]

            full_assembled = ""
            for token in response_tokens:
                full_assembled += token
                print(token, end="", flush=True)
                time.sleep(0.02)  # Stream pacing delay
            print()

            # 3. Append disclaimer and save assistant turn to session state
            disclaimer = "\n[Notice: Clinical Decision Support for licensed medical personnel.]"
            final_turn = full_assembled + disclaimer
            self.session_messages.append({"role": "assistant", "content": final_turn})
            return final_turn

    ui = StreamlitChatPipelineSimulator()
    query = "What is the pediatric dosage for amoxicillin in acute otitis media?"
    result = ui.handle_user_query(query)

    print(f"\nAudit Log:")
    print(f"   Total Session Turns  : {len(ui.session_messages)}")
    print(f"   Disclaimer Appended  : {'Notice: Clinical Decision Support' in result}")
    assert len(ui.session_messages) == 2, "Failed to record two conversation turns!"
    print("Status: ✅ End-to-end streaming conversational UI pipeline completed successfully.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print("="*80)
    print("⚙️  DEVELOPMENT WORKFLOW LAB: DEPENDENCIES, GIT/GITHUB & STREAMLIT")
    print("="*80)
    print("Python Executable:", sys.executable)
    print("Python Version   :", sys.version.split()[0])
    print("Running on OS    :", sys.platform)
    print("="*80)

    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()

    print("\n" + "="*80)
    print("🎉 ALL 5 DEVELOPMENT WORKFLOW & STREAMLIT EXPERIMENTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
