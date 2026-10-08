# ⚡ AI Coding Question & Assessment Generator

> **Zero to Hero GenAI Course — Phase 01: GenAI Foundations (Day 04 Capstone Project)**

An end-to-end, production-grade AI system that dynamically generates LeetCode/HackerRank-style coding interview questions, enforces strict Pydantic schemas, safely compiles and runs candidate submissions in a sandboxed test harness with AST security checks, and generates automated Big-O complexity reviews and progressive hints.

---

## 🌟 Key Features

1. **Multi-LLM Question Generation:**
   - OpenAI (`gpt-4o`, `gpt-4o-mini`) via Structured Outputs
   - Google Gemini (`gemini-1.5-flash`) via JSON Mode
   - High-fidelity **Offline Mock Engine** for 100% reliable local offline execution without API keys.

2. **Strict Schema & Type Safety:**
   - Powered by Pydantic models for problem specifications, public test cases, private hidden edge cases, starter code, and optimal solutions.

3. **Secure AST Sandboxing & Test Runner:**
   - Static AST validation blocks malicious imports (`os`, `sys`, `subprocess`, `socket`) and unsafe calls (`eval`, `exec`, `open`).
   - Timeout protection (default: 2.0s) prevents infinite loops.
   - Deep equality checks with float tolerance.

4. **Automated AI Code Review & Feedback:**
   - Detects code smells and computes empirical loop nesting and recursion.
   - Compares candidate complexity against theoretical Big-O optimal limits.
   - Delivers progressive, 3-tier hints.

5. **Dual Interfaces:**
   - 💻 **Interactive CLI:** Terminal application powered by `rich` tables and syntax highlighting.
   - 🌐 **Modern Web UI:** Streamlit application with live problem generation, in-browser code editor, and instant test grading.

6. **Multi-Format Export:**
   - Export problem sheets directly to **Markdown** or raw **JSON**.

---

## 📁 Project Architecture

```
project/
├── models.py        # Pydantic schemas (CodingProblem, TestCase, EvaluationReport)
├── prompts.py       # Prompt engineering templates & few-shot demonstrations
├── generator.py     # Multi-provider question generator (OpenAI, Gemini, Mock)
├── evaluator.py     # Sandboxed test execution engine with AST security & timeouts
├── reviewer.py      # AI reviewer for Big-O analysis, feedback & hints
├── exporter.py      # Markdown & JSON exporter
├── cli.py           # Interactive rich terminal UI
├── app.py           # Streamlit interactive web application
├── demo.py          # Automated verification script
└── requirements.txt # Project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Automated Demo

Verify all systems (generation, sandbox evaluation, security interception, grading):

```bash
python demo.py
```

### 3. Launch the Interactive CLI

```bash
python cli.py
```

### 4. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

---

## 🔑 Environment Variables (Optional)

To enable live LLM generation with cloud providers, set your API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:GEMINI_API_KEY="your-gemini-api-key"

# Linux / macOS
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

*Note: If no API key is set, the application automatically uses the built-in Offline Mock Engine seamlessly.*
