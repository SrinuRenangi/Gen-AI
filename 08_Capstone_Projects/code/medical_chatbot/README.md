# 🩺 Clinical Medical Chatbot & Diagnostic Decision Support Assistant

A production-grade, clinical AI decision support system featuring HIPAA Safe Harbor PHI de-identification, an emergency triage circuit breaker, hybrid BM25 + dense semantic retrieval with Reciprocal Rank Fusion (RRF), and structured SBAR differential diagnosis synthesis.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Verification Demo
```bash
python demo.py
```

### 3. Launch the Streamlit Clinical Interface
```bash
streamlit run app.py
```

---

## 📁 Architecture & File Layout

- **`models.py`**: Pydantic v2 data models for clinical entities, SBAR payloads, and triage levels.
- **`triage_guard.py`**: Ingress HIPAA PHI scrubber and deterministic red-flag emergency circuit breaker.
- **`hybrid_retriever.py`**: BM25 keyword matching + dense semantic search fused via RRF ($k=60$).
- **`clinical_engine.py`**: Orchestration engine with OpenAI, Gemini, and offline clinical mock modes.
- **`app.py`**: Interactive Streamlit web interface for clinical triage and differential diagnosis.
- **`demo.py`**: End-to-end automated system verification script.
