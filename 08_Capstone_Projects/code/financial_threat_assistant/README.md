# 🛡️ Enterprise Financial & Threat Intelligence Assistant

An autonomous multi-agent system powered by the ReAct (Reasoning + Acting) architecture that analyzes SEC EDGAR 10-K filings, evaluates Altman Z-Score corporate solvency, correlates MITRE ATT&CK cybersecurity threats, and synthesizes executive risk reports.

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

### 3. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```

---

## 📁 Architecture & File Layout

- **`models.py`**: Pydantic v2 schemas for solvency metrics, SEC filings, threat entities, and ReAct steps.
- **`tools.py`**: Deterministic callable tools (`sec_edgar_retriever`, `threat_intel_graph`, `financial_ratio_calculator`, `financial_sentiment_scorer`).
- **`agent.py`**: Autonomous ReAct orchestrator managing the `Thought -> Action -> ActionInput -> Observation` reasoning loop.
- **`app.py`**: Interactive Streamlit web interface with chain-of-thought accordion and threat correlation grids.
- **`demo.py`**: Automated end-to-end system verification script.
