"""
Enterprise Financial & Threat Intelligence Assistant - Streamlit Dashboard
===========================================================================
Module 08: Capstone Projects - Project 03: Financial & Threat Intelligence
File: app.py

Interactive Web Interface:
- Multi-Agent ReAct Execution Stream.
- Altman Z-Score Solvency Analytics.
- Active CVE & MITRE ATT&CK Threat Correlation Grid.
- Executive Risk Summary & Governance Directives.
"""

import streamlit as st
from agent import EnterpriseIntelligenceAgent
from models import RiskSeverity


st.set_page_config(
    page_title="Enterprise Financial & Threat Intel Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .risk-critical { color: #DC2626; font-weight: 700; }
    .risk-high { color: #EA580C; font-weight: 700; }
    .risk-medium { color: #D97706; font-weight: 700; }
    .risk-low { color: #16A34A; font-weight: 700; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .agent-thought { background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 10px 14px; margin: 8px 0; border-radius: 4px; }
    .agent-action { background-color: #EFF6FF; border-left: 4px solid #2563EB; padding: 10px 14px; margin: 8px 0; border-radius: 4px; }
    .agent-obs { background-color: #F8FAFC; border-left: 4px solid #64748B; padding: 10px 14px; margin: 8px 0; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return EnterpriseIntelligenceAgent(provider="auto")

agent = get_agent()

# Sidebar Controls
st.sidebar.title("🛡️ Intelligence Controls")
st.sidebar.markdown(f"**Autonomous Agent Core:** `{agent.active_provider.upper()}`")

TICKERS = {
    "NVDA (NVIDIA Corporation)": "NVDA",
    "CROWD (CrowdStrike Holdings)": "CROWD",
    "TSLA (Tesla, Inc.)": "TSLA"
}

selected_label = st.sidebar.selectbox("Select Target Enterprise Ticker:", list(TICKERS.keys()))
target_ticker = TICKERS[selected_label]

st.markdown("<div class='main-title'>🛡️ Enterprise Financial & Cyber Threat Intelligence Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Autonomous Multi-Agent ReAct Synthesis of SEC 10-K Filings, Solvency Ratios & MITRE ATT&CK Threat Graphs</div>", unsafe_allow_html=True)

col_run, col_info = st.columns([1, 3])
with col_run:
    run_btn = st.button("🚀 Launch ReAct Autonomous Investigation", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"Agent executing autonomous multi-tool investigation on {target_ticker}..."):
        report = agent.execute_investigation(target_ticker)

    st.markdown("---")

    # High-Level Metric Tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Target Company", report.company_name)
    with m2:
        st.metric("Composite Risk Score", f"{report.overall_risk_score} / 100")
    with m3:
        st.metric("Altman Z-Score", f"{report.altman_metrics.altman_z_score} ({report.altman_metrics.distress_category})")
    with m4:
        st.metric("Active Cyber Threats", len(report.cyber_threat_exposure))

    # Executive Summary Box
    st.markdown("### 📋 Executive Risk Summary")
    st.info(report.executive_summary)

    # ReAct Autonomous Agent Trajectory Inspector
    with st.expander("🤖 Inspect ReAct Agent Reasoning Trajectory (Chain of Thought)", expanded=True):
        for step in report.agent_trajectory:
            st.markdown(f"**[Iteration {step.iteration}]**")
            st.markdown(f"<div class='agent-thought'><b>💭 Thought:</b> {step.thought}</div>", unsafe_allow_html=True)
            if step.action:
                st.markdown(f"<div class='agent-action'><b>⚙️ Action:</b> <code>{step.action}</code>(input={step.action_input})</div>", unsafe_allow_html=True)
            if step.observation:
                st.markdown(f"<div class='agent-obs'><b>👁️ Observation:</b> <code>{step.observation[:200]}...</code></div>", unsafe_allow_html=True)
            st.markdown("")

    # Financial Solvency vs Cyber Threat Two-Column Layout
    col_fin, col_threat = st.columns(2)

    with col_fin:
        st.markdown("### 📊 Fundamental Solvency & Balance Sheet Ratios")
        metrics = report.altman_metrics
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Altman Z-Score Breakdown: <b>{metrics.altman_z_score}</b></h4>
            <p>Status: <b>{metrics.distress_category}</b></p>
            <ul>
                <li>Working Capital / Assets (X₁): <code>{metrics.working_capital_to_assets}</code></li>
                <li>Retained Earnings / Assets (X₂): <code>{metrics.retained_earnings_to_assets}</code></li>
                <li>EBIT / Total Assets (X₃): <code>{metrics.ebit_to_assets}</code></li>
                <li>Market Value of Equity / Liabilities (X₄): <code>{metrics.market_equity_to_liabilities}</code></li>
                <li>Sales / Total Assets (X₅): <code>{metrics.sales_to_assets}</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_threat:
        st.markdown("### 🔒 Active Cybersecurity Threat Graph Correlation")
        for threat in report.cyber_threat_exposure:
            sev_class = f"risk-{threat.severity.value.lower()}"
            st.markdown(f"""
            <div class='metric-card'>
                <div style='display:flex; justify-content:space-between;'>
                    <b>[{threat.ioc_type}] {threat.value}</b>
                    <span class='{sev_class}'>[{threat.severity.value.upper()}]</span>
                </div>
                <small>MITRE ATT&CK: <code>{threat.mitre_attack_id or 'N/A'}</code></small>
                <p style='margin-top:6px; font-size:0.9rem;'>{threat.description}</p>
            </div>
            """, unsafe_allow_html=True)

    # Executive Governance Directives
    st.markdown("### ⚖️ Strategic Governance & Risk Recommendations")
    for idx, rec in enumerate(report.compliance_recommendations, start=1):
        st.markdown(f"{idx}. 📌 {rec}")
