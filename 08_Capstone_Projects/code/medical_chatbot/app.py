"""
Clinical Medical Chatbot - Streamlit Clinical Decision Support Interface
=========================================================================
Module 08: Capstone Projects - Project 02: Clinical Medical Chatbot
File: app.py

Features:
- Live HIPAA Safe Harbor PHI Redaction Display.
- Emergency Triage Circuit Breaker Alerts (911 Directives).
- Structured SBAR Diagnostic Reasoning Cards.
- Hybrid Evidence Inspector (RRF-fused PubMed Guidelines).
"""

import streamlit as st
from clinical_engine import ClinicalReasoningEngine, STATUTORY_DISCLAIMER
from models import TriageSeverity


st.set_page_config(
    page_title="Clinical Medical CDS Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical dashboard aesthetics
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .phi-badge { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .emergency-card { background-color: #FEE2E2; border-left: 6px solid #DC2626; padding: 16px; border-radius: 6px; margin: 15px 0; }
    .sbar-header { font-weight: 700; color: #1E40AF; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 0.05em; }
    .card-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return ClinicalReasoningEngine(provider="auto")

engine = get_engine()

# Sidebar: Presets & Provider Config
st.sidebar.title("🩺 Clinical Control Panel")
st.sidebar.markdown(f"**Active Inference Provider:** `{engine.active_provider.upper()}`")

PRESET_CASES = {
    "Case 1: Autoimmune Joint Pain (Non-Emergency)": (
        "Patient Sarah Jenkins (DOB: 04/12/1982, SSN: 123-45-6789, MRN #8849201) reports worsening bilateral "
        "hand and wrist pain for the past 3 months. She experiences severe morning stiffness lasting > 45 minutes "
        "that improves slightly with warm showers. Contact daughter at sjenkins@medmail.org."
    ),
    "Case 2: Acute Myocardial Infarction (Emergency Alert)": (
        "Patient Robert Davis (MRN #99231) presents with crushing chest pain radiating to the left arm and jaw. "
        "Onset 45 minutes ago while shoveling snow, associated with diaphoresis, shortness of breath, and nausea."
    ),
    "Case 3: Productive Cough & Fever (Nominal)": (
        "Patient John Smith complains of a 4-day history of productive cough with purulent rust-colored sputum, "
        "subjective fevers, chills, and mild right-sided pleuritic chest discomfort. No recent travel."
    )
}

selected_case = st.sidebar.selectbox("Load Sample Patient Presentation:", list(PRESET_CASES.keys()))

st.markdown("<div class='main-header'>🩺 Clinical Decision Support (CDS) Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Evidence-Based Differential Diagnosis & Triage Architecture with HIPAA De-Identification</div>", unsafe_allow_html=True)

# Statutory Disclaimer Banner
st.info(f"⚖️ **Statutory Notice:** {STATUTORY_DISCLAIMER}")

# Ingress Input Area
patient_input = st.text_area(
    "Enter Clinical Presentation / Patient Triage Note:",
    value=PRESET_CASES[selected_case],
    height=130
)

col_btn, col_phi = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("🔍 Run Clinical Assessment", type="primary", use_container_width=True)

if analyze_btn and patient_input.strip():
    with st.spinner("Processing clinical ingress, evaluating triage safety, and querying evidence..."):
        triage, sbar, docs = engine.process_consultation(patient_input)

    st.markdown("---")

    # Ingress Safety Firewall Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Triage Severity", triage.severity.value)
    with col2:
        st.metric("PHI Elements Redacted", triage.redacted_phi_count)
    with col3:
        status_text = "🚨 CRITICAL TRIAGE" if triage.is_emergency else "✅ CLINICAL EVALUATION"
        st.metric("Safety Status", status_text)

    # Display PHI Sanitized Note
    with st.expander("🛡️ View HIPAA De-Identified Ingress Text", expanded=(triage.redacted_phi_count > 0)):
        st.markdown(f"**Sanitized Text Sent to Model:**")
        st.code(triage.sanitized_text, language="markdown")

    # Handle Emergency Short-Circuit
    if triage.is_emergency:
        st.markdown(f"""
        <div class='emergency-card'>
            <h3 style='color: #991B1B; margin-top: 0;'>🚨 ACUTE EMERGENCY TRIAGE CIRCUIT BREAKER ACTIVATED</h3>
            <p><b>Triggered Red-Flag Conditions:</b> {", ".join(triage.red_flag_triggers)}</p>
            <p style='font-size: 1.1rem; font-weight: 600; color: #7F1D1D;'>{triage.emergency_directive}</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Algorithmic LLM generation suspended to prevent delay in emergency medical care.")

    # Handle Structured SBAR Clinical Assessment
    elif sbar:
        st.markdown("### 📋 SBAR Structured Clinical Assessment")
        
        # Situation & Background
        c_sit, c_bg = st.columns(2)
        with c_sit:
            st.markdown("<div class='card-box'><span class='sbar-header'>Situation</span><br>" + sbar.situation + "</div>", unsafe_allow_html=True)
        with c_bg:
            st.markdown("<div class='card-box'><span class='sbar-header'>Background</span><br>" + sbar.background + "</div>", unsafe_allow_html=True)

        # Ranked Differential Diagnosis (Assessment)
        st.markdown("#### 🩺 Ranked Differential Diagnosis (Assessment)")
        for idx, diag in enumerate(sbar.assessment, start=1):
            badge_color = "#DC2626" if diag.likelihood == "High" else "#D97706" if diag.likelihood == "Moderate" else "#2563EB"
            with st.container():
                st.markdown(f"""
                <div class='card-box'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-size: 1.1rem; font-weight: 700;'>#{idx} {diag.condition_name}</span>
                        <span>ICD-10: <code>{diag.icd10_code}</code> | Likelihood: <b style='color: {badge_color};'>{diag.likelihood}</b></span>
                    </div>
                    <div style='margin-top: 8px;'>
                        <small><b>Pertinent Positives:</b> {", ".join(diag.pertinent_positives) if diag.pertinent_positives else "None reported"}</small><br>
                        <small><b>Pertinent Negatives:</b> {", ".join(diag.pertinent_negatives) if diag.pertinent_negatives else "None reported"}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("#### 📝 Recommended Next Steps & Clinical Workup")
        for rec in sbar.recommendations:
            st.markdown(f"- 📌 {rec}")

        # Retrieved Evidence Inspector
        if docs:
            with st.expander(f"📚 Retrieved Clinical Practice Guidelines ({len(docs)} Evidence Sources)"):
                for d in docs:
                    st.markdown(f"**[{d.id}] {d.title}**")
                    st.caption(f"Citation: {d.citation}")
                    st.write(d.content)
                    st.markdown("---")
