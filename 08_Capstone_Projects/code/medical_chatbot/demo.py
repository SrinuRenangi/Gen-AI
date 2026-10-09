"""
Clinical Medical Chatbot - Automated System Verification Demo
=============================================================
Module 08: Capstone Projects - Project 02: Clinical Medical Chatbot
File: demo.py

Executes all 5 clinical assessment experiments without requiring external API keys.
"""

import sys
from models import SBARClinicalPayload, TriageSeverity
from triage_guard import EmergencyTriageCircuitBreaker, PHIScrubber
from hybrid_retriever import ClinicalHybridRetriever, CLINICAL_CORPUS
from clinical_engine import ClinicalReasoningEngine


def run_demo():
    print("=" * 76)
    print("🩺 MODULE 08 CAPSTONE 02: CLINICAL MEDICAL CHATBOT VERIFICATION DEMO")
    print("=" * 76)

    # Step 1: Initialize Engine
    print("\n[Step 1] Initializing Clinical Reasoning Engine...")
    engine = ClinicalReasoningEngine(provider="auto")
    print(f"  ✅ Engine Initialized. Active Provider: '{engine.active_provider}'")

    # Step 2: PHI Redaction Verification
    print("\n[Step 2] Testing HIPAA Safe Harbor Ingress PHI Redaction...")
    raw_note = (
        "Patient Sarah Jenkins (DOB: 04/12/1982, SSN: 123-45-6789, MRN #8849201) "
        "called from (555) 234-5678 or sjenkins@medmail.org complaining of knee pain."
    )
    sanitized, redactions = PHIScrubber.sanitize(raw_note)
    print(f"  • Total PHI Elements Redacted: {redactions}")
    print(f"  • Sanitized Output: {sanitized}")
    assert "<SSN_REDACTED>" in sanitized, "Failed to redact SSN!"
    assert "<MRN_REDACTED>" in sanitized, "Failed to redact MRN!"
    assert "<PHONE_REDACTED>" in sanitized, "Failed to redact phone!"
    assert "<PATIENT_NAME_REDACTED>" in sanitized, "Failed to redact patient name!"
    print("  ✅ HIPAA PHI Ingress Firewall Verified.")

    # Step 3: Emergency Triage Circuit Breaker Interception
    print("\n[Step 3] Testing Acute Emergency Triage Circuit Breaker...")
    emergency_note = "Patient presents with crushing chest pain radiating to left arm and jaw, sweating profusely."
    triage_result = EmergencyTriageCircuitBreaker.evaluate(emergency_note)
    print(f"  • Is Emergency: {triage_result.is_emergency}")
    print(f"  • Severity: {triage_result.severity.value}")
    print(f"  • Detected Triggers: {triage_result.red_flag_triggers}")
    print(f"  • Emergency Directive: {triage_result.emergency_directive[:80]}...")
    assert triage_result.is_emergency is True, "Emergency circuit breaker failed to trigger!"
    assert triage_result.severity == TriageSeverity.CRITICAL, "Severity misclassified!"
    print("  ✅ Emergency Circuit Breaker Verified (Short-Circuited Safely).")

    # Step 4: Hybrid BM25 + Dense RRF Retrieval
    print("\n[Step 4] Testing Clinical Hybrid Retrieval (RRF k=60)...")
    retriever = ClinicalHybridRetriever(CLINICAL_CORPUS)
    results = retriever.retrieve_hybrid("Adult patient with community acquired pneumonia fever cough", top_k=2)
    print(f"  • Top Retrieved Guideline: {results[0][0].title}")
    print(f"  • RRF Combined Score: {results[0][1]:.5f}")
    assert "Pneumonia" in results[0][0].title, "Retriever failed to identify pneumonia guideline!"
    print("  ✅ Hybrid RRF Retrieval Verified.")

    # Step 5: Full Clinical Reasoning & SBAR Structured Synthesis
    print("\n[Step 5] Testing End-to-End Nominal Clinical Consultation...")
    nominal_note = (
        "Patient Emily Watson presents with 4 months of bilateral hand and wrist pain, "
        "accompanied by morning stiffness lasting 1 hour every morning."
    )
    triage, sbar, docs = engine.process_consultation(nominal_note)
    assert triage.is_emergency is False, "Nominal presentation falsely triggered emergency!"
    assert sbar is not None, "Failed to generate SBAR clinical assessment!"
    print(f"  • Clinical Situation: {sbar.situation}")
    print(f"  • Differential Diagnoses Generated: {len(sbar.assessment)}")
    for idx, diag in enumerate(sbar.assessment, start=1):
        print(f"    #{idx} {diag.condition_name} ({diag.icd10_code}) - Likelihood: {diag.likelihood}")
    print(f"  • Recommended Clinical Workup Items: {len(sbar.recommendations)}")
    print(f"  • Confidence Score: {sbar.confidence_score * 100:.1f}%")
    print(f"  • Statutory Disclaimer Enforced: {'YES ✅' if sbar.disclaimer else 'NO ❌'}")
    print("  ✅ Structured SBAR Clinical Diagnostic Engine Verified.")

    print("\n" + "=" * 76)
    print("🎉 ALL CLINICAL MEDICAL CHATBOT SYSTEMS VERIFIED SUCCESSFULLY!")
    print("=" * 76)


if __name__ == "__main__":
    run_demo()
