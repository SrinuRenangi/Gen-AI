"""
Clinical Medical Chatbot - Core Clinical Reasoning Engine
==========================================================
Module 08: Capstone Projects - Project 02: Clinical Medical Chatbot
File: clinical_engine.py

Orchestrates:
1. Ingress PHI De-Identification & Emergency Circuit Breaker.
2. Hybrid Clinical Guideline Retrieval (Dense + BM25 via RRF).
3. SBAR Diagnostic Reasoning Engine (OpenAI, Gemini, or Offline Mock).
4. Strict Pydantic Schema Validation & Disclaimer Enforcement.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from models import (
    SBARClinicalPayload,
    DiagnosticHypothesis,
    TriageSeverity,
    ClinicalDocument
)
from triage_guard import EmergencyTriageCircuitBreaker, TriageAssessment
from hybrid_retriever import ClinicalHybridRetriever, CLINICAL_CORPUS


STATUTORY_DISCLAIMER = (
    "DISCLAIMER: This system is an AI-powered educational Clinical Decision Support (CDS) tool. "
    "It is NOT a certified medical device and does NOT provide medical diagnoses or treatment plans. "
    "All recommendations must be evaluated and verified by a licensed healthcare professional before clinical application."
)


class ClinicalReasoningEngine:
    """
    Production-grade Clinical Decision Support orchestrator with deterministic
    safety guards, hybrid retrieval, and SBAR-structured output generation.
    """

    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.retriever = ClinicalHybridRetriever(CLINICAL_CORPUS)
        self._init_provider()

    def _init_provider(self):
        """Initializes API client or falls back to offline clinical mode."""
        self.active_provider = "mock"
        if self.provider in ("openai", "auto") and os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self.openai_client = openai.OpenAI()
                self.active_provider = "openai"
                return
            except ImportError:
                pass

        if self.provider in ("gemini", "auto") and os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ["GEMINI_API_KEY"])
                self.active_provider = "gemini"
                return
            except ImportError:
                pass

        self.active_provider = "mock"

    def process_consultation(self, patient_note: str) -> Tuple[TriageAssessment, Optional[SBARClinicalPayload], List[ClinicalDocument]]:
        """
        End-to-End Clinical Processing:
        1. Ingress PHI Scrubbing & Emergency Triage Evaluation.
        2. Short-circuit if emergency detected.
        3. Hybrid Evidence Retrieval.
        4. Structured SBAR Reasoning & Validation.
        """
        # Step 1: Ingress Triage & PHI Sanitization
        triage_result = EmergencyTriageCircuitBreaker.evaluate(patient_note)

        if triage_result.is_emergency:
            # Short-circuit immediately for patient safety
            return triage_result, None, []

        # Step 2: Evidence Retrieval
        retrieved_docs_with_scores = self.retriever.retrieve_hybrid(triage_result.sanitized_text, top_k=2)
        retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]

        # Step 3: Diagnostic Reasoning & SBAR Synthesis
        if self.active_provider == "openai":
            payload = self._reason_with_openai(triage_result.sanitized_text, retrieved_docs)
        elif self.active_provider == "gemini":
            payload = self._reason_with_gemini(triage_result.sanitized_text, retrieved_docs)
        else:
            payload = self._reason_offline_mock(triage_result.sanitized_text, retrieved_docs)

        return triage_result, payload, retrieved_docs

    def _reason_offline_mock(self, sanitized_text: str, evidence: List[ClinicalDocument]) -> SBARClinicalPayload:
        """
        Deterministic, offline clinical reasoning engine. Provides high-fidelity
        diagnostic differential hypotheses based on recognized symptom clusters.
        """
        text_lower = sanitized_text.lower()

        # Clinical Case 1: Joint pain & morning stiffness -> Rheumatoid Arthritis
        if "joint" in text_lower or "arthralgia" in text_lower or "stiffness" in text_lower:
            return SBARClinicalPayload(
                situation="Patient presents with persistent bilateral joint pain and morning stiffness.",
                background="No prior autoimmune diagnoses. Patient reports stiffness lasting > 45 minutes every morning.",
                assessment=[
                    DiagnosticHypothesis(
                        condition_name="Rheumatoid Arthritis",
                        icd10_code="M05.79",
                        likelihood="High",
                        pertinent_positives=["Bilateral symmetric joint pain", "Morning stiffness > 45 minutes", "Small joint involvement"],
                        pertinent_negatives=["Absence of acute monoarticular podagra (argues against acute gout)", "Absence of malar rash (argues against SLE)"]
                    ),
                    DiagnosticHypothesis(
                        condition_name="Osteoarthritis",
                        icd10_code="M15.9",
                        likelihood="Moderate",
                        pertinent_positives=["Chronic articular pain worsened with activity"],
                        pertinent_negatives=["Prolonged morning stiffness typically < 30 minutes in primary OA"]
                    ),
                    DiagnosticHypothesis(
                        condition_name="Psoriatic Arthritis",
                        icd10_code="L40.50",
                        likelihood="Low",
                        pertinent_positives=["Inflammatory joint complaints"],
                        pertinent_negatives=["Absence of cutaneous psoriasis or dactylitis"]
                    )
                ],
                recommendations=[
                    "Order serum Rheumatoid Factor (RF) and anti-Cyclic Citrullinated Peptide (anti-CCP) antibodies.",
                    "Order baseline Inflammatory Markers: ESR and High-Sensitivity CRP.",
                    "Obtain bilateral bilateral hand and wrist radiographs to evaluate for periarticular osteopenia or marginal erosions.",
                    "Referral to Rheumatology for formal disease activity assessment (CDAI/DAS28) and potential csDMARD initiation (Methotrexate)."
                ],
                disclaimer=STATUTORY_DISCLAIMER,
                confidence_score=0.91
            )

        # Clinical Case 2: Dyspnea, cough, fever -> Community-Acquired Pneumonia
        elif "cough" in text_lower or "fever" in text_lower or "sputum" in text_lower:
            return SBARClinicalPayload(
                situation="Patient presents with acute productive cough, fever, and mild pleuritic chest discomfort.",
                background="Previous history of mild seasonal allergies. Non-smoker.",
                assessment=[
                    DiagnosticHypothesis(
                        condition_name="Community-Acquired Bacterial Pneumonia",
                        icd10_code="J15.9",
                        likelihood="High",
                        pertinent_positives=["Productive cough with purulent sputum", "Fever and pleuritic discomfort", "Subacute onset"],
                        pertinent_negatives=["Absence of calf swelling or hemoptysis (argues against Pulmonary Embolism)"]
                    ),
                    DiagnosticHypothesis(
                        condition_name="Acute Bronchitis",
                        icd10_code="J20.9",
                        likelihood="Moderate",
                        pertinent_positives=["Cough and upper airway irritation"],
                        pertinent_negatives=["Absence of focal pulmonary consolidations on physical exam"]
                    )
                ],
                recommendations=[
                    "Obtain Posteroanterior and Lateral Chest Radiography (CXR) to confirm focal pulmonary infiltrate.",
                    "Calculate CURB-65 or Pneumonia Severity Index (PSI) score to determine outpatient vs inpatient disposition.",
                    "Empiric outpatient antibiotic coverage per ATS/IDSA guidelines: Amoxicillin 1g TID or Doxycycline 100mg BID if uncomplicated."
                ],
                disclaimer=STATUTORY_DISCLAIMER,
                confidence_score=0.88
            )

        # Default clinical catch-all
        return SBARClinicalPayload(
            situation=f"Clinical evaluation requested for presenting symptoms: {sanitized_text[:120]}...",
            background="Comprehensive baseline clinical history review pending.",
            assessment=[
                DiagnosticHypothesis(
                    condition_name="Undifferentiated Clinical Presentation",
                    icd10_code="R69",
                    likelihood="Moderate",
                    pertinent_positives=["Reported symptom constellation"],
                    pertinent_negatives=["No acute organ failure identified in baseline description"]
                )
            ],
            recommendations=[
                "Perform comprehensive in-person history and targeted physical examination.",
                "Obtain baseline complete metabolic panel (CMP) and complete blood count (CBC).",
                "Correlate findings with retrieved clinical practice guidelines."
            ],
            disclaimer=STATUTORY_DISCLAIMER,
            confidence_score=0.75
        )

    def _reason_with_openai(self, sanitized_text: str, evidence: List[ClinicalDocument]) -> SBARClinicalPayload:
        """Invokes OpenAI with structured JSON outputs."""
        evidence_str = "\n".join([f"[{d.id}] {d.title}: {d.content}" for d in evidence])
        prompt = (
            f"You are a clinical decision support assistant. Analyze this patient presentation using the provided evidence.\n"
            f"Patient Presentation: {sanitized_text}\n\n"
            f"Clinical Evidence:\n{evidence_str}\n\n"
            f"Return a valid JSON object matching the SBARClinicalPayload schema. Ensure the disclaimer explicitly includes "
            f"the phrase 'educational clinical decision support'."
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        return SBARClinicalPayload.model_validate(data)

    def _reason_with_gemini(self, sanitized_text: str, evidence: List[ClinicalDocument]) -> SBARClinicalPayload:
        """Invokes Google Gemini with JSON schema output."""
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        evidence_str = "\n".join([f"[{d.id}] {d.title}: {d.content}" for d in evidence])
        prompt = (
            f"You are a clinical decision support assistant. Analyze this presentation: {sanitized_text}\n"
            f"Evidence:\n{evidence_str}\n"
            f"Output JSON conforming to SBAR with situation, background, assessment (list of condition_name, icd10_code, likelihood, pertinent_positives, pertinent_negatives), recommendations, disclaimer, and confidence_score."
        )
        response = model.generate_content(prompt)
        data = json.loads(response.text)
        return SBARClinicalPayload.model_validate(data)
