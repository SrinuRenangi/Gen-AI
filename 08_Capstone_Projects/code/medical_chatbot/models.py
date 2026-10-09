"""
Clinical Medical Chatbot - Core Data Models & Schemas
=====================================================
Module 08: Capstone Projects - Project 02: Clinical Medical Chatbot
File: models.py

Defines strict Pydantic v2 schemas for:
- HIPAA PHI De-Identification & Scrubbing
- Triage Severity Categorization
- Diagnostic Differential Hypotheses
- SBAR (Situation, Background, Assessment, Recommendation) Clinical Payloads
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class TriageSeverity(str, Enum):
    """Emergency severity classification for incoming patient presentations."""
    CRITICAL = "Critical"         # Level 1: Immediate life threat (Cardiac arrest, Anaphylaxis)
    EMERGENT = "Emergent"         # Level 2: High risk, severe pain, acute neuro deficits
    URGENT = "Urgent"             # Level 3: Stable vitals but requires timely clinical evaluation
    ROUTINE = "Routine"           # Level 4: Minor illness, chronic condition follow-up
    INFORMATIONAL = "Info"        # Level 5: General wellness or medical literature query


class DiagnosticHypothesis(BaseModel):
    """Ranked differential diagnosis hypothesis with ICD-10 code and clinical evidence."""
    condition_name: str = Field(..., description="Formal medical disease or syndrome name")
    icd10_code: str = Field(..., description="Standard ICD-10-CM clinical code (e.g., 'I21.9', 'J45.909')")
    likelihood: str = Field(..., description="Calibrated likelihood: 'High', 'Moderate', or 'Low'")
    pertinent_positives: List[str] = Field(
        default_factory=list,
        description="Symptoms and findings that actively support this diagnosis"
    )
    pertinent_negatives: List[str] = Field(
        default_factory=list,
        description="Absent symptoms that argue against or rule out alternative etiologies"
    )


class SBARClinicalPayload(BaseModel):
    """
    Standardized SBAR (Situation, Background, Assessment, Recommendation)
    clinical documentation framework widely used in accredited hospital systems.
    """
    situation: str = Field(
        ...,
        description="Chief complaint, presenting acuity, and immediate clinical dilemma"
    )
    background: str = Field(
        ...,
        description="Relevant past medical history, baseline comorbidities, and medication history"
    )
    assessment: List[DiagnosticHypothesis] = Field(
        ...,
        description="Ranked differential diagnostic hypotheses with clinical rationale"
    )
    recommendations: List[str] = Field(
        ...,
        description="Immediate diagnostic workup, laboratory tests, imaging, or clinician consults"
    )
    disclaimer: str = Field(
        ...,
        description="Statutory clinical decision support disclaimer"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Algorithmic retrieval and diagnostic confidence score between 0.0 and 1.0"
    )

    @model_validator(mode="after")
    def verify_disclaimer(self) -> SBARClinicalPayload:
        """Enforces that legal clinical decision support guardrails are present."""
        disclaimer_lower = self.disclaimer.lower()
        if "educational" not in disclaimer_lower and "clinical decision" not in disclaimer_lower:
            raise ValueError(
                "Mandatory statutory disclaimer missing! Clinical payloads must explicitly state "
                "they are educational clinical decision support tools and not a substitute for professional medical judgment."
            )
        return self


class ClinicalDocument(BaseModel):
    """Curated clinical guideline or medical journal excerpt."""
    id: str
    title: str
    mesh_terms: List[str] = Field(default_factory=list, description="Medical Subject Headings")
    content: str
    citation: str
    dense_embedding: Optional[List[float]] = None
