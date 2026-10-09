"""
Clinical Medical Chatbot - Ingress Safety Firewall & Emergency Triage
======================================================================
Module 08: Capstone Projects - Project 02: Clinical Medical Chatbot
File: triage_guard.py

Implements:
1. HIPAA Safe Harbor PHI De-Identification & Sanitizer.
2. Acute Emergency Triage Circuit Breaker (Deterministic Red-Flag Interceptor).
"""

from __future__ import annotations
import re
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from models import TriageSeverity


@dataclass
class TriageAssessment:
    """Outcome of safety firewall and triage circuit breaker inspection."""
    is_emergency: bool
    severity: TriageSeverity
    red_flag_triggers: List[str]
    emergency_directive: Optional[str]
    sanitized_text: str
    redacted_phi_count: int


class PHIScrubber:
    """
    Ingress safety firewall scrubbing HIPAA Safe Harbor identifiers:
    - Social Security Numbers (SSN)
    - Medical Record Numbers (MRN)
    - Telephone & Fax numbers
    - Email addresses
    - Full dates (birth dates, admission dates)
    - Known patient identifiers / names
    """

    PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "<SSN_REDACTED>"),
        (r"\bMRN\s*#?\s*\d{6,10}\b", "<MRN_REDACTED>"),
        (r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "<PHONE_REDACTED>"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "<EMAIL_REDACTED>"),
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "<DATE_REDACTED>"),
        (r"\b(?:DOB|Date of Birth):\s*\S+", "<DOB_REDACTED>"),
    ]

    KNOWN_PATIENT_NAMES = [
        "Sarah Jenkins", "Robert Davis", "John Smith", "David Miller",
        "Emily Watson", "Michael Brown", "Jessica Taylor", "William Garcia"
    ]

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, int]:
        """
        Scrub structured and unstructured PHI from incoming clinical text.
        Returns: (sanitized_text, count_of_redactions)
        """
        redactions = 0
        sanitized = text

        for pattern, replacement in cls.PATTERNS:
            matches = len(re.findall(pattern, sanitized, re.IGNORECASE))
            if matches > 0:
                redactions += matches
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        for name in cls.KNOWN_PATIENT_NAMES:
            if name.lower() in sanitized.lower():
                sanitized = re.sub(re.escape(name), "<PATIENT_NAME_REDACTED>", sanitized, flags=re.IGNORECASE)
                redactions += 1

        return sanitized, redactions


class EmergencyTriageCircuitBreaker:
    """
    Deterministic safety circuit breaker that intercepts acute life threats
    prior to any LLM invocation. Prevents conversational latency in emergencies.
    """

    RED_FLAG_CONDITIONS: Dict[str, Dict[str, Any]] = {
        "Acute Coronary Syndrome / MI": {
            "keywords": [
                r"\bcrushing chest pain\b", r"\bradiating to (?:the )?(?:left )?arm\b",
                r"\bradiating to jaw\b", r"\bsubsternal pressure\b", r"\bheart attack\b"
            ],
            "severity": TriageSeverity.CRITICAL,
            "directive": (
                "🚨 CRITICAL EMERGENCY DETECTED: Symptoms are consistent with Acute Myocardial Infarction. "
                "CALL 911 / LOCAL EMERGENCY MEDICAL SERVICES IMMEDIATELY. "
                "Have the patient rest, chew 325mg non-enteric coated aspirin if not contraindicated, "
                "and prepare for immediate emergency transport."
            )
        },
        "Acute Ischemic Stroke (FAST)": {
            "keywords": [
                r"\bfacial droop\b", r"\barm weakness\b", r"\bslurred speech\b",
                r"\bsudden numbness\b", r"\bworst headache of (?:my )?life\b", r"\bthunderclap headache\b"
            ],
            "severity": TriageSeverity.CRITICAL,
            "directive": (
                "🚨 CRITICAL EMERGENCY DETECTED: Acute neurological deficit meets FAST criteria for Stroke. "
                "CALL 911 IMMEDIATELY. Record the exact time symptoms began. Do NOT administer food, liquids, or medication."
            )
        },
        "Anaphylaxis / Airway Compromise": {
            "keywords": [
                r"\bthroat (?:closing|tightening)\b", r"\bunable to breathe\b",
                r"\bstridor\b", r"\bsevere wheezing\b", r"\blip swelling\b"
            ],
            "severity": TriageSeverity.CRITICAL,
            "directive": (
                "🚨 CRITICAL EMERGENCY DETECTED: Imminent airway compromise or anaphylaxis. "
                "Administer autoinjector epinephrine (EpiPen) immediately into the anterolateral thigh. "
                "CALL 911 IMMEDIATELY."
            )
        },
        "Acute Hemorrhage / Shock": {
            "keywords": [
                r"\buncontrolled bleeding\b", r"\bhemorrhage\b", r"\bcoughing up large amounts of blood\b"
            ],
            "severity": TriageSeverity.EMERGENT,
            "directive": (
                "⚠️ EMERGENT CLINICAL SITUATION: Massive active hemorrhage. "
                "Apply direct continuous pressure with sterile gauze. Call 911 immediately."
            )
        },
        "Acute Psychiatric Crisis / Suicidal Ideation": {
            "keywords": [
                r"\bwant to kill myself\b", r"\bsuicidal\b", r"\bend my life\b", r"\boverdose on purpose\b"
            ],
            "severity": TriageSeverity.CRITICAL,
            "directive": (
                "🚨 CRISIS INTERVENTION REQUIRED: Please reach out immediately to the National Suicide & Crisis Lifeline: "
                "Dial or text 988 (USA/Canada), or go to the nearest emergency department. You are not alone."
            )
        }
    }

    @classmethod
    def evaluate(cls, raw_text: str) -> TriageAssessment:
        """
        Runs both the PHI scrubber and the deterministic red-flag triage rules.
        """
        sanitized_text, redacted_count = PHIScrubber.sanitize(raw_text)

        detected_triggers: List[str] = []
        highest_severity = TriageSeverity.ROUTINE
        emergency_directive = None

        text_lower = sanitized_text.lower()

        for condition, config in cls.RED_FLAG_CONDITIONS.items():
            for kw_pattern in config["keywords"]:
                if re.search(kw_pattern, text_lower):
                    detected_triggers.append(condition)
                    if config["severity"] == TriageSeverity.CRITICAL:
                        highest_severity = TriageSeverity.CRITICAL
                        emergency_directive = config["directive"]
                    elif config["severity"] == TriageSeverity.EMERGENT and highest_severity != TriageSeverity.CRITICAL:
                        highest_severity = TriageSeverity.EMERGENT
                        emergency_directive = config["directive"]
                    break

        is_emergency = len(detected_triggers) > 0 and highest_severity in (TriageSeverity.CRITICAL, TriageSeverity.EMERGENT)

        return TriageAssessment(
            is_emergency=is_emergency,
            severity=highest_severity if is_emergency else TriageSeverity.ROUTINE,
            red_flag_triggers=detected_triggers,
            emergency_directive=emergency_directive,
            sanitized_text=sanitized_text,
            redacted_phi_count=redacted_count
        )
