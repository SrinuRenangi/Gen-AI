"""
Enterprise Financial & Threat Intelligence Assistant - Data Models
====================================================================
Module 08: Capstone Projects - Project 03: Financial & Threat Intelligence
File: models.py

Defines strict Pydantic v2 schemas for:
- SEC EDGAR Financial Disclosures & Solvency Metrics
- Cybersecurity Threat Intelligence (CVEs, IoCs, MITRE ATT&CK)
- Multi-Agent ReAct Reasoning Traces
- Aggregated Enterprise Risk Synthesis Reports
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class RiskSeverity(str, Enum):
    """Calibrated enterprise risk categorization."""
    CRITICAL = "Critical"       # Immediate existential solvency or active APT breach threat
    HIGH = "High"               # Substantial material risk, significant vulnerability exposure
    MEDIUM = "Medium"           # Manageable operational or regulatory risk
    LOW = "Low"                 # Nominal variance, minor routine patch or disclosure
    NEUTRAL = "Neutral"         # Stable baseline


class SECFilingSection(BaseModel):
    """Structured segment of an SEC 10-K or 10-Q annual/quarterly filing."""
    ticker: str
    filing_type: str = Field(description="'10-K' Annual or '10-Q' Quarterly report")
    fiscal_year: int
    item_title: str = Field(description="e.g. 'Item 1A: Risk Factors' or 'Item 7: MD&A'")
    content_snippet: str
    citation_url: str


class ThreatEntity(BaseModel):
    """Cybersecurity threat intelligence indicator and exposure profile."""
    ioc_type: str = Field(description="'IP', 'Domain', 'CVE', 'FileHash', or 'ThreatActor'")
    value: str = Field(description="Actual IoC string (e.g., 'CVE-2024-3400', 'LockBit 3.0')")
    mitre_attack_id: Optional[str] = Field(default=None, description="e.g., 'T1190: Exploit Public-Facing App'")
    severity: RiskSeverity
    description: str


class SolvencyMetrics(BaseModel):
    """Fundamental quantitative financial ratios and Altman Z-Score."""
    ticker: str
    working_capital_to_assets: float
    retained_earnings_to_assets: float
    ebit_to_assets: float
    market_equity_to_liabilities: float
    sales_to_assets: float
    altman_z_score: float = Field(
        description="Altman Z-Score: > 2.99 Safe Zone, 1.81 - 2.99 Grey Zone, < 1.81 Distress Zone"
    )
    distress_category: str = Field(description="'Safe Zone', 'Grey Zone', or 'Distress Zone'")


class AgentStep(BaseModel):
    """Single step in a ReAct (Reasoning + Acting) execution trajectory."""
    iteration: int
    thought: str = Field(description="The agent's internal chain-of-thought rationale")
    action: Optional[str] = Field(default=None, description="The name of the tool selected")
    action_input: Optional[Dict[str, Any]] = Field(default=None, description="Parameters passed to the tool")
    observation: Optional[str] = Field(default=None, description="Output returned from the tool execution")


class EnterpriseRiskReport(BaseModel):
    """
    Executive synthesis report combining quantitative financial health
    and qualitative cybersecurity threat intelligence.
    """
    target_ticker: str
    company_name: str
    report_timestamp: str
    executive_summary: str
    financial_health_verdict: str
    altman_metrics: SolvencyMetrics
    cyber_threat_exposure: List[ThreatEntity]
    mitre_attack_vectors: List[str]
    compliance_recommendations: List[str]
    overall_risk_score: float = Field(ge=0.0, le=100.0, description="Risk index 0 (Safe) to 100 (Extreme Danger)")
    agent_trajectory: List[AgentStep] = Field(default_factory=list)
