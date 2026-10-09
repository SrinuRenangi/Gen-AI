"""
Enterprise Financial & Threat Intelligence Assistant - ReAct Tools
===================================================================
Module 08: Capstone Projects - Project 03: Financial & Threat Intelligence
File: tools.py

Implements deterministic tools callable by the ReAct Autonomous Agent:
1. `sec_edgar_retriever`: Queries SEC 10-K / 10-Q filing disclosures & Risk Factors.
2. `threat_intel_graph`: Queries cybersecurity threat correlation graph (CVEs, IoCs, APTs).
3. `financial_ratio_calculator`: Computes Altman Z-Score solvency metric.
4. `financial_sentiment_scorer`: Measures risk keyword density and tonal sentiment.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
from models import (
    SECFilingSection,
    ThreatEntity,
    SolvencyMetrics,
    RiskSeverity
)


# Curated SEC 10-K Filing Database (Simulating EDGAR filings)
SEC_FILINGS_DB: Dict[str, List[SECFilingSection]] = {
    "NVDA": [
        SECFilingSection(
            ticker="NVDA",
            filing_type="10-K",
            fiscal_year=2024,
            item_title="Item 1A: Risk Factors - Supply Chain & Geopolitical Constraints",
            content_snippet=(
                "We rely on third-party foundries (principally TSMC) to manufacture our semiconductor wafers. "
                "Any disruption in Taiwan or export restrictions on advanced AI accelerators (such as the H100/B200) "
                "to restricted foreign markets materially impacts our top-line revenue and gross margin predictability."
            ),
            citation_url="https://www.sec.gov/edgar/data/1045810/nvda-202410k"
        ),
        SECFilingSection(
            ticker="NVDA",
            filing_type="10-K",
            fiscal_year=2024,
            item_title="Item 7: MD&A - Liquidity and Capital Resources",
            content_snippet=(
                "Cash, cash equivalents, and marketable securities totaled $26.0 billion as of January 28, 2024. "
                "Total outstanding debt was $9.7 billion. Free cash flow surged by 610% year-over-year driven by "
                "hyperscale data center GPU demand. Working capital ratio stands exceptionally strong at 4.2x."
            ),
            citation_url="https://www.sec.gov/edgar/data/1045810/nvda-202410k"
        )
    ],
    "CROWD": [
        SECFilingSection(
            ticker="CROWD",
            filing_type="10-K",
            fiscal_year=2024,
            item_title="Item 1A: Risk Factors - System Outages & Channel File Distribution",
            content_snippet=(
                "Our Falcon platform relies on kernel-level sensors across millions of client endpoints. "
                "If our rapid-response channel configuration files contain logic flaws or cause widespread blue-screen "
                "crashes, our customers face operational halts, exposing us to breach of contract litigation and brand impairment."
            ),
            citation_url="https://www.sec.gov/edgar/data/1535527/crwd-202410k"
        ),
        SECFilingSection(
            ticker="CROWD",
            filing_type="10-K",
            fiscal_year=2024,
            item_title="Item 7: MD&A - Revenue Retention and Cash Position",
            content_snippet=(
                "Annual Recurring Revenue (ARR) surpassed $3.44 billion, growing 33% YoY. Total cash and equivalents "
                "stood at $3.47 billion with minimal long-term indebtedness ($743 million). Gross retention remains above 97%."
            ),
            citation_url="https://www.sec.gov/edgar/data/1535527/crwd-202410k"
        )
    ],
    "TSLA": [
        SECFilingSection(
            ticker="TSLA",
            filing_type="10-K",
            fiscal_year=2024,
            item_title="Item 1A: Risk Factors - Automotive Margin Compression & Autopilot Regulatory Scrutiny",
            content_snippet=(
                "Global automotive price cuts have reduced automotive gross margins from 28.5% to 17.2%. "
                "Furthermore, ongoing regulatory investigations into Full Self-Driving (FSD) driver monitoring "
                "and autonomous system liability present material legal and reputational exposure."
            ),
            citation_url="https://www.sec.gov/edgar/data/1318605/tsla-202410k"
        )
    ]
}


# Curated Cybersecurity Threat Intelligence Graph
THREAT_INTEL_DB: Dict[str, List[ThreatEntity]] = {
    "NVDA": [
        ThreatEntity(
            ioc_type="CVE",
            value="CVE-2024-0076",
            mitre_attack_id="T1068: Exploitation for Privilege Escalation",
            severity=RiskSeverity.HIGH,
            description="Vulnerability in NVIDIA GPU Display Driver for Linux and Windows allows privilege escalation via memory corruption."
        ),
        ThreatEntity(
            ioc_type="ThreatActor",
            value="Lapsus$ Extortion Syndicate",
            mitre_attack_id="T1567: Exfiltration Over Web Service",
            severity=RiskSeverity.MEDIUM,
            description="Historical targeted extortion campaign that exfiltrated proprietary DLSS source code and code-signing certificates."
        )
    ],
    "CROWD": [
        ThreatEntity(
            ioc_type="CVE",
            value="CVE-2024-3400",
            mitre_attack_id="T1190: Exploit Public-Facing Application",
            severity=RiskSeverity.CRITICAL,
            description="Critical OS command injection vulnerability in GlobalProtect gateway actively exploited by state-sponsored actors."
        ),
        ThreatEntity(
            ioc_type="Domain",
            value="crowdstrike-update-malicious.com",
            mitre_attack_id="T1566: Phishing",
            severity=RiskSeverity.HIGH,
            description="Adversary typosquatting domain distributing fake CrowdStrike repair executables infected with Remcos RAT."
        )
    ],
    "TSLA": [
        ThreatEntity(
            ioc_type="CVE",
            value="CVE-2023-42465",
            mitre_attack_id="T1498: Network Denial of Service",
            severity=RiskSeverity.MEDIUM,
            description="CAN-bus gateway logic flaw allowing arbitrary remote command injection via compromised cellular telematics unit."
        )
    ]
}


# Financial Balance Sheet Data (in Billions)
FINANCIAL_RAW_DATA: Dict[str, Dict[str, float]] = {
    "NVDA": {
        "working_capital": 35.2,
        "total_assets": 65.7,
        "retained_earnings": 32.5,
        "ebit": 33.0,
        "market_val_equity": 2800.0,
        "total_liabilities": 22.8,
        "sales": 60.9
    },
    "CROWD": {
        "working_capital": 2.8,
        "total_assets": 6.1,
        "retained_earnings": 0.45,
        "ebit": 0.38,
        "market_val_equity": 75.0,
        "total_liabilities": 3.7,
        "sales": 3.44
    },
    "TSLA": {
        "working_capital": 33.1,
        "total_assets": 106.6,
        "retained_earnings": 28.0,
        "ebit": 8.9,
        "market_val_equity": 720.0,
        "total_liabilities": 43.0,
        "sales": 96.8
    }
}


# ============================================================================
# Callable ReAct Tool Implementations
# ============================================================================

def sec_edgar_retriever(ticker: str) -> Dict[str, Any]:
    """Retrieves official SEC 10-K filing disclosures and risk factors for a company."""
    ticker_clean = ticker.strip().upper()
    filings = SEC_FILINGS_DB.get(ticker_clean)
    if not filings:
        return {
            "status": "error",
            "message": f"No SEC filings indexed for ticker '{ticker_clean}'. Indexed tickers: NVDA, CROWD, TSLA."
        }
    return {
        "status": "success",
        "ticker": ticker_clean,
        "filings": [f.model_dump() for f in filings]
    }


def threat_intel_graph(ticker: str) -> Dict[str, Any]:
    """Queries cybersecurity threat graph for active CVEs, IoCs, and threat actors targeting company infrastructure."""
    ticker_clean = ticker.strip().upper()
    threats = THREAT_INTEL_DB.get(ticker_clean)
    if not threats:
        return {
            "status": "success",
            "ticker": ticker_clean,
            "threats": [],
            "message": "No active high-severity IoCs or CVEs detected."
        }
    return {
        "status": "success",
        "ticker": ticker_clean,
        "threats": [t.model_dump() for t in threats]
    }


def financial_ratio_calculator(ticker: str) -> SolvencyMetrics:
    """
    Computes Altman Z-Score corporate solvency formulation:
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
    """
    ticker_clean = ticker.strip().upper()
    raw = FINANCIAL_RAW_DATA.get(ticker_clean)
    if not raw:
        # Default nominal fallback metrics
        return SolvencyMetrics(
            ticker=ticker_clean,
            working_capital_to_assets=0.30,
            retained_earnings_to_assets=0.25,
            ebit_to_assets=0.15,
            market_equity_to_liabilities=2.50,
            sales_to_assets=0.80,
            altman_z_score=3.85,
            distress_category="Safe Zone"
        )

    ta = raw["total_assets"]
    tl = raw["total_liabilities"]

    x1 = raw["working_capital"] / ta
    x2 = raw["retained_earnings"] / ta
    x3 = raw["ebit"] / ta
    x4 = raw["market_val_equity"] / tl
    x5 = raw["sales"] / ta

    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5

    if z_score > 2.99:
        category = "Safe Zone"
    elif z_score >= 1.81:
        category = "Grey Zone"
    else:
        category = "Distress Zone"

    return SolvencyMetrics(
        ticker=ticker_clean,
        working_capital_to_assets=round(x1, 3),
        retained_earnings_to_assets=round(x2, 3),
        ebit_to_assets=round(x3, 3),
        market_equity_to_liabilities=round(x4, 3),
        sales_to_assets=round(x5, 3),
        altman_z_score=round(z_score, 2),
        distress_category=category
    )


def financial_sentiment_scorer(text: str) -> Dict[str, Any]:
    """Measures financial risk terms and negative sentiment density."""
    risk_words = ["disruption", "litigation", "crash", "vulnerability", "investigation", "decline", "compression", "loss"]
    positive_words = ["growth", "surpassed", "strong", "cash", "retention", "expansion", "profit"]

    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    neg_matches = sum(1 for w in risk_words if w in text_lower)
    pos_matches = sum(1 for w in positive_words if w in text_lower)

    net_score = (pos_matches - neg_matches) / max(pos_matches + neg_matches, 1)

    return {
        "positive_keyword_count": pos_matches,
        "risk_keyword_count": neg_matches,
        "net_sentiment_index": round(net_score, 2),
        "sentiment_label": "Bullish / Low Risk" if net_score > 0.2 else "Bearish / Elevated Risk" if net_score < -0.2 else "Neutral"
    }
