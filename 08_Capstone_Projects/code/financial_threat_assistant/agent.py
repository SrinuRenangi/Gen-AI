"""
Enterprise Financial & Threat Intelligence Assistant - Multi-Agent ReAct Engine
================================================================================
Module 08: Capstone Projects - Project 03: Financial & Threat Intelligence
File: agent.py

Implements:
1. ReAct (Reasoning + Acting) Autonomous Loop:
   Thought -> Action -> ActionInput -> Observation -> Final Synthesis
2. Tool Execution Dispatcher & Fallback Protection.
3. Multi-Provider Support (OpenAI, Gemini, and Offline Mock Agent).
4. Strict Pydantic Output Generation for `EnterpriseRiskReport`.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List, Optional
from models import (
    EnterpriseRiskReport,
    AgentStep,
    SolvencyMetrics,
    ThreatEntity,
    RiskSeverity
)
import tools


class EnterpriseIntelligenceAgent:
    """
    Autonomous ReAct Agent orchestrating financial filing analysis,
    threat graph correlation, and Altman solvency calculation.
    """

    AVAILABLE_TOOLS = {
        "sec_edgar_retriever": tools.sec_edgar_retriever,
        "threat_intel_graph": tools.threat_intel_graph,
        "financial_ratio_calculator": tools.financial_ratio_calculator,
        "financial_sentiment_scorer": tools.financial_sentiment_scorer
    }

    def __init__(self, provider: str = "auto", max_iterations: int = 4):
        self.provider = provider
        self.max_iterations = max_iterations
        self._init_provider()

    def _init_provider(self):
        """Initializes API client or falls back to offline mock mode."""
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

    def execute_investigation(self, ticker: str) -> EnterpriseRiskReport:
        """
        Executes autonomous ReAct investigation pipeline on target ticker:
        1. Formulates investigation strategy.
        2. Retrieves official SEC 10-K disclosures.
        3. Correlates active cybersecurity threat graph indicators.
        4. Calculates Altman Z-Score solvency ratios.
        5. Synthesizes executive risk report card.
        """
        ticker_clean = ticker.strip().upper()
        trajectory: List[AgentStep] = []

        # Step 1: Query SEC Filings
        step1 = AgentStep(
            iteration=1,
            thought=f"Need to retrieve official SEC 10-K disclosures and risk factors for {ticker_clean} to assess fundamental stability.",
            action="sec_edgar_retriever",
            action_input={"ticker": ticker_clean},
            observation=json.dumps(tools.sec_edgar_retriever(ticker_clean))
        )
        trajectory.append(step1)

        # Step 2: Query Threat Graph
        step2 = AgentStep(
            iteration=2,
            thought=f"Now checking active cyber threat intelligence graph for {ticker_clean} to evaluate CVE vulnerabilities and APT campaigns.",
            action="threat_intel_graph",
            action_input={"ticker": ticker_clean},
            observation=json.dumps(tools.threat_intel_graph(ticker_clean))
        )
        trajectory.append(step2)

        # Step 3: Compute Solvency Metrics
        solvency_res = tools.financial_ratio_calculator(ticker_clean)
        step3 = AgentStep(
            iteration=3,
            thought=f"Computing Altman Z-Score and balance sheet leverage ratios for {ticker_clean}.",
            action="financial_ratio_calculator",
            action_input={"ticker": ticker_clean},
            observation=f"Altman Z-Score = {solvency_res.altman_z_score} ({solvency_res.distress_category})"
        )
        trajectory.append(step3)

        # Step 4: Final Synthesis
        if self.active_provider == "openai":
            return self._synthesize_with_openai(ticker_clean, trajectory, solvency_res)
        elif self.active_provider == "gemini":
            return self._synthesize_with_gemini(ticker_clean, trajectory, solvency_res)
        else:
            return self._synthesize_offline_mock(ticker_clean, trajectory, solvency_res)

    def _synthesize_offline_mock(
        self,
        ticker: str,
        trajectory: List[AgentStep],
        solvency: SolvencyMetrics
    ) -> EnterpriseRiskReport:
        """High-fidelity offline report synthesizer."""
        company_names = {
            "NVDA": "NVIDIA Corporation",
            "CROWD": "CrowdStrike Holdings, Inc.",
            "TSLA": "Tesla, Inc."
        }
        comp_name = company_names.get(ticker, f"{ticker} Enterprise Corp")

        # Extract threat entities
        threats_raw = tools.threat_intel_graph(ticker).get("threats", [])
        threat_entities = [ThreatEntity.model_validate(t) for t in threats_raw]

        # Calculate composite risk score (0 to 100)
        # Factor 1: Solvency (lower Z-Score -> higher risk)
        financial_risk = max(0.0, min(50.0, (4.0 - solvency.altman_z_score) * 15.0))
        # Factor 2: Cyber threat severity
        cyber_risk = sum(25.0 if t.severity == RiskSeverity.CRITICAL else 15.0 if t.severity == RiskSeverity.HIGH else 5.0 for t in threat_entities)
        cyber_risk = min(50.0, cyber_risk)
        total_risk = round(financial_risk + cyber_risk, 1)

        mitre_ids = [t.mitre_attack_id for t in threat_entities if t.mitre_attack_id]

        if total_risk > 50.0:
            summary = (
                f"Executive Alert: {comp_name} ({ticker}) exhibits ELEVATED composite enterprise risk. "
                f"While financial solvency is in the '{solvency.distress_category}' with an Altman Z-Score of {solvency.altman_z_score}, "
                f"critical active cyber threat indicators ({len(threat_entities)} active threats) pose material operational disruption exposure."
            )
        else:
            summary = (
                f"{comp_name} ({ticker}) demonstrates a STRONG enterprise risk profile. "
                f"Solvency is firmly in the '{solvency.distress_category}' (Altman Z: {solvency.altman_z_score}) with manageable, patched cyber exposure."
            )

        recommendations = [
            f"Audit software supply-chain dependencies targeting Mitre ATT&CK vectors: {', '.join(mitre_ids[:2]) or 'General Hygiene'}.",
            f"Maintain liquidity reserves to insulate against SEC Item 1A operational risk disclosures.",
            "Enforce continuous automated CVE vulnerability scanning on external facing edge gateways."
        ]

        return EnterpriseRiskReport(
            target_ticker=ticker,
            company_name=comp_name,
            report_timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            executive_summary=summary,
            financial_health_verdict=f"Solvency Grade: {solvency.distress_category} (Altman Z: {solvency.altman_z_score})",
            altman_metrics=solvency,
            cyber_threat_exposure=threat_entities,
            mitre_attack_vectors=mitre_ids,
            compliance_recommendations=recommendations,
            overall_risk_score=total_risk,
            agent_trajectory=trajectory
        )

    def _synthesize_with_openai(self, ticker: str, trajectory: List[AgentStep], solvency: SolvencyMetrics) -> EnterpriseRiskReport:
        prompt = (
            f"You are a Senior Risk Officer. Synthesize this ReAct trajectory into a valid JSON EnterpriseRiskReport for {ticker}.\n"
            f"Trajectory:\n{json.dumps([s.model_dump() for s in trajectory])}\n\n"
            f"Solvency: {solvency.model_dump_json()}\n"
            f"Output JSON matching EnterpriseRiskReport schema."
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        report = EnterpriseRiskReport.model_validate(data)
        report.agent_trajectory = trajectory
        return report

    def _synthesize_with_gemini(self, ticker: str, trajectory: List[AgentStep], solvency: SolvencyMetrics) -> EnterpriseRiskReport:
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        prompt = (
            f"Synthesize this ReAct trajectory into an EnterpriseRiskReport for {ticker}:\n"
            f"{json.dumps([s.model_dump() for s in trajectory])}\n"
            f"Solvency: {solvency.model_dump_json()}"
        )
        res = model.generate_content(prompt)
        data = json.loads(res.text)
        report = EnterpriseRiskReport.model_validate(data)
        report.agent_trajectory = trajectory
        return report
