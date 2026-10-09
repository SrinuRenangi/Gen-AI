"""
Enterprise Financial & Threat Intelligence Assistant - Verification Demo
========================================================================
Module 08: Capstone Projects - Project 03: Financial & Threat Intelligence
File: demo.py

Automated test harness verifying the multi-agent ReAct loop, tool execution,
Altman Z-Score calculation, and executive risk report generation.
"""

from agent import EnterpriseIntelligenceAgent
import tools


def run_demo():
    print("=" * 80)
    print("🛡️ MODULE 08 CAPSTONE 03: FINANCIAL & THREAT INTELLIGENCE VERIFICATION DEMO")
    print("=" * 80)

    # Step 1: Tool Verification - Financial Ratio Calculator
    print("\n[Step 1] Testing Altman Z-Score Financial Ratio Calculator...")
    solvency_nvda = tools.financial_ratio_calculator("NVDA")
    print(f"  • NVDA Altman Z-Score: {solvency_nvda.altman_z_score}")
    print(f"  • NVDA Distress Status: {solvency_nvda.distress_category}")
    assert solvency_nvda.altman_z_score > 2.99, "NVDA should be in Safe Zone!"
    assert solvency_nvda.distress_category == "Safe Zone"
    print("  ✅ Solvency Calculator Verified.")

    # Step 2: Tool Verification - SEC EDGAR Filing Retrieval
    print("\n[Step 2] Testing SEC 10-K Disclosures Retrieval...")
    filing_res = tools.sec_edgar_retriever("NVDA")
    assert filing_res["status"] == "success"
    print(f"  • Ticker: {filing_res['ticker']}")
    print(f"  • Total Filing Sections Indexed: {len(filing_res['filings'])}")
    print(f"  • Sample Section: {filing_res['filings'][0]['item_title']}")
    print("  ✅ SEC Filing Retriever Verified.")

    # Step 3: Tool Verification - Threat Intelligence Graph
    print("\n[Step 3] Testing Cybersecurity Threat Graph Correlation...")
    threat_res = tools.threat_intel_graph("CROWD")
    assert threat_res["status"] == "success"
    print(f"  • Target: {threat_res['ticker']}")
    print(f"  • Threat Count: {len(threat_res['threats'])}")
    for t in threat_res["threats"]:
        print(f"    - [{t['ioc_type']}] {t['value']} ({t['severity']}) -> {t['mitre_attack_id']}")
    print("  ✅ Cyber Threat Graph Verified.")

    # Step 4: Autonomous ReAct Agent Investigation
    print("\n[Step 4] Launching Autonomous Multi-Agent ReAct Investigation on 'CROWD'...")
    agent = EnterpriseIntelligenceAgent(provider="auto")
    print(f"  • Active Inference Core: '{agent.active_provider}'")

    report = agent.execute_investigation("CROWD")
    print(f"\n  ✅ Investigation Concluded! Generated Executive Report for: {report.company_name}")
    print(f"  • Overall Composite Risk Score: {report.overall_risk_score} / 100")
    print(f"  • Financial Solvency Verdict: {report.financial_health_verdict}")
    print(f"  • Total ReAct Trajectory Steps: {len(report.agent_trajectory)}")
    for step in report.agent_trajectory:
        print(f"    [Step {step.iteration}] Thought: {step.thought[:65]}...")
        print(f"             Action: {step.action}")

    print(f"\n  • Executive Summary:\n    {report.executive_summary[:140]}...")
    print(f"  • Compliance Directives ({len(report.compliance_recommendations)}):")
    for r in report.compliance_recommendations:
        print(f"    📌 {r}")

    assert report.target_ticker == "CROWD"
    assert len(report.cyber_threat_exposure) > 0
    assert len(report.agent_trajectory) >= 3

    print("\n" + "=" * 80)
    print("🎉 ALL FINANCIAL & THREAT INTELLIGENCE SYSTEMS VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
