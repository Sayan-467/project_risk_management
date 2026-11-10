"""
reporter.py
===========
Reporting Agent.

Synthesizes the comprehensive multi-agent findings into a structured,
actionable project risk assessment report.
"""

import time
from typing import Dict, Any, List


class ReportingAgent:
    """
    Transforms multi-agent analytic findings into a human-readable risk report.
    """
    def __init__(self):
        self.name = "Reporting Agent"
        self.role = "Structured Assessment & Mitigation Synthesizer"

    def generate_report(
        self,
        project_id: str,
        scoring_output: Dict[str, Any],
        status_output: Dict[str, Any],
        market_output: Dict[str, Any],
        retrieval_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate structured risk report including key drivers and mitigations.
        """
        start_t = time.perf_counter()

        category = scoring_output.get('risk_category', 'In Progress')
        score = scoring_output.get('risk_score', 50.0)

        # 1. Synthesize Risk Factors
        risk_factors = list(status_output.get('risk_flags', []))
        market_obs = market_output.get('observations', [])
        sentiment_info = market_output.get('sentiment_analysis', {})
        if sentiment_info.get('sentiment') == 'Negative':
            risk_factors.append("Adverse Communication Sentiment")

        if not risk_factors:
            risk_factors.append("Nominal variance within operational tolerances")

        # 2. Synthesize Actionable Mitigations
        mitigations = []
        # Incorporate historical case mitigations
        matches = retrieval_output.get('matches', [])
        for m in matches[:2]:
            mitigations.append(f"Historical Precedent ({m['title']}): {m['mitigation_strategy']}")

        # Context-specific mitigations
        if "Schedule Delay" in risk_factors or "Moderate Schedule Delay" in risk_factors:
            mitigations.append("Schedule Re-baselining: De-scope non-critical requirements and prioritize critical path deliverables.")
        if "Budget Overrun" in risk_factors:
            mitigations.append("Cost Governance: Enforce weekly sprint burn-rate reviews and freeze discretionary compute resources.")
        if "Quality Blocker" in risk_factors or "Defect Accumulation" in risk_factors:
            mitigations.append("Defect Burn-down: Dedicate immediate sprint cycles strictly to regression fixes before new feature delivery.")
        if "Staff Turnover" in risk_factors:
            mitigations.append("Knowledge Transfer: Pair senior developers with junior maintainers and conduct workload redistribution.")

        if not mitigations:
            mitigations.append("Maintain existing sprint cadence and continuous automated integration testing.")

        # 3. Build Executive Summary
        summary = (
            f"Project {project_id} is evaluated as '{category}' with a consolidated risk score of {score:.1f}/100. "
            f"Primary factors include {', '.join(risk_factors[:3])}."
        )

        execution_time_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "agent": self.name,
            "project_id": project_id,
            "executive_summary": summary,
            "risk_score": score,
            "risk_category": category,
            "identified_risk_factors": risk_factors,
            "supporting_observations": status_output.get('observations', []) + market_obs,
            "prioritized_mitigations": mitigations[:4],
            "execution_time_ms": round(execution_time_ms, 3)
        }
