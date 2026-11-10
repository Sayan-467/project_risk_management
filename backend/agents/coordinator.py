"""
coordinator.py
==============
Project Risk Manager (Coordinator Agent).

Orchestrates the complete multi-agent workflow:
Project Risk Manager -> Status Tracking Agent / Market Analysis Agent
                     -> Historical Knowledge Retrieval
                     -> Risk Scoring Agent
                     -> Reporting Agent

Collects high-resolution performance metrics (time.perf_counter()) for each agent
and end-to-end execution.
"""

import time
from typing import Dict, Any, Optional

from backend.agents.status_tracker import ProjectStatusTrackingAgent
from backend.agents.market_analyst import MarketAnalysisAgent
from backend.agents.risk_scorer import RiskScoringAgent
from backend.agents.reporter import ReportingAgent
from backend.retrieval.historical_store import HistoricalRiskKnowledgeStore


class ProjectRiskManager:
    """
    Coordinator agent responsible for multi-agent dispatch, workflow timing, and result aggregation.
    """
    def __init__(
        self,
        model_path: str = "backend/ml/models/it_risk_model.pkl",
        preprocessor_path: str = "backend/ml/models/preprocessor.pkl",
        label_encoder_path: str = "backend/ml/models/label_encoder.pkl"
    ):
        init_start = time.perf_counter()
        self.name = "Project Risk Manager"
        self.role = "Workflow Coordinator"

        # Instantiate specialized agents
        self.status_agent = ProjectStatusTrackingAgent()
        self.market_agent = MarketAnalysisAgent()
        self.retrieval_store = HistoricalRiskKnowledgeStore()
        self.scoring_agent = RiskScoringAgent(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            label_encoder_path=label_encoder_path
        )
        self.reporting_agent = ReportingAgent()

        self.initialization_time_ms = round((time.perf_counter() - init_start) * 1000.0, 3)

    def assess_project(
        self,
        project_data: Dict[str, Any],
        project_id: str = "PRJ-001",
        communication_text: str = "",
        variant: str = "complete"
    ) -> Dict[str, Any]:
        """
        Execute full coordinated risk assessment workflow.

        Parameters
        ----------
        project_data : dict
            Structured numeric and categorical features of the project snapshot.
        project_id : str
            Identifier for reporting.
        communication_text : str
            Unstructured communication notes / feedback text.
        variant : str
            Ablation variant: 'structured_only', 'structured_nlp',
            'structured_historical', or 'complete'.

        Returns
        -------
        dict containing:
            - report: Comprehensive generated risk report
            - scoring: Risk scoring outputs and probability breakdown
            - component_timings: Exact latency in milliseconds for each agent
            - total_latency_ms: Total end-to-end workflow execution latency
        """
        workflow_start = time.perf_counter()
        timings = {}

        # 1. Project Status Tracking Agent
        t_status_start = time.perf_counter()
        status_output = self.status_agent.analyze_status(project_data)
        timings["Project Status Tracking Agent"] = round((time.perf_counter() - t_status_start) * 1000.0, 3)

        # 2. Market Analysis Agent (NLP & Qualitative Signals)
        t_market_start = time.perf_counter()
        if variant in ("structured_nlp", "complete"):
            comm_health = project_data.get('communication_health', 'Good')
            stakeholder_sat = project_data.get('stakeholder_satisfaction', 4.0)
            market_output = self.market_agent.analyze_market_context(
                communication_text=communication_text,
                communication_health=comm_health,
                stakeholder_satisfaction=stakeholder_sat
            )
        else:
            market_output = {
                "agent": "Market Analysis Agent (Bypassed)",
                "sentiment_analysis": {"sentiment": "Neutral", "polarity_score": 0.0},
                "external_risk_score": 20.0,
                "observations": ["Market analysis bypassed in this configuration."]
            }
        timings["Market Analysis Agent"] = round((time.perf_counter() - t_market_start) * 1000.0, 3)

        # 3. Historical Knowledge Retrieval
        t_retrieval_start = time.perf_counter()
        if variant in ("structured_historical", "complete"):
            # Construct semantic search query from detected risks
            query_parts = []
            if project_data.get('schedule_delay_days', 0) > 15:
                query_parts.append("schedule delay")
            if project_data.get('budget_variance_percentage', 0) > 15:
                query_parts.append("budget overrun")
            if project_data.get('team_attrition_rate', 0) > 12:
                query_parts.append("team attrition")
            if project_data.get('critical_issue_count', 0) > 3:
                query_parts.append("defect blocker quality")
            
            query_str = " ".join(query_parts) if query_parts else "software project development risk management"
            retrieval_output = self.retrieval_store.query(query_str, top_k=2)
        else:
            retrieval_output = {
                "matches": [],
                "latency_ms": 0.0,
                "engine": "Historical Retrieval (Bypassed)"
            }
        timings["Historical Knowledge Retrieval"] = round((time.perf_counter() - t_retrieval_start) * 1000.0, 3)

        # 4. Risk Scoring Agent
        t_score_start = time.perf_counter()
        scoring_output = self.scoring_agent.compute_risk_score(
            project_features=project_data,
            status_output=status_output,
            market_output=market_output,
            retrieval_output=retrieval_output,
            variant=variant
        )
        timings["Risk Scoring Agent"] = round((time.perf_counter() - t_score_start) * 1000.0, 3)

        # 5. Reporting Agent
        t_report_start = time.perf_counter()
        report_output = self.reporting_agent.generate_report(
            project_id=project_id,
            scoring_output=scoring_output,
            status_output=status_output,
            market_output=market_output,
            retrieval_output=retrieval_output
        )
        timings["Reporting Agent"] = round((time.perf_counter() - t_report_start) * 1000.0, 3)

        # 6. Coordinator Aggregation Overhead
        total_latency = round((time.perf_counter() - workflow_start) * 1000.0, 3)
        coordinator_overhead = round(total_latency - sum(timings.values()), 3)
        timings["Project Risk Manager (Coordinator)"] = max(0.001, coordinator_overhead)

        return {
            "project_id": project_id,
            "report": report_output,
            "scoring": scoring_output,
            "status": status_output,
            "market": market_output,
            "retrieval": retrieval_output,
            "component_timings_ms": timings,
            "total_latency_ms": total_latency,
            "initialization_time_ms": self.initialization_time_ms
        }
