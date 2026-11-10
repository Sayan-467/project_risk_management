"""
status_tracker.py
=================
Project Status Tracking Agent.

Analyzes internal project indicators:
- Schedule progress vs baseline
- Defect accumulation and test pass rate
- Resource allocation and team availability
- Scope changes and milestone completion

Produces structured observations and internal risk flags.
"""

import time
from typing import Dict, Any, List


class ProjectStatusTrackingAgent:
    """
    Evaluates internal project health indicators and detects operational anomalies.
    """
    def __init__(self):
        self.name = "Project Status Tracking Agent"
        self.role = "Internal Project Health Monitor"

    def analyze_status(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze structured indicators for a single project record.
        """
        start_t = time.perf_counter()

        observations = []
        risk_flags = []
        severity_score = 0.0

        # 1. Schedule Tracking
        delay_days = project_data.get('schedule_delay_days', 0.0)
        planned_pct = project_data.get('planned_completion_percentage', 0.0)
        actual_pct = project_data.get('project_completion_percentage', 0.0)
        
        if delay_days > 25.0 or (planned_pct - actual_pct) > 15.0:
            flag = f"Critical schedule slippage detected: {delay_days:.1f} days behind schedule."
            observations.append(flag)
            risk_flags.append("Schedule Delay")
            severity_score += 25.0
        elif delay_days > 10.0:
            observations.append(f"Moderate schedule delay observed ({delay_days:.1f} days).")
            risk_flags.append("Moderate Schedule Delay")
            severity_score += 12.0
        else:
            observations.append("Schedule performance is within acceptable variance.")

        # 2. Budget & Variance Tracking
        budget_var = project_data.get('budget_variance_percentage', 0.0)
        budget_util = project_data.get('budget_utilization_percentage', 0.0)
        
        if budget_var > 20.0 or budget_util > 115.0:
            flag = f"Budget overrun alert: +{budget_var:.1f}% variance above planned burn rate."
            observations.append(flag)
            risk_flags.append("Budget Overrun")
            severity_score += 22.0
        elif budget_var > 8.0:
            observations.append(f"Minor budget drift noticed (+{budget_var:.1f}%).")
            severity_score += 10.0
        else:
            observations.append("Budget consumption aligns with earned progress.")

        # 3. Quality & Defect Tracking
        critical_bugs = project_data.get('critical_issue_count', 0)
        test_pass_rate = project_data.get('test_pass_percentage', 100.0)
        bug_count = project_data.get('bug_count', 0)

        if critical_bugs >= 5 or test_pass_rate < 75.0:
            flag = f"Severe quality degradation: {critical_bugs} critical blockers; test pass rate {test_pass_rate:.1f}%."
            observations.append(flag)
            risk_flags.append("Quality Blocker")
            severity_score += 25.0
        elif critical_bugs >= 2 or test_pass_rate < 85.0:
            observations.append(f"Quality warning: {critical_bugs} critical defects logged.")
            risk_flags.append("Defect Accumulation")
            severity_score += 12.0
        else:
            observations.append("Quality metrics and automated test suites remain healthy.")

        # 4. Team Capacity & Attrition
        attrition = project_data.get('team_attrition_rate', 0.0)
        availability = project_data.get('team_availability_percentage', 100.0)

        if attrition > 18.0 or availability < 65.0:
            flag = f"Team stability distress: {attrition:.1f}% attrition; availability at {availability:.1f}%."
            observations.append(flag)
            risk_flags.append("Staff Turnover")
            severity_score += 20.0
        elif attrition > 10.0:
            observations.append(f"Elevated team turnover ({attrition:.1f}%).")
            severity_score += 8.0

        # 5. Scope Volatility
        scope_changes = project_data.get('requirement_change_count', 0)
        if scope_changes > 15:
            flag = f"Severe scope creep: {scope_changes} requirement change requests."
            observations.append(flag)
            risk_flags.append("Scope Creep")
            severity_score += 15.0

        execution_time_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "agent": self.name,
            "observations": observations,
            "risk_flags": risk_flags,
            "internal_severity_score": min(100.0, severity_score),
            "execution_time_ms": round(execution_time_ms, 3)
        }
