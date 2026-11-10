"""
data_generator.py
=================
Synthetic IT Project-Risk Benchmark Dataset Generator.

Generates realistic, domain-inspired IT project risk snapshots for experimental
evaluation of machine learning and multi-agent risk assessment pipelines.

Author: Syed Sayan et al. (AI-Powered IT Project Risk Management Research)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_it_project_risk_dataset(
    n_samples: int = 3500,
    random_seed: int = 42,
    introduce_missingness: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic IT project snapshots with realistic multivariate correlations,
    domain-inspired composite risk indicators, and class boundary noise.

    Parameters
    ----------
    n_samples : int
        Number of synthetic project records to generate (default: 3500).
    random_seed : int
        Seed for pseudo-random number generator for reproducibility (default: 42).
    introduce_missingness : bool
        Whether to introduce realistic missing values in a small fraction of records.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 19 project features and 1 target variable ('risk_category').
    """
    rng = np.random.RandomState(random_seed)

    # 1. Project Duration & Timeline
    project_duration_days = rng.randint(60, 540, size=n_samples)
    
    # Actual completion percentage (5% to 98%)
    project_completion_percentage = np.round(rng.uniform(5.0, 98.0, size=n_samples), 2)
    
    # Planned completion percentage (often higher than actual when delayed, occasionally lower when ahead)
    schedule_slip_intent = rng.normal(loc=6.0, scale=12.0, size=n_samples)
    planned_completion_percentage = np.clip(
        np.round(project_completion_percentage + schedule_slip_intent, 2),
        5.0, 100.0
    )
    
    # Schedule delay days: proportional to slip and total duration
    delay_ratio = (planned_completion_percentage - project_completion_percentage) / 100.0
    base_delay = delay_ratio * project_duration_days
    schedule_delay_days = np.clip(
        np.round(np.maximum(0.0, base_delay + rng.normal(0, 5, size=n_samples)), 1),
        0.0, 180.0
    )

    # Remaining duration days
    remaining_ratio = np.maximum(0.0, 1.0 - (project_completion_percentage / 100.0))
    remaining_duration_days = np.round(
        np.maximum(0.0, remaining_ratio * project_duration_days + rng.normal(0, 7, size=n_samples)), 0
    ).astype(int)

    # 2. Budget Indicators
    # Budget variance percentage (-30% under budget to +80% over budget)
    cost_creep = delay_ratio * 35.0 + rng.normal(loc=4.0, scale=14.0, size=n_samples)
    budget_variance_percentage = np.clip(np.round(cost_creep, 2), -30.0, 95.0)

    # Budget utilization percentage
    budget_utilization_percentage = np.clip(
        np.round(project_completion_percentage + budget_variance_percentage + rng.normal(0, 4, size=n_samples), 2),
        5.0, 180.0
    )

    # 3. Resource & Team Availability
    resource_utilization_percentage = np.clip(
        np.round(rng.normal(loc=86.0, scale=16.0, size=n_samples), 2),
        40.0, 150.0
    )
    team_availability_percentage = np.clip(
        np.round(rng.normal(loc=84.0, scale=11.0, size=n_samples), 2),
        45.0, 100.0
    )
    team_attrition_rate = np.clip(
        np.round(rng.exponential(scale=6.5, size=n_samples), 2),
        0.0, 40.0
    )

    # 4. Issue and Defect Quality Indicators
    # Unresolved issues scale with project size, schedule delay, and random noise
    base_issues = 8 + (schedule_delay_days * 0.25) + rng.poisson(lam=12, size=n_samples)
    unresolved_issue_count = np.clip(base_issues, 0, 95).astype(int)

    # Critical issues are typically 10-25% of unresolved issues
    critical_prob = rng.uniform(0.08, 0.25, size=n_samples)
    critical_issue_count = np.clip(
        rng.binomial(unresolved_issue_count, critical_prob),
        0, 22
    ).astype(int)

    # Bug count
    base_bugs = (unresolved_issue_count * 1.5) + rng.poisson(lam=25, size=n_samples)
    bug_count = np.clip(base_bugs, 0, 240).astype(int)

    # Test pass percentage: inversely related to bug count and critical issues
    quality_drag = (bug_count * 0.12) + (critical_issue_count * 1.8)
    test_pass_percentage = np.clip(
        np.round(98.0 - quality_drag + rng.normal(0, 4, size=n_samples), 2),
        45.0, 99.8
    )

    # 5. Milestone & Requirement Management
    milestone_ratio = (project_completion_percentage / 100.0) * rng.uniform(0.85, 1.05, size=n_samples)
    milestone_completion_percentage = np.clip(
        np.round(milestone_ratio * 100.0, 2),
        0.0, 100.0
    )
    requirement_change_count = rng.poisson(lam=7.0, size=n_samples)
    requirement_change_count = np.clip(requirement_change_count, 0, 32).astype(int)

    # 6. Categorical Contextual Features
    complexity_choices = ['Low', 'Medium', 'High', 'Critical']
    complexity_weights = [0.25, 0.40, 0.25, 0.10]
    project_complexity = rng.choice(complexity_choices, size=n_samples, p=complexity_weights)

    communication_choices = ['Poor', 'Fair', 'Good', 'Excellent']
    communication_weights = [0.15, 0.30, 0.35, 0.20]
    communication_health = rng.choice(communication_choices, size=n_samples, p=communication_weights)

    # Stakeholder satisfaction (1.0 to 5.0 scale)
    base_satisfaction = 4.2 - (schedule_delay_days * 0.015) - (budget_variance_percentage * 0.012)
    stakeholder_satisfaction = np.clip(
        np.round(base_satisfaction + rng.normal(0, 0.45, size=n_samples), 2),
        1.0, 5.0
    )

    # 7. Composite Latent Risk Score (CRI) Formulation
    # Domain-inspired multi-factor latent index combining schedule, budget, quality, and human factors
    complexity_map = {'Low': 0.85, 'Medium': 1.0, 'High': 1.25, 'Critical': 1.55}
    communication_map = {'Poor': 1.4, 'Fair': 0.5, 'Good': -0.4, 'Excellent': -1.1}

    complexity_factor = np.array([complexity_map[c] for c in project_complexity])
    comm_modifier = np.array([communication_map[ch] for ch in communication_health])

    # Normalized component risk scores
    schedule_risk = np.clip(schedule_delay_days / 28.0, 0.0, 3.5)
    budget_risk = np.clip(budget_variance_percentage / 22.0, -1.0, 3.5)
    quality_risk = (critical_issue_count * 0.28) + ((100.0 - test_pass_percentage) / 18.0)
    team_risk = (team_attrition_rate / 12.0) + np.maximum(0.0, (80.0 - team_availability_percentage) / 15.0)
    scope_risk = requirement_change_count / 10.0
    satisfaction_risk = (5.0 - stakeholder_satisfaction) * 0.7

    # Raw Latent Risk Index with stochastic Gaussian noise
    noise = rng.normal(loc=0.0, scale=0.85, size=n_samples)
    latent_risk_index = (
        (0.24 * schedule_risk +
         0.22 * budget_risk +
         0.20 * quality_risk +
         0.14 * team_risk +
         0.10 * scope_risk +
         0.10 * satisfaction_risk) * complexity_factor +
        0.18 * comm_modifier +
        noise
    )

    # 8. Calibration to 3 Balanced Classes
    # Class 0: On Track (~34%)
    # Class 1: In Progress (~38%)
    # Class 2: At Risk (~28%)
    t_lower = np.percentile(latent_risk_index, 34.0)
    t_upper = np.percentile(latent_risk_index, 72.0)

    risk_category = []
    for val in latent_risk_index:
        if val < t_lower:
            risk_category.append('On Track')
        elif val < t_upper:
            risk_category.append('In Progress')
        else:
            risk_category.append('At Risk')

    # Construct DataFrame
    df = pd.DataFrame({
        'project_completion_percentage': project_completion_percentage,
        'planned_completion_percentage': planned_completion_percentage,
        'schedule_delay_days': schedule_delay_days,
        'budget_utilization_percentage': budget_utilization_percentage,
        'budget_variance_percentage': budget_variance_percentage,
        'resource_utilization_percentage': resource_utilization_percentage,
        'team_availability_percentage': team_availability_percentage,
        'unresolved_issue_count': unresolved_issue_count,
        'critical_issue_count': critical_issue_count,
        'bug_count': bug_count,
        'test_pass_percentage': test_pass_percentage,
        'milestone_completion_percentage': milestone_completion_percentage,
        'requirement_change_count': requirement_change_count,
        'team_attrition_rate': team_attrition_rate,
        'project_complexity': project_complexity,
        'stakeholder_satisfaction': stakeholder_satisfaction,
        'communication_health': communication_health,
        'project_duration_days': project_duration_days,
        'remaining_duration_days': remaining_duration_days,
        'risk_category': risk_category
    })

    # 9. Realistic Missing Data Injection
    if introduce_missingness:
        # Inject 1.5% to 2.5% missingness into 3 non-target columns
        mask_team = rng.rand(n_samples) < 0.02
        df.loc[mask_team, 'team_availability_percentage'] = np.nan

        mask_sat = rng.rand(n_samples) < 0.025
        df.loc[mask_sat, 'stakeholder_satisfaction'] = np.nan

        mask_pass = rng.rand(n_samples) < 0.015
        df.loc[mask_pass, 'test_pass_percentage'] = np.nan

    return df


def generate_evaluation_scenarios(
    n_scenarios: int = 100,
    random_seed: int = 2026
) -> pd.DataFrame:
    """
    Generate an independent synthetic evaluation scenario set for end-to-end multi-agent evaluation.
    Uses a separate random seed to prevent data overlap.

    Parameters
    ----------
    n_scenarios : int
        Number of project scenarios (default: 100).
    random_seed : int
        Seed for evaluation set (default: 2026).

    Returns
    -------
    pd.DataFrame
        DataFrame of evaluation scenarios with scenario_id and ground_truth label.
    """
    df = generate_it_project_risk_dataset(
        n_samples=n_scenarios,
        random_seed=random_seed,
        introduce_missingness=False
    )
    df.insert(0, 'scenario_id', [f'SCENARIO-{i+1:03d}' for i in range(n_scenarios)])
    return df


def save_dataset_and_documentation(
    df: pd.DataFrame,
    csv_path: str = "data/synthetic_it_project_risk.csv",
    doc_path: str = "data/dataset_documentation.md",
    random_seed: int = 42
):
    """
    Save generated dataset to CSV and generate comprehensive dataset documentation.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[+] Dataset saved to {csv_path} (Shape: {df.shape})")

    # Documentation Generation
    class_counts = df['risk_category'].value_counts()
    class_pcts = df['risk_category'].value_counts(normalize=True) * 100

    doc_content = f"""# Synthetic IT Project-Risk Benchmark Dataset Documentation

## 1. Overview
This dataset contains synthetic IT project health snapshots created specifically for the academic research project:
**"AI-Powered IT Project Risk Management Using a Multi-Agent Intelligence Framework"**.

> **IMPORTANT NOTICE:** This dataset is entirely synthetic and was procedurally generated for experimental benchmarking, multi-agent evaluation, and machine learning model validation. It must NOT be represented as real-world enterprise project observations.

- **File Name:** `synthetic_it_project_risk.csv`
- **Total Records:** {len(df):,}
- **Total Features:** {df.shape[1] - 1} (17 numerical, 2 categorical)
- **Target Feature:** `risk_category`
- **Random Seed:** `{random_seed}`
- **Generation Method:** Domain-inspired parametric simulation with multivariate correlations and latent stochastic risk formulation.

## 2. Target Classes and Distribution
The classification target `risk_category` is divided into three distinct operational states:

| Class | Description | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| **On Track** | Project operating within acceptable schedule, cost, and quality bounds | {class_counts.get('On Track', 0):,} | {class_pcts.get('On Track', 0.0):.2f}% |
| **In Progress** | Project experiencing moderate schedule slip, budget drift, or issue accumulation | {class_counts.get('In Progress', 0):,} | {class_pcts.get('In Progress', 0.0):.2f}% |
| **At Risk** | Severe schedule delay, budget overruns, critical defects, or team distress requiring intervention | {class_counts.get('At Risk', 0):,} | {class_pcts.get('At Risk', 0.0):.2f}% |

## 3. Feature Dictionary

| Feature Name | Data Type | Range / Domain | Description |
| :--- | :--- | :--- | :--- |
| `project_completion_percentage` | Float | 5.0% - 98.0% | Actual percentage of work completed to date. |
| `planned_completion_percentage` | Float | 5.0% - 100.0% | Baseline planned completion percentage according to project schedule. |
| `schedule_delay_days` | Float | 0.0 - 180.0 days | Computed schedule slip based on planned vs actual completion. |
| `budget_utilization_percentage` | Float | 5.0% - 180.0% | Cumulative budget consumed relative to allocated baseline. |
| `budget_variance_percentage` | Float | -30.0% to +95.0% | Variance between consumed budget and earned project progress. |
| `resource_utilization_percentage` | Float | 40.0% - 150.0% | Workload allocation rate of assigned project engineering resources. |
| `team_availability_percentage` | Float | 45.0% - 100.0% | Effective engineering capacity available (accounting for leave/unavailability). |
| `unresolved_issue_count` | Integer | 0 - 95 | Total active tickets/blockers logged in tracking system. |
| `critical_issue_count` | Integer | 0 - 22 | High-severity blocker issues that halt key milestones. |
| `bug_count` | Integer | 0 - 240 | Cumulative open defect count across QA cycles. |
| `test_pass_percentage` | Float | 45.0% - 99.8% | Automated test suite execution passing rate. |
| `milestone_completion_percentage` | Float | 0.0% - 100.0% | Ratio of contractual project milestones formally signed off. |
| `requirement_change_count` | Integer | 0 - 32 | Number of approved scope or requirement change requests. |
| `team_attrition_rate` | Float | 0.0% - 40.0% | Turnover rate of key engineering and leadership personnel. |
| `project_complexity` | Categorical | Low, Medium, High, Critical | Structural complexity and architectural novelty of the IT project. |
| `stakeholder_satisfaction` | Float | 1.0 - 5.0 | Survey score measuring client and sponsor satisfaction. |
| `communication_health` | Categorical | Poor, Fair, Good, Excellent | Metric measuring transparency and cadence across project teams. |
| `project_duration_days` | Integer | 60 - 540 days | Planned duration of the overall project lifecycle. |
| `remaining_duration_days` | Integer | 0 - 540 days | Estimated remaining timeline to scheduled delivery. |

## 4. Synthetic Data Generation Methodology
1. **Multi-variable Correlation:** Project delay is modeled as a function of the divergence between planned and actual progress, amplified by project duration.
2. **Defect & Quality Dynamics:** Test pass rate decreases with defect density and critical blocker accumulation.
3. **Budget Slip:** Budget variance is modeled with positive covariance with schedule delay (representing burn rate during project extensions).
4. **Latent Composite Risk Score (CRI):**
   $$\\text{{CRI}} = \\left( w_s S_{{delay}} + w_b B_{{variance}} + w_q Q_{{defects}} + w_t T_{{attrition}} + w_r R_{{scope}} + w_k K_{{satisfaction}} \\right) \\times C_{{complexity}} + M_{{comm}} + \\epsilon$$
   where $\\epsilon \\sim \\mathcal{{N}}(0, \\sigma^2)$ introduces non-linear stochastic variance.
5. **Boundary Calibration:** Percentile thresholds partition the continuous latent index into balanced classes, ensuring overlapping distributions that replicate real-world classification complexity rather than a trivial deterministic boundary.

## 5. Limitations & Threats to Validity
- **Lack of Empirical Field Validation:** Generated records represent parameterized domain heuristics rather than longitudinal observational logs from enterprise Jira/GitHub environments.
- **Parametric Distributions:** Real IT projects often exhibit heavy-tailed distributions and sudden catastrophic step-changes (e.g., sudden bankruptcy or executive cancellation).
- **Intended Scope:** Strictly intended for benchmarking algorithmic stability, runtime efficiency, and multi-agent coordination.
"""
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content)
    print(f"[+] Dataset documentation saved to {doc_path}")


if __name__ == "__main__":
    df_data = generate_it_project_risk_dataset(n_samples=3500, random_seed=42)
    save_dataset_and_documentation(df_data)
