# Synthetic IT Project-Risk Benchmark Dataset Documentation

## 1. Overview
This dataset contains synthetic IT project health snapshots created specifically for the academic research project:
**"AI-Powered IT Project Risk Management Using a Multi-Agent Intelligence Framework"**.

> **IMPORTANT NOTICE:** This dataset is entirely synthetic and was procedurally generated for experimental benchmarking, multi-agent evaluation, and machine learning model validation. It must NOT be represented as real-world enterprise project observations.

- **File Name:** `synthetic_it_project_risk.csv`
- **Total Records:** 3,500
- **Total Features:** 19 (17 numerical, 2 categorical)
- **Target Feature:** `risk_category`
- **Random Seed:** `42`
- **Generation Method:** Domain-inspired parametric simulation with multivariate correlations and latent stochastic risk formulation.

## 2. Target Classes and Distribution
The classification target `risk_category` is divided into three distinct operational states:

| Class | Description | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| **On Track** | Project operating within acceptable schedule, cost, and quality bounds | 1,190 | 34.00% |
| **In Progress** | Project experiencing moderate schedule slip, budget drift, or issue accumulation | 1,330 | 38.00% |
| **At Risk** | Severe schedule delay, budget overruns, critical defects, or team distress requiring intervention | 980 | 28.00% |

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
   $$\text{CRI} = \left( w_s S_{delay} + w_b B_{variance} + w_q Q_{defects} + w_t T_{attrition} + w_r R_{scope} + w_k K_{satisfaction} \right) \times C_{complexity} + M_{comm} + \epsilon$$
   where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ introduces non-linear stochastic variance.
5. **Boundary Calibration:** Percentile thresholds partition the continuous latent index into balanced classes, ensuring overlapping distributions that replicate real-world classification complexity rather than a trivial deterministic boundary.

## 5. Limitations & Threats to Validity
- **Lack of Empirical Field Validation:** Generated records represent parameterized domain heuristics rather than longitudinal observational logs from enterprise Jira/GitHub environments.
- **Parametric Distributions:** Real IT projects often exhibit heavy-tailed distributions and sudden catastrophic step-changes (e.g., sudden bankruptcy or executive cancellation).
- **Intended Scope:** Strictly intended for benchmarking algorithmic stability, runtime efficiency, and multi-agent coordination.
