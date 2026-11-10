# Final Quantitative Research Statistics & Statistical Analysis

**Project Title:** AI-Powered IT Project Risk Management Using a Multi-Agent Intelligence Framework  
**Affiliation:** Department of Computer Science & Engineering, Institute of Engineering and Management, Kolkata  
**Target Paper Sections:** Section V (Experimental Setup) & Section VI (Results & Discussion)  
**Status:** Post-Audit Verified & Empirically Validated  

---

## 1. Dataset Statistical Analysis (Section V - Benchmark Dataset)

The benchmark dataset consists of procedural IT project health snapshots modeled with multi-factor non-linear latent correlations and stochastic boundary noise.

### Table 1.1: Dataset Composition & Quality Overview

| Dimension / Property | Measured Value | Academic Description |
| :--- | :---: | :--- |
| **Total Project Snapshots** | **3,500** | Synthetically generated project lifecycle snapshots |
| **Total Input Features** | **19** | 17 continuous numerical metrics, 2 multi-level categorical factors |
| **Numerical Features** | 17 | Schedule, cost, defect, team capacity, and scope variables |
| **Categorical Features** | 2 | `project_complexity` (4 levels), `communication_health` (4 levels) |
| **Target Classes** | 3 | `At Risk` (0), `In Progress` (1), `On Track` (2) |
| **Missing Values Injected** | 226 (0.34%) | Realistic missingness: `stakeholder_satisfaction` (101), `team_availability` (68), `test_pass_rate` (57) |
| **Exact Duplicate Records** | **0** | Verified zero exact duplicate rows across all features |
| **Random Generator Seed** | 42 | Fixed pseudo-random seed for full procedural reproducibility |

### Table 1.2: Class Distribution & Stratified Partitions

| Risk Category Class | Full Dataset ($N=3,500$) | Train Partition (80%, $N=2,800$) | Test Partition (20%, $N=700$) | Operational Risk Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **At Risk (0)** | **980 (28.00%)** | 784 (28.00%) | 196 (28.00%) | Severe schedule slip, budget overrun, blocker defects |
| **In Progress (1)** | **1,330 (38.00%)** | 1,064 (38.00%) | 266 (38.00%) | Moderate operational variance within recoverable margins |
| **On Track (2)** | **1,190 (34.00%)** | 952 (34.00%) | 238 (34.00%) | Normal trajectory matching planned baseline deliverables |

### Table 1.3: Descriptive Statistics for Numerical Project Indicators ($N = 3,500$)

| Numerical Feature | Mean | Median | Std Dev | Min | Q1 (25%) | Q3 (75%) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `project_completion_percentage` (%) | 51.10 | 50.81 | 26.73 | 5.00 | 28.32 | 74.03 | 97.96 |
| `planned_completion_percentage` (%) | 56.24 | 56.51 | 28.03 | 5.00 | 34.08 | 80.08 | 100.00 |
| `schedule_delay_days` (days) | 22.88 | 11.60 | 29.41 | 0.00 | 0.00 | 35.12 | 180.00 |
| `budget_utilization_percentage` (%) | 57.58 | 57.26 | 29.87 | 5.00 | 34.36 | 80.82 | 166.74 |
| `budget_variance_percentage` (%) | 6.18 | 6.13 | 14.41 | -30.00 | -3.55 | 16.14 | 66.45 |
| `resource_utilization_percentage` (%) | 85.56 | 85.73 | 15.42 | 40.00 | 74.80 | 96.02 | 136.19 |
| `team_availability_percentage` (%) | 83.64 | 84.05 | 10.30 | 50.39 | 76.23 | 91.53 | 100.00 |
| `unresolved_issue_count` (tickets) | 25.38 | 23.00 | 8.14 | 10.00 | 20.00 | 29.00 | 72.00 |
| `critical_issue_count` (blockers) | 4.18 | 4.00 | 2.58 | 0.00 | 2.00 | 6.00 | 19.00 |
| `bug_count` (defects) | 62.92 | 61.00 | 13.08 | 33.00 | 54.00 | 69.00 | 131.00 |
| `test_pass_percentage` (%) | 82.95 | 83.59 | 6.91 | 50.05 | 78.85 | 87.65 | 99.80 |
| `milestone_completion_percentage` (%) | 48.58 | 48.18 | 25.63 | 4.31 | 26.77 | 70.67 | 100.00 |
| `requirement_change_count` (changes) | 6.95 | 7.00 | 2.64 | 0.00 | 5.00 | 9.00 | 18.00 |
| `team_attrition_rate` (%) | 6.43 | 4.48 | 6.32 | 0.00 | 1.87 | 8.94 | 40.00 |
| `stakeholder_satisfaction` (1-5 scale) | 3.78 | 3.85 | 0.67 | 1.00 | 3.38 | 4.25 | 5.00 |
| `project_duration_days` (days) | 300.54 | 300.50 | 138.77 | 60.00 | 183.00 | 420.25 | 539.00 |
| `remaining_duration_days` (days) | 147.56 | 120.00 | 112.04 | 0.00 | 56.75 | 217.00 | 502.00 |

### Table 1.4: Extreme Feature Correlations with Risk Severity Scale (0=On Track, 1=In Progress, 2=At Risk)

| Rank | Feature Name | Pearson Correlation ($r$) | Impact on Project Risk Profile |
| :---: | :--- | :---: | :--- |
| **Top 1 (+)** | `schedule_delay_days` | **+0.3552** | Primary risk driver; slippage severely escalates failure probability |
| **Top 2 (+)** | `unresolved_issue_count` | **+0.3389** | High issue backlog correlates with engineering delivery stall |
| **Top 3 (+)** | `critical_issue_count` | **+0.3347** | Blocker defects directly prevent milestone certification |
| **Top 4 (+)** | `bug_count` | **+0.3123** | Defect density increases rework overhead and testing regression |
| **Top 5 (+)** | `budget_variance_percentage` | **+0.2116** | Unplanned budget burn amplifies financial distress |
| **Top 1 (-)** | `test_pass_percentage` | **-0.3380** | Strongest stabilizing factor; robust test suites protect delivery |
| **Top 2 (-)** | `stakeholder_satisfaction` | **-0.2918** | High client satisfaction provides organizational buffer |
| **Top 3 (-)** | `team_availability_percentage` | **-0.0179** | High team capacity slightly dampens risk escalation |
| **Top 4 (-)** | `project_completion_percentage` | **+0.0046** | Near orthogonal; completion percentage alone is non-predictive |
| **Top 5 (-)** | `milestone_completion_percentage` | **+0.0070** | Near orthogonal; milestone percentage alone is non-predictive |

---

## 2. Machine Learning Results (Section VI - Model Comparison)

Evaluated on the independent test set ($N = 700$) after training on $N = 2,800$ instances. Class mapping: $0 = 	ext{At Risk}$, $1 = 	ext{In Progress}$, $2 = 	ext{On Track}$.

### Table 2.1: Final Supervised Model Benchmark Performance

| Evaluation Metric | Logistic Regression (Balanced) | Random Forest (100 Trees) | Delta ($\Delta_{RF - LR}$) | Academic Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | **0.5243** | 0.5043 | -0.0200 | Linear baseline slightly outperforms ensemble on overall accuracy |
| **Macro Precision** | 0.5162 | **0.5283** | +0.0121 | Random Forest achieves marginally tighter multi-class precision |
| **Macro Recall** | **0.5427** | 0.5017 | -0.0410 | Logistic Regression delivers superior balanced recall across classes |
| **Macro F1-Score** | **0.5167** | 0.5097 | -0.0070 | Comparable balanced macro F1 scores |
| **Weighted F1-Score** | 0.5039 | **0.5065** | +0.0026 | Comparable weighted representation |
| **Multiclass ROC-AUC (OvR)** | **0.7238** | 0.6827 | -0.0411 | Genuine discriminative capacity well above random guessing (0.50) |
| **At Risk Precision** | 0.5622 | **0.6319** | +0.0697 | Random Forest produces fewer false alarms when flagging At Risk |
| **At Risk Recall** | **0.6684** | 0.4643 | -0.2041 | **Logistic Regression captures 66.84% of failing projects (131/196)** |
| **At Risk F1-Score** | **0.6107** | 0.5353 | -0.0754 | Balanced LR provides stronger safety detection margin |
| **In Progress Recall** | 0.2707 | **0.5113** | +0.2406 | Random Forest handles intermediate boundary overlap better |
| **On Track Precision** | **0.5307** | 0.5185 | -0.0122 | Comparable precision on healthy projects |
| **On Track Recall** | **0.6891** | 0.5294 | -0.1597 | Logistic Regression correctly identifies 68.91% of On Track projects |
| **On Track F1-Score** | **0.5996** | 0.5239 | -0.0757 | Balanced LR outperforms RF on healthy project classification |

### Table 2.2: Stratified 5-Fold Cross-Validation Stability (Training Set, N = 2,800)

| Model Architecture | CV Accuracy (Mean $\pm$ Std) | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.5054 $\pm$ 0.0202** | **0.5011 $\pm$ 0.0200** | **0.5191 $\pm$ 0.0234** | **0.5028 $\pm$ 0.0201** | **0.4940 $\pm$ 0.0180** |
| **Random Forest (100 Trees)** | 0.4718 $\pm$ 0.0143 | 0.4901 $\pm$ 0.0172 | 0.4697 $\pm$ 0.0147 | 0.4759 $\pm$ 0.0155 | 0.4733 $\pm$ 0.0148 |

### Table 2.3: Corrected Numerical Confusion Matrices ($N = 700$)

#### Logistic Regression
| Ground Truth \ Predicted | Pred At Risk (0) | Pred In Progress (1) | Pred On Track (2) | Total Actual | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual At Risk (0)** | **131** | 39 | 26 | 196 | **66.84%** |
| **Actual In Progress (1)** | 75 | **72** | 119 | 266 | 27.07% |
| **Actual On Track (2)** | 27 | 47 | **164** | 238 | **68.91%** |
| **Total Predicted** | 233 | 158 | 309 | 700 | - |
| **Class Precision** | 56.22% | 45.57% | 53.07% | - | Overall Acc: 52.43% |

#### Random Forest (100 Trees)
| Ground Truth \ Predicted | Pred At Risk (0) | Pred In Progress (1) | Pred On Track (2) | Total Actual | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual At Risk (0)** | **91** | 81 | 24 | 196 | **46.43%** |
| **Actual In Progress (1)** | 37 | **136** | 93 | 266 | 51.13% |
| **Actual On Track (2)** | 16 | 96 | **126** | 238 | 52.94% |
| **Total Predicted** | 144 | 313 | 243 | 700 | - |
| **Class Precision** | 63.19% | 43.45% | 51.85% | - | Overall Acc: 50.43% |

---

## 3. Local Procedural Agent Runtime Statistics (Section VI - Runtime Evaluation)

Timings recorded over 30 repeated steady-state runs using Python's `time.perf_counter()`. Cold-start initialization ($96.76	ext{ ms}$) is excluded from steady-state measurements.

> **CRITICAL QUALIFICATION:**  
> These latencies represent the physical execution speed of the **local procedural Python multi-agent harness**. They do **not** reflect external cloud LLM API network latencies (which typically require $1,500–5,000	ext{ ms}$ per agent call).

### Table 3.1: Procedural Agent Execution Latency Statistics (30 Iterations)

| Procedural Agent Component | Mean (ms) | Median (ms) | Std Dev (ms) | Min (ms) | Max (ms) | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Project Status Tracking Agent** | 0.016 | 0.015 | 0.006 | 0.006 | 0.029 | 0.026 |
| **Market Analysis Agent** | 0.051 | 0.039 | 0.068 | 0.021 | 0.412 | 0.065 |
| **Historical Knowledge Retrieval** | 1.314 | 1.316 | 0.432 | 0.671 | 2.317 | 2.169 |
| **Risk Scoring Agent** | 16.231 | 17.359 | 3.814 | 7.552 | 21.129 | 21.013 |
| **Reporting Agent** | 0.027 | 0.019 | 0.042 | 0.009 | 0.251 | 0.027 |
| **Project Risk Manager (Coordinator)** | 0.007 | 0.007 | 0.002 | 0.003 | 0.011 | 0.010 |
| **Total End-to-End Workflow** | **17.646** | **18.477** | **4.126** | **8.337** | **23.324** | **22.713** |

*Throughput:* Approximately **56.6 project assessments per second** on local CPU hardware. The Risk Scoring Agent constitutes $91.9\%$ of execution time due to DataFrame serialization and tree traversal.

---

## 4. End-to-End Multi-Agent Evaluation (Section VI - System Evaluation)

Tested against 100 independently generated synthetic evaluation scenarios (`SCENARIO-001` through `SCENARIO-100`, random seed 2026).

### Table 4.1: End-to-End System Diagnostic Performance ($N = 100$)

| Metric / Dimension | Value | Interpretation |
| :--- | :---: | :--- |
| **Total Evaluation Scenarios** | 100 | Independent synthetic scenarios |
| **Ground Truth Distribution** | In Progress: 38, On Track: 34, At Risk: 28 | Realistic three-class distribution |
| **Predicted Distribution** | In Progress: 45, On Track: 43, At Risk: 12 | Modest conservatism on extreme failure prediction |
| **Correct Predictions** | **47 (47.0%)** | Matches multiclass accuracy expectation under noisy boundaries |
| **Incorrect Predictions** | 53 (53.0%) | Errors primarily between adjacent classes |
| **Overall Accuracy** | **0.4700** | Full multi-agent diagnostic accuracy |
| **Macro Precision** | 0.5183 | Average precision across all 3 classes |
| **Macro Recall** | 0.4590 | Average recall across all 3 classes |
| **Macro F1-Score** | 0.4597 | Balanced macro diagnostic score |
| **Weighted F1-Score** | 0.4623 | Support-weighted diagnostic score |
| **At Risk Class (N = 28)** | Precision: 0.6667, Recall: 0.2857, F1: 0.4000 | High precision on failing projects (few false alarms) |
| **In Progress Class (N = 38)** | Precision: 0.4000, Recall: 0.4737, F1: 0.4337 | Effective intermediate capture |
| **On Track Class (N = 34)** | Precision: 0.4884, Recall: 0.6176, F1: 0.5455 | Reliable detection of healthy projects |
| **Mean End-to-End Latency** | 12.67 ms (Median: 10.48 ms, P95: 22.58 ms) | Sub-30 ms complete diagnostic turnaround |

### Table 4.2: End-to-End Confusion Matrix ($N = 100$)

| Actual Class \ Predicted | Pred At Risk | Pred In Progress | Pred On Track | Total Actual | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual At Risk** | **8** | 14 | 6 | 28 | **28.57%** |
| **Actual In Progress** | 4 | **18** | 16 | 38 | **47.37%** |
| **Actual On Track** | 0 | 13 | **21** | 34 | **61.76%** |

*Note on Risk Score:* The predicted risk score ($0–100$) averaged $47.24$ (range: $11.42–79.58$), derived from weighted ML probabilities blended with status tracker severity and sentiment penalties.

---

## 5. NLP and Historical Retrieval Performance (Section VI - Subsystems)

### Table 5.1: NLP Sentiment Subsystem Benchmark (Local Domain Lexicon Engine)

| Metric | Measured Value | Scope & Architectural Qualification |
| :--- | :---: | :--- |
| **NLP Engine** | Domain Lexicon Engine | Deterministic software engineering keyword lexicon |
| **Evaluated Communications** | 60 samples | 20 Positive, 20 Neutral, 20 Negative sprint retro samples |
| **Overall Accuracy** | **0.7500 (75.0%)** | 45 / 60 correctly classified |
| **Macro Precision** | 0.8378 | High overall sentiment precision |
| **Macro Recall** | 0.7500 | Balanced recall across sentiments |
| **Macro F1-Score** | 0.7533 | Strong sentiment identification baseline |
| **Negative Class Precision** | **1.0000 (100.0%)** | Zero false alarms when flagging adverse sentiment |
| **Negative Class Recall** | 0.5500 | Identifies 11 of 20 negative retro notes |
| **Negative Class F1-Score** | 0.7097 | Reliable blocker and distress detection |
| **Mean Inference Latency** | **0.010 ms** | Microsecond regex-based keyword parsing |

### Table 5.2: Historical Knowledge Retrieval Subsystem (Vector Cosine TF-IDF Fallback)

| Metric | Measured Value | Scope & Architectural Qualification |
| :--- | :---: | :--- |
| **Retrieval Engine** | Vector Cosine (TF-IDF Space) | ChromaDB API compatible vector space fallback |
| **Historical Knowledge Base** | 15 Case Studies | Curated IT project post-mortems (banking, healthcare, cloud) |
| **Benchmark Query Suite** | 8 Queries | Predefined software engineering failure modes |
| **Hit Rate @ 1 (Top-1 Relevance)** | **100.0% (8 / 8)** | Qualified as a small-scale functional sanity check |
| **Hit Rate @ 3 (Top-3 Relevance)** | **100.0% (8 / 8)** | Precedent retrieved in top 3 ranks across all queries |
| **Mean Precision @ 3** | **0.667** | Average of 2 out of 3 retrieved cases are highly relevant |
| **Mean Retrieval Latency** | **0.560 ms** | Fast in-memory sparse matrix vector lookup |
| **P95 Retrieval Latency** | **1.069 ms** | Sub-millisecond steady-state query response |

---

## 6. Component-Wise Ablation Study (Section VI - Ablation Analysis)

Tested across the identical 100 evaluation scenarios to isolate the marginal diagnostic value of each architectural intelligence layer.

### Table 6.1: Component-Wise Architectural Ablation Benchmark ($N = 100$)

| Architectural Variant | Accuracy | Macro F1 | Weighted F1 | At Risk Recall | At Risk F1 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Structured Indicators Only** (Tabular ML) | 0.3700 | 0.3136 | 0.3244 | 0.1429 | 0.2353 | 10.92 |
| **2. Structured + NLP Signals** | 0.4200 | 0.4243 | 0.4184 | **0.4286** | **0.4528** | 11.96 |
| **3. Structured + Historical Retrieval** | 0.3800 | 0.3477 | 0.3533 | 0.2143 | 0.3243 | 13.43 |
| **4. Complete Multi-Agent Framework** | **0.4700** | **0.4597** | **0.4623** | 0.2857 | 0.4000 | 13.69 |

### Table 6.2: Absolute Marginal Gains ($\Delta$) Against Structured Baseline

| Architectural Variant | $\Delta$ Accuracy (pp) | $\Delta$ Macro F1 (pp) | $\Delta$ At-Risk Recall (pp) | $\Delta$ At-Risk F1 (pp) | Additional Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Structured + NLP** | **+5.00 pp** | **+11.07 pp** | **+28.57 pp** | **+21.75 pp** | +1.04 ms |
| **Structured + Historical** | +1.00 pp | +3.41 pp | +7.14 pp | +8.90 pp | +2.51 ms |
| **Complete Multi-Agent Framework** | **+10.00 pp** | **+14.61 pp** | **+14.28 pp** | **+16.47 pp** | +2.77 ms |

*(pp = percentage points. No claim of asymptotic statistical significance is asserted due to sample size N = 100).*

### Key Research Insights:
1. **NLP Catches Early Distress:** Integrating sentiment signals yields the largest boost in At-Risk recall ($+28.57	ext{ pp}$), confirming that qualitative engineer friction and stakeholder dissatisfaction emerge before quantitative telemetry degrades.
2. **Complete Framework Balances Global Accuracy:** While `Structured + NLP` triggers aggressive failure warnings, the `Complete Framework` blends operational status tracker severity, dampening false alarms and achieving the **highest overall accuracy ($47.00\%$, $+10.00	ext{ pp}$) and highest Macro F1 ($0.4597$, $+14.61	ext{ pp}$)**.

---

## 7. Recommended Table Assignments for Research Paper Draft

* **For Section V (Experimental Setup):**
  * Use **Table 1.1** (Dataset Composition & Quality Overview)
  * Use **Table 1.2** (Class Distribution & Partitions)
  * Use **Table 1.3** (Descriptive Statistics)
* **For Section VI (Results & Discussion):**
  * Use **Table 2.1** (ML Model Comparison) and **Table 2.3** (Confusion Matrices)
  * Use **Table 3.1** (Agent Execution Latency Benchmark)
  * Use **Table 4.1** (End-to-End System Diagnostic Performance)
  * Use **Table 6.1 & 6.2** (Ablation Benchmark and Marginal Gains)
