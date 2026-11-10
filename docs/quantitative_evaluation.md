# Quantitative Evaluation Report: AI-Powered IT Project Risk Management

**Project Title:** AI-Powered IT Project Risk Management Using a Multi-Agent Intelligence Framework  
**Authors:** Syed Sayan, Sk. Aman Ahmed, Prof. Sourav Malakar  
**Affiliation:** Department of Computer Science & Engineering, Institute of Engineering and Management, Kolkata  
**Date:** November 2026 (Updated with Post-Audit Validated Metrics)  
**Artifact Classification:** Synthetic Benchmark & Empirical Procedural Runtime Evaluation  

---

> ### Academic Disclaimer & Notice of Synthetic Benchmark Nature
> All quantitative performance figures, project records, and evaluation scenarios reported in this document are derived from a procedurally generated, synthetic benchmark environment (`synthetic_it_project_risk.csv` and 100 independent evaluation scenarios). They are **not** observational records from live enterprise IT projects. Runtime latencies, agent execution times, and computational resource usages represent actual physical measurements conducted on local execution hardware using high-resolution timers (`time.perf_counter()`). The multi-agent workflow evaluated herein is a **local procedural Python evaluation adapter / algorithmic simulation harness**, not autonomous cloud LLM agents. Commercial cloud LLM API network latencies and external cloud calls were unmeasured as local execution was enforced without exposing external API keys.

---

## 1. Experimental Objectives

The primary research objective of this evaluation is to provide an empirical, reproducible assessment of the multi-agent IT project-risk management framework described in our research report. Specific experimental goals include:

1. **Benchmark Feasibility in Data-Constrained Domains:** Address the total absence of standardized, publicly available IT project-risk classification datasets by designing a multivariate, domain-grounded synthetic benchmark generator with realistic noise, class overlap, and non-trivial decision boundaries.
2. **Supervised Model Comparison:** Establish rigorous baseline classification performance between a generalized linear model (Logistic Regression) and an ensemble tree model (Random Forest with 100 estimators), evaluating overall accuracy, macro/weighted F1-scores, multiclass ROC-AUC (OvR), and per-class recall (with explicit emphasis on the safety-critical **At Risk** class).
3. **Cross-Validation Stability:** Assess algorithmic generalization stability across stratified 5-fold cross-validation on the training partition without data leakage.
4. **Procedural Multi-Agent Runtime Profiling:** Instrument each functional agent component (Project Risk Manager, Project Status Tracking Agent, Market Analysis Agent, Historical Knowledge Retrieval, Risk Scoring Agent, Reporting Agent) to measure physical execution latencies (Mean, Median, Std, Min, Max, P95) across repeated steady-state runs.
5. **End-to-End Pipeline Validation:** Test the full coordinated workflow against 100 independently generated project scenarios to assess system-level diagnostic accuracy and automated mitigation recommendation relevance.
6. **Component-Wise Ablation Analysis:** Quantify the individual and synergistic contributions of structured telemetry, qualitative sentiment signals, and historical case retrieval in mitigating classification error and detecting critical project risks.

---

## 2. Dataset Generation Methodology

Because real-world enterprise repositories (e.g., Jira, GitHub Enterprise, Linear) rarely publish raw failure telemetry due to NDA and proprietary IP constraints, we established a reproducible parametric simulation protocol.

### 2.1 Domain-Inspired Feature Formulation
The dataset models a snapshot of an IT project mid-lifecycle across 19 critical indicators:
- **Timeline & Schedule Dynamics:** Actual completion percentage ($P_{actual} \in [5\%, 98\%]$), baseline planned completion ($P_{planned} \in [5\%, 100\%]$), and calculated schedule delay days ($D_{delay} \ge 0$).
- **Cost & Budget Drift:** Budget variance percentage ($V_{budget} \in [-30\%, +95\%]$) exhibiting positive covariance with schedule delay, reflecting the burn-rate penalty of project timeline extensions.
- **Defect & Quality Stress:** Unresolved issue count, high-severity critical blocker count ($C_{issues}$), cumulative bug count ($B_{count}$), and automated test pass percentage ($T_{pass} \in [45\%, 99.8\%]$) exhibiting inverse correlation with defect density.
- **Human Resources & Attrition:** Team availability percentage ($A_{team}$) and key engineer turnover rate ($R_{attrition} \in [0\%, 40\%]$).
- **Scope & Governance:** Requirement change count ($S_{scope}$), architectural complexity (Low, Medium, High, Critical), stakeholder satisfaction survey score ($K_{sat} \in [1.0, 5.0]$), and communication health rating (Poor, Fair, Good, Excellent).

### 2.2 Latent Composite Risk Score (CRI) Formulation
To ensure realistic classification complexity where no single feature trivially determines the target label, each record's target class was generated via a multi-factor continuous latent risk index:

$$\text{CRI} = \left( w_s S_{delay} + w_b B_{variance} + w_q Q_{defects} + w_t T_{attrition} + w_r R_{scope} + w_k K_{satisfaction} \right) \cdot C_{complexity} + M_{comm} + \epsilon$$

where:
- Component weights: $w_s = 0.24, w_b = 0.22, w_q = 0.20, w_t = 0.14, w_r = 0.10, w_k = 0.10$.
- Complexity scaling factor $C_{complexity} \in \{0.85, 1.00, 1.25, 1.55\}$.
- Qualitative communication modifier $M_{comm} \in \{+1.4, +0.5, -0.4, -1.1\}$.
- Stochastic disturbance $\epsilon \sim \mathcal{N}(0, 0.85^2)$ introducing non-linear boundary noise and class overlap.

### 2.3 Class Partitioning & Realistic Missingness
- Percentile thresholds at the 34th and 72nd percentiles partition $\text{CRI}$ into three balanced operational states: **On Track**, **In Progress**, and **At Risk**.
- A small fraction (1.5% to 2.5%) of realistic missing data was injected into non-target fields (`team_availability_percentage`, `stakeholder_satisfaction`, `test_pass_percentage`) to validate imputer robustness.
- Fixed random seed: `42`. File saved at `data/synthetic_it_project_risk.csv`.

---

## 3. Dataset Statistics

| Metric / Dimension | Measured Value |
| :--- | :--- |
| **Total Project Snapshots** | 3,500 |
| **Total Features** | 19 (17 numerical, 2 categorical) |
| **Target Variable** | `risk_category` (3 classes) |
| **At Risk Records** | 980 (28.00%) |
| **In Progress Records** | 1,330 (38.00%) |
| **On Track Records** | 1,190 (34.00%) |
| **Random Seed** | 42 |
| **Target Leakage Safeguard** | Target excluded from feature matrix; preprocessing strictly train-fit |

---

## 4. Preprocessing Methodology

A zero-leakage Scikit-learn `ColumnTransformer` pipeline was implemented:

1. **Numerical Pipeline:**
   - Missing value imputation: `SimpleImputer(strategy='median')`
   - Feature scaling: `StandardScaler()` (Z-score zero-mean unit-variance transformation)
2. **Categorical Pipeline:**
   - Missing value imputation: `SimpleImputer(strategy='most_frequent')`
   - Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
   - Categorical dimensions encoded: `project_complexity` (4 levels) and `communication_health` (4 levels).
3. **Target Encoding:**
   - `LabelEncoder` strictly yields alphabetical class mapping:
     - Index $0 \rightarrow$ **`At Risk`**
     - Index $1 \rightarrow$ **`In Progress`**
     - Index $2 \rightarrow$ **`On Track`**
4. **Dimensionality:**
   - Raw features: 19
   - Post-encoding feature dimensionality: **25 features**
5. **Artifact Persistence:**
   - Preprocessor saved to: `backend/ml/models/preprocessor.pkl`
   - Target label encoder saved to: `backend/ml/models/label_encoder.pkl`

---

## 5. Train-Test Split Details

- **Partition Ratio:** 80% Training ($N = 2,800$), 20% Independent Testing ($N = 700$).
- **Stratification:** Stratified by `risk_category` to maintain identical class ratios across splits.
- **Random State:** Fixed at `42`.
- **Class Breakdown in Splits:**
  - **Training ($N = 2,800$):** At Risk: 784 (28.0%), In Progress: 1,064 (38.0%), On Track: 952 (34.0%).
  - **Testing ($N = 700$):** At Risk: 196 (28.0%), In Progress: 266 (38.0%), On Track: 238 (34.0%).

---

## 6. Machine Learning Model Configurations

Two supervised classification architectures were trained and benchmarked:

1. **Logistic Regression (Linear Baseline):**
   - Solver: L-BFGS
   - Regularization: $L_2$ penalty ($C = 1.0$)
   - Multi-class Strategy: Multinomial / Cross-Entropy Loss
   - Class Weight: `'balanced'` (inverse frequency weighting)
   - Max Iterations: 1,000
   - Random State: `42`
2. **Random Forest Classifier (Primary Non-Linear Ensemble):**
   - Number of Trees: $n_{estimators} = 100$
   - Criterion: Gini Impurity
   - Max Features: $\sqrt{n_{features}}$
   - Class Weight: `'balanced'`
   - Random State: `42`
   - Concurrency: $n_{jobs} = 1$ (sandbox execution safety)

---

## 7. Cross-Validation Methodology & Results

Stratified 5-fold cross-validation was executed strictly on the 2,800-sample training partition. The independent test set was isolated to eliminate data leakage.

### Table 1: Stratified 5-Fold Cross-Validation Performance (Training Set, N = 2,800)

| Model | CV Accuracy (Mean $\pm$ Std) | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.5054 $\pm$ 0.0202** | 0.5011 $\pm$ 0.0200 | **0.5191 $\pm$ 0.0234** | **0.5028 $\pm$ 0.0201** | 0.4940 $\pm$ 0.0180 |
| **Random Forest (100 Trees)** | 0.4718 $\pm$ 0.0143 | 0.4901 $\pm$ 0.0172 | 0.4697 $\pm$ 0.0147 | 0.4759 $\pm$ 0.0155 | 0.4733 $\pm$ 0.0148 |

*Observation:* Both classifiers exhibit low standard deviations across folds ($\sigma \approx 0.014 - 0.023$), demonstrating that model convergence is stable across varied project partitions and free of fold-specific failure.

---

## 8. Machine Learning Evaluation Results (Independent Test Set, N = 700)

Both models were fitted on the complete 2,800-sample training partition and evaluated on the untouched 700-sample test set. The metrics below represent the **post-audit verified values** with exact class alignment.

### Table 2: Model Comparison on Independent Test Set (Post-Audit Verified)

| Performance Metric | Logistic Regression | Random Forest Classifier | Delta ($\Delta_{RF - LR}$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.5243** | 0.5043 | -0.0200 |
| **Macro Precision** | 0.5162 | **0.5283** | +0.0121 |
| **Macro Recall** | **0.5427** | 0.5017 | -0.0410 |
| **Macro F1-Score** | **0.5167** | 0.5097 | -0.0070 |
| **Weighted F1-Score** | 0.5039 | **0.5065** | +0.0026 |
| **Multiclass ROC-AUC (OvR)** | **0.7238** | 0.6827 | -0.0411 |
| **At Risk Precision** | 0.5622 | **0.6319** | +0.0697 |
| **At Risk Recall** | **0.6684** | 0.4643 | -0.2041 |
| **At Risk F1-Score** | **0.6107** | 0.5353 | -0.0754 |
| **On Track Precision** | **0.5307** | 0.5185 | -0.0122 |
| **On Track Recall** | **0.6891** | 0.5294 | -0.1597 |
| **On Track F1-Score** | **0.5996** | 0.5239 | -0.0757 |

### Table 3: Detailed Per-Class Classification Breakdown (Post-Audit Verified)

| Model | Class Label | Precision | Recall | F1-Score | Support ($N_{test}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **At Risk** | 0.5622 | **0.6684** | 0.6107 | 196 (28%) |
| | **In Progress** | 0.4557 | 0.2707 | 0.3396 | 266 (38%) |
| | **On Track** | 0.5307 | **0.6891** | 0.5996 | 238 (34%) |
| **Random Forest** | **At Risk** | **0.6319** | 0.4643 | 0.5353 | 196 (28%) |
| | **In Progress** | 0.4345 | 0.5113 | 0.4698 | 266 (38%) |
| | **On Track** | 0.5185 | 0.5294 | 0.5239 | 238 (34%) |

---

## 9. Confusion Matrix Interpretation

### Table 4: Corrected Confusion Matrices (Test Set $N = 700$)

#### Logistic Regression
| Actual \ Predicted | At Risk (Pred) | In Progress (Pred) | On Track (Pred) | Total Actual | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **At Risk** | **131** | 39 | 26 | 196 | **66.84%** |
| **In Progress** | 75 | **72** | 119 | 266 | 27.07% |
| **On Track** | 27 | 47 | **164** | 238 | **68.91%** |

#### Random Forest (100 Trees)
| Actual \ Predicted | At Risk (Pred) | In Progress (Pred) | On Track (Pred) | Total Actual | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **At Risk** | **91** | 81 | 24 | 196 | **46.43%** |
| **In Progress** | 37 | **136** | 93 | 266 | 51.13% |
| **On Track** | 16 | 96 | **126** | 238 | **52.94%** |

### Analytical Interpretation:
1. **At-Risk Sensitivity:** In safety-critical IT project-risk management, failing to detect an *At Risk* project is costly. Balanced Logistic Regression achieved **66.84% recall** on the At Risk class (identifying 131 of 196 failing projects), outperforming Random Forest's 46.43% recall.
2. **Precision vs. Recall Trade-off:** Random Forest achieved higher precision on At Risk projects (**63.19%** vs. 56.22%), meaning fewer false alarms when flagging a project as failing, but caught fewer total failures.
3. **Intermediate Class Ambiguity:** The *In Progress* intermediate category exhibits substantial boundary overlap. Logistic Regression polarized predictions toward the extremes (At Risk and On Track), whereas Random Forest predicted the intermediate category more effectively (136 true positives for In Progress).
4. **Multiclass Discrimination:** The One-vs-Rest ROC-AUC of **0.7238** for Logistic Regression and **0.6827** for Random Forest confirms that both models possess genuine discriminative capacity well above random guessing (AUC = 0.50).

---

## 10. Local Multi-Agent Runtime Measurements

Execution latencies were measured across 30 repeated steady-state runs using Python's high-resolution `time.perf_counter()` timer. System initialization and model loading were timed separately.

> **IMPORTANT ARCHITECTURAL NOTE:**  
> The latencies reported below represent the execution speed of the **local procedural Python multi-agent adapter**. They do **not** represent autonomous cloud LLM calls (e.g., GPT-4o or Claude 3.5 via CrewAI), which incur significant network RTT and token generation latency ($1.5–5.0$ seconds).

### Table 5: Local Multi-Agent Execution Latency Benchmark (30 Iterations)

| Workflow Component | Mean (ms) | Median (ms) | Std Dev (ms) | Min (ms) | Max (ms) | 95th Percentile (P95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Project Status Tracking Agent** | 0.017 | 0.015 | 0.006 | 0.007 | 0.033 | 0.028 |
| **Market Analysis Agent** | 0.043 | 0.038 | 0.015 | 0.023 | 0.086 | 0.076 |
| **Historical Knowledge Retrieval** | 1.477 | 1.319 | 0.392 | 0.848 | 2.419 | 2.191 |
| **Risk Scoring Agent** | 17.445 | 16.880 | 3.904 | 12.430 | 28.684 | 26.730 |
| **Reporting Agent** | 0.021 | 0.019 | 0.007 | 0.011 | 0.048 | 0.035 |
| **Coordinator Overhead** | 0.009 | 0.007 | 0.006 | 0.005 | 0.041 | 0.013 |
| **Total End-to-End Workflow Latency** | **19.011** | **18.454** | **4.202** | **13.486** | **31.193** | **28.838** |

### Execution Timing Characteristics:
- **Initialization & Model Loading:** 32.81 ms (one-time cold start for preprocessor, vectorizer, and ensemble deserialization).
- **Steady-State Processing:** Average end-to-end latency per project is **19.01 ms**, with a P95 ceiling of **28.84 ms**, demonstrating local throughput of approximately 52 projects per second on a single CPU thread.
- **Latency Bottleneck:** The Risk Scoring Agent accounts for approximately 91% of steady-state execution time due to DataFrame serialization, pipeline transformation, and Random Forest 100-tree probability evaluation.

---

## 11. End-to-End Risk Assessment Evaluation

An independent test set of 100 project scenarios (`SCENARIO-001` through `SCENARIO-100`) was evaluated through the full procedural multi-agent workflow (Status Tracker $\rightarrow$ Market Analyst $\rightarrow$ Historical Store $\rightarrow$ Risk Scorer $\rightarrow$ Reporting Agent).

### Table 6: End-to-End Multi-Agent Performance Summary ($N = 100$ Scenarios)

| Evaluation Metric | Measured Value |
| :--- | :--- |
| **Overall Classification Accuracy** | **0.4700** |
| **Macro Precision** | 0.5183 |
| **Macro Recall** | 0.4590 |
| **Macro F1-Score** | 0.4597 |
| **Weighted F1-Score** | 0.4623 |
| **At Risk (Support = 28)** | Precision: 0.6667, Recall: 0.2857, F1: 0.4000 |
| **In Progress (Support = 38)** | Precision: 0.4000, Recall: 0.4737, F1: 0.4337 |
| **On Track (Support = 34)** | Precision: 0.4884, Recall: 0.6176, F1: 0.5455 |
| **Mean End-to-End Latency** | 14.34 ms (Median: 12.81 ms, P95: 27.99 ms) |

---

## 12. Retrieval and NLP Sentiment Evaluation

### 12.1 Historical Knowledge Retrieval Evaluation
The retrieval engine was evaluated against 8 predefined domain queries matching historical case studies.

| Metric | Measured Result |
| :--- | :--- |
| **Active Retrieval Engine** | Vector Cosine Similarity (TF-IDF Space, ChromaDB API compatible) |
| **Hit Rate @ 1 (Top-1 Relevance)** | **100.0%** (8 / 8 benchmark queries) |
| **Hit Rate @ 3 (Top-3 Relevance)** | **100.0%** (8 / 8 benchmark queries) |
| **Mean Precision @ 3** | **0.667** |
| **Mean Retrieval Latency** | **1.331 ms** (Median: 1.372 ms, P95: 2.384 ms) |

> **QUALIFICATION:**  
> The 100% Hit@1 result represents a functional sanity check on a curated knowledge collection of 15 case studies. It demonstrates that vector representation correctly matches key architectural and defect terms to relevant precedents, but should not be characterized as an enterprise-scale retrieval benchmark.

### 12.2 NLP Sentiment Analysis Evaluation
A curated corpus of 60 synthetic project communication samples (20 Positive, 20 Neutral, 20 Negative) was evaluated.

| Metric | Measured Result |
| :--- | :--- |
| **Active NLP Engine** | Domain Lexicon Engine (Local deterministic IT risk keyword lexicon) |
| **Overall Sentiment Accuracy** | **0.7500** (45 / 60 correct) |
| **Macro Precision** | 0.8378 |
| **Macro Recall** | 0.7500 |
| **Macro F1-Score** | 0.7533 |
| **Negative Class Precision** | **1.0000** (Zero false negative alarms) |
| **Negative Class Recall** | 0.5500 |
| **Mean Inference Latency** | **0.011 ms** per communication record |

> **QUALIFICATION:**  
> The sentiment analysis module operates using a domain-specific software engineering keyword lexicon. It effectively detects explicit negative indicators ("blocker", "resignation", "burnout") with 100% precision, but is a rule-calibrated fallback rather than a pre-trained transformer model (e.g., DistilBERT).

---

## 13. Component-Wise Ablation Study

To evaluate whether multi-agent information fusion produces quantifiable benefits over isolated tabular machine learning, a controlled ablation experiment was conducted on the identical 100 evaluation scenarios.

### Table 7: Architectural Ablation Study ($N = 100$ Scenarios)

| Framework Configuration | Accuracy | Macro F1 | Weighted F1 | At-Risk Recall | At-Risk F1 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Structured Indicators Only** (Tabular ML) | 0.3700 | 0.3136 | 0.3244 | 0.1429 | 0.2353 | 14.03 |
| **2. Structured + NLP Signals** | 0.4200 | 0.4243 | 0.4184 | **0.4286** | **0.4528** | 15.15 |
| **3. Structured + Historical Retrieval** | 0.3800 | 0.3477 | 0.3533 | 0.2143 | 0.3243 | 15.55 |
| **4. Complete Multi-Agent Framework** | **0.4700** | **0.4597** | **0.4623** | 0.2857 | 0.4000 | **12.14** |

### Key Ablation Findings:
1. **Multi-Modal Superiority:** Adding NLP sentiment signals dramatically increased Macro F1 from **0.3136 to 0.4243** (+35.3% relative gain) and tripled At-Risk recall from **14.29% to 42.86%**. Qualitative communication discontent often precedes metric deterioration.
2. **Understanding At-Risk Recall Trade-Off:** In `Structured + NLP`, adverse sentiment directly penalizes the score by up to $+9$ points, pushing borderline failing projects over the 70 threshold. In the `Complete Framework`, internal severity from the status tracker is linearly blended ($0.70 \times \text{score} + 0.30 \times \text{internal\_severity}$), which dampens extreme scores. While this moderates At-Risk recall from $42.86\%$ to $28.57\%$, it prevents false alarms on normal projects, enabling the **Complete Framework to achieve the highest overall accuracy (0.4700) and highest Macro F1 (0.4597)**.

---

## 14. Limitations and Threats to Validity

1. **Synthetic Data Realism:** Although parameterized using realistic software engineering dynamics (Brook's Law, cost-delay covariance, defect growth models), the synthetic benchmark cannot fully capture the long-tail anomalies, political reporting biases, and sudden executive restructuring events typical of live enterprise projects.
2. **Local Procedural Adapter vs Cloud LLM:** Measured latencies reflect highly optimized, local deterministic agent logic (10–30 ms). If agents are delegated to commercial cloud LLMs (e.g. OpenAI GPT-4o or Anthropic Claude 3.5 Sonnet via CrewAI), network latency and LLM token generation will dominate, increasing response times to 2.0–8.0 seconds per project evaluation.
3. **Retrieval Scale:** The historical store was tested on 15 representative case studies. Enterprise deployment would require indexing thousands of post-mortem reports, necessitating vector database indexing benchmarks (e.g., HNSW indexing with ChromaDB / Milvus).

---

## 15. Summary of Experimental Artifacts

All artifacts were generated via reproducible execution and are stored in the repository:

```
data/
  ├── synthetic_it_project_risk.csv          # 3,500 project records (356 KB)
  └── dataset_documentation.md               # Dataset metadata and dictionary
backend/ml/models/
  ├── preprocessor.pkl                       # Fitted ColumnTransformer (5.5 KB)
  ├── label_encoder.pkl                      # Target class mapping
  ├── logistic_regression_model.pkl          # Trained baseline classifier (1.5 KB)
  ├── random_forest_model.pkl                # Trained 100-tree ensemble (12.5 MB)
  └── it_risk_model.pkl                      # Production multi-agent ML model
results/
  ├── cross_validation_results.csv           # 5-fold CV statistics
  ├── model_comparison.csv                   # Comprehensive test metrics
  ├── classification_reports.json            # Full per-class reports
  ├── confusion_matrix_logistic_regression.png
  ├── confusion_matrix_random_forest.png
  ├── model_comparison.png                   # Grouped comparison chart
  ├── class_distribution.png                 # Split class distribution chart
  ├── agent_runtime.csv                      # Raw 30-run latency data
  ├── agent_runtime_summary.json             # Agent latency stats (P95, Mean)
  ├── end_to_end_predictions.csv             # 100 scenario predictions & mitigations
  ├── end_to_end_metrics.json                # Pipeline metrics & confusion matrix
  ├── retrieval_benchmark.csv                # Retrieval latency & Hit@K
  ├── nlp_evaluation.json                    # Sentiment classification metrics
  └── ablation_results.csv                   # 4-variant ablation comparison
docs/
  ├── quantitative_evaluation.md             # This comprehensive academic report
  └── research_methodology_audit.md          # Complete audit report
```

---

## 16. Reproduction Command

To reproduce every table, metric, chart, and model artifact from scratch:

```bash
python -m evaluation.run_all
```
