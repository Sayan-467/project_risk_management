# Research Methodology and Experimental Results Audit Report

**Project Title:** AI-Powered IT Project Risk Management Using a Multi-Agent Intelligence Framework  
**Audit Date:** November 2026  
**Auditor:** Independent Research Methodology & Experimental Audit  
**Status:** COMPLETE  
**Overall Verdict:** **CONDITIONAL PASS (CRITICAL REVISIONS REQUIRED BEFORE SUBMISSION)**

---

## 1. Overall Audit Summary

This audit independently inspects the quantitative evaluation framework implemented in the repository to evaluate mathematical rigor, absence of data leakage, experimental reproducibility, and research integrity.

### Summary Assessment Matrix

| Dimension | Audit Status | Key Finding |
| :--- | :---: | :--- |
| **Dataset Generation** | **PASS** | Statistically sound non-linear latent formulation; realistic boundary noise; no single-feature dominance; zero duplicate rows. |
| **Target Leakage** | **PASS** | Preprocessing strictly fitted on training split; no test set features or target representations leaked into transformers or cross-validation. |
| **ML Evaluation** | **FAIL** | **Critical Label Inversion:** `LabelEncoder` sorted classes alphabetically (`At Risk`, `In Progress`, `On Track`), but `evaluate_models.py` passed target names in non-alphabetical order (`On Track`, `In Progress`, `At Risk`). True At-Risk recall is **66.84%** (not 68.91%, which was On-Track recall). Multiclass ROC-AUC (0.7238) is mathematically verified. |
| **Agent Implementation** | **WARNING** | Components in `backend/agents/` are **deterministic procedural Python classes**, NOT autonomous CrewAI agents or LLM-powered entities. Calling them "CrewAI agents" in documentation is factually incorrect. |
| **Runtime Benchmark** | **WARNING** | Timings (~19 ms end-to-end) are physically authentic for local procedural Python execution, but implausibly low if presented as LLM execution. Must be explicitly labeled as local algorithmic harness latency. |
| **NLP & Retrieval** | **WARNING** | Neither HuggingFace `transformers` nor `chromadb` is active in the environment. Both executed under lightweight rule/TF-IDF fallbacks. 100% Hit@1 is a toy artifact of 8 queries against 15 documents. |
| **Ablation Study** | **PASS** | Clean component disabling; identical 100 scenarios; mathematical explanation for why Structured + NLP achieved higher At-Risk recall (42.86%) than Complete Framework (28.57%) due to score dampening. |
| **End-to-End Evaluation** | **PASS** | Fully reproducible; genuine multi-stage pipeline invocation; independent ground-truth scenarios. |

---

## 2. Dataset Generation Audit (`evaluation/data_generator.py`)

**Audit Status:** **PASS**

### 2.1 Target Generation Formula
In `evaluation/data_generator.py`, the target `risk_category` is derived from a continuous latent composite risk index ($\text{CRI}$):
$$\text{CRI} = \left( 0.24 S_{delay} + 0.22 B_{variance} + 0.20 Q_{defects} + 0.14 T_{attrition} + 0.10 R_{scope} + 0.10 K_{satisfaction} \right) \cdot C_{complexity} + 0.18 M_{comm} + \epsilon$$
where $\epsilon \sim \mathcal{N}(0, 0.85^2)$ injects non-linear boundary disturbance. Percentile cutoffs (34th and 72nd percentiles) partition the continuous index into:
* **On Track:** $\text{CRI} < t_{34}$ ($1,190$ records, $34.0\%$)
* **In Progress:** $t_{34} \le \text{CRI} < t_{72}$ ($1,330$ records, $38.0\%$)
* **At Risk:** $\text{CRI} \ge t_{72}$ ($980$ records, $28.0\%$)

### 2.2 Feature-Target Correlation & Absence of Trivial Boundaries
A Pearson correlation audit between every numerical feature and the numeric target was executed:

| Feature Name | Pearson Correlation ($r$) with Target | Leakage Assessment |
| :--- | :---: | :--- |
| `schedule_delay_days` | $+0.3552$ | Moderate correlation; realistic signal |
| `unresolved_issue_count` | $+0.3389$ | Moderate correlation |
| `critical_issue_count` | $+0.3347$ | Moderate correlation |
| `bug_count` | $+0.3123$ | Moderate correlation |
| `test_pass_percentage` | $-0.3380$ | Negative correlation as expected |
| `stakeholder_satisfaction` | $-0.2918$ | Negative correlation as expected |
| `budget_variance_percentage` | $+0.2116$ | Moderate correlation |
| `team_attrition_rate` | $+0.1055$ | Weak correlation |
| `milestone_completion_percentage` | $+0.0070$ | Independence verified |
| `project_completion_percentage` | $+0.0046$ | Independence verified |

**Audit Findings:**
1. **No Single Feature Dominance:** The maximum correlation magnitude is $|r| = 0.3552$. No single feature determines the target category.
2. **Class Overlap:** The additive noise $\epsilon$ produces realistic overlap along class boundaries. A project with moderate schedule slip can be *In Progress* or *At Risk* depending on team attrition and quality drag.
3. **Exact Duplicates:** An audit for duplicate rows returned **0 exact duplicates** across all 3,500 records.
4. **Class Balance:** Class representation is balanced ($28\%$, $38\%$, $34\%$), preventing trivial majority-class collapse.

---

## 3. Target Leakage Findings

**Audit Status:** **PASS**

1. **Target Feature Isolation:** `latent_risk_index` is used solely as an internal calibration variable during generation and is **never** included in the output DataFrame saved to `data/synthetic_it_project_risk.csv`.
2. **Transformation Isolation:** In `evaluation/preprocessing.py`, `preprocessor.fit_transform(X_train_raw)` fits strictly on training instances. `X_test_raw` is transformed using the fitted training statistics without refitting.
3. **Cross-Validation Isolation:** In `evaluation/train_models.py`, `cross_validate()` is passed exclusively `X_train` and `y_train`. The 700 test set instances are never observed during model selection or cross-validation.

---

## 4. Machine Learning Evaluation Audit

**Audit Status:** **FAIL (CRITICAL LABEL INVERSION IN REPORTING)**

### 4.1 Stratified Splitting & Imputation
* **Split Ratio:** Exactly $2,800$ train records ($80.0\%$) and $700$ test records ($20.0\%$).
* **Stratification:** Class distributions in both partitions are identical down to two decimal places:
  * Class 0 (At Risk): $28.00\%$ train ($784$), $28.00\%$ test ($196$)
  * Class 1 (In Progress): $38.00\%$ train ($1,064$), $38.00\%$ test ($266$)
  * Class 2 (On Track): $34.00\%$ train ($952$), $34.00\%$ test ($238$)
* **Data Cleansing:** Imputers replace missing values cleanly without error: zero NaNs in transformed partitions.
* **Class Weighting:** Both models utilize `class_weight='balanced'`.

### 4.2 The Critical Label Inversion Bug
A critical programmatic mismatch was discovered between `LabelEncoder` ordering and classification report labeling in `evaluation/evaluate_models.py`:

```python
# evaluation/preprocessing.py
TARGET_CLASSES = ['On Track', 'In Progress', 'At Risk']
label_encoder = LabelEncoder()
label_encoder.fit(TARGET_CLASSES)
# Scikit-learn sorts classes alphabetically during fit():
# le.classes_ = ['At Risk', 'In Progress', 'On Track']
# Index 0 -> 'At Risk' (196 samples in test set)
# Index 1 -> 'In Progress' (266 samples in test set)
# Index 2 -> 'On Track' (238 samples in test set)
```

However, in `evaluation/evaluate_models.py` (lines 66–71):
```python
report_dict = classification_report(
    y_test, y_pred,
    target_names=target_classes,  # Passed ['On Track', 'In Progress', 'At Risk']
    output_dict=True
)
```

Because `target_names` was passed in the order `['On Track', 'In Progress', 'At Risk']` while `y_test` integer classes were ordered `['At Risk', 'In Progress', 'On Track']`:
* Scikit-learn assigned the label **"On Track"** to integer $0$ (which was actually **At Risk**, support $196$).
* Scikit-learn assigned the label **"At Risk"** to integer $2$ (which was actually **On Track**, support $238$).

### 4.3 Corrected Numerical Verification of Model Performance

#### Logistic Regression (L-BFGS, Balanced)
* **Reported in previous summary:** At-Risk Recall = $68.91\%$ (Support = $238$) $\rightarrow$ **FALSE (This was On-Track Recall)**.
* **Actual Audited Metrics:**
  * **At Risk Recall:** **$66.84\%$** ($131 / 196$, Precision: $56.22\%$, F1: $61.07\%$)
  * **In Progress Recall:** **$27.07\%$** ($72 / 266$, Precision: $45.57\%$, F1: $33.96\%$)
  * **On Track Recall:** **$68.91\%$** ($164 / 238$, Precision: $53.07\%$, F1: $59.96\%$)
  * **Accuracy:** **$52.43\%$** ($367 / 700$)
  * **Multiclass ROC-AUC (OvR macro):** **$0.7238$** (Mathematically verified)

#### Random Forest (100 Trees, Balanced)
* **Reported in previous summary:** At-Risk Recall = $52.94\%$ (Support = $238$) $\rightarrow$ **FALSE (This was On-Track Recall)**.
* **Actual Audited Metrics:**
  * **At Risk Recall:** **$46.43\%$** ($91 / 196$, Precision: $63.19\%$, F1: $53.53\%$)
  * **In Progress Recall:** **$51.13\%$** ($136 / 266$, Precision: $43.45\%$, F1: $46.98\%$)
  * **On Track Recall:** **$52.94\%$** ($126 / 238$, Precision: $51.85\%$, F1: $52.39\%$)
  * **Accuracy:** **$50.43\%$** ($353 / 700$)
  * **Multiclass ROC-AUC (OvR macro):** **$0.6827$** (Mathematically verified)

#### True Confusion Matrices ($N = 700$)

```
Logistic Regression:
Actual \ Pred    At Risk    In Progress    On Track    Total
At Risk            131           39           26        196  (Recall: 66.84%)
In Progress         75           72          119        266  (Recall: 27.07%)
On Track            27           47          164        238  (Recall: 68.91%)

Random Forest:
Actual \ Pred    At Risk    In Progress    On Track    Total
At Risk             91           81           24        196  (Recall: 46.43%)
In Progress         37          136           93        266  (Recall: 51.13%)
On Track            16           96          126        238  (Recall: 52.94%)
```

---

## 5. Agent Implementation Audit (`backend/agents/`)

**Audit Status:** **WARNING (TERMINOLOGY & IMPLEMENTATION DIVERGENCE)**

### 5.1 Inspection of Code Files
Every file inside `backend/agents/` was inspected:
* `coordinator.py`: Contains `class ProjectRiskManager`.
* `status_tracker.py`: Contains `class ProjectStatusTrackingAgent`.
* `market_analyst.py`: Contains `class MarketAnalysisAgent`.
* `risk_scorer.py`: Contains `class RiskScoringAgent`.
* `reporter.py`: Contains `class ReportingAgent`.

### 5.2 Agent Nature Determination
* **Are they CrewAI agents?** **NO.** None of the files import `crewai.Agent`, `crewai.Crew`, or `crewai.Task`.
* **Are they LLM-powered agents?** **NO.** None of the classes make API calls to OpenAI, Anthropic, Gemini, or local Ollama LLMs.
* **Are they mock agents?** **NO.** They are not dummy stubs returning hardcoded constants; they execute functional mathematical calculations, ML predictions, and vector transformations.
* **What are they actually?** They are **deterministic procedural Python classes / modular analytical engines**.

### 5.3 Actual Information Flow
Information flow is strictly synchronous and procedural:
1. `ProjectRiskManager.assess_project()` receives tabular inputs and optional text.
2. Calls `ProjectStatusTrackingAgent.analyze_status()` $\rightarrow$ applies heuristic threshold rules to compute internal severity ($0-100$).
3. Calls `MarketAnalysisAgent.analyze_market_context()` $\rightarrow$ runs regex-based keyword sentiment polarity $[-1.0, +1.0]$.
4. Calls `HistoricalRiskKnowledgeStore.query()` $\rightarrow$ computes cosine similarity against 15 static case studies.
5. Passes collected signals to `RiskScoringAgent.compute_risk_score()` $\rightarrow$ queries Scikit-learn Random Forest model via `predict_proba()`, weights probabilities, and blends with heuristics.
6. Calls `ReportingAgent.generate_report()` $\rightarrow$ formats strings into structured dictionaries.

**Academic Obligation:** The paper **must not** describe these components as "autonomous CrewAI LLM agents". They must be accurately described as an **algorithmic multi-agent simulation framework / procedural decision-support pipeline**.

---

## 6. Runtime Benchmark Audit (`evaluation/runtime_benchmark.py`)

**Audit Status:** **WARNING (AUTHENTIC FOR PROCEDURAL CODE, MISLEADING IF CLAIMED AS LLM)**

1. **High-Resolution Timing:** Timings use `time.perf_counter()`.
2. **Cold-Start Separation:** Initialization and model deserialization are measured separately ($32.8$–$215.9$ ms) and excluded from steady-state measurements.
3. **Repetition:** Exactly 30 steady-state iterations are performed following an unmeasured warm-up cycle.
4. **Latency Verification:**
   * Status Tracking Agent: $0.017$ ms ($17$ microseconds)
   * Market Analysis Agent: $0.043$ ms ($43$ microseconds)
   * Historical Retrieval: $1.477$ ms ($1.48$ milliseconds)
   * Risk Scoring Agent: $17.445$ ms ($17.45$ milliseconds)
   * Reporting Agent: $0.021$ ms ($21$ microseconds)
   * Total End-to-End Latency: **$19.011$ ms**
5. **Plausibility Audit:**
   * $19.011$ ms is **completely authentic** for single-threaded Python execution of Scikit-learn pipelines and in-memory TF-IDF matrix operations.
   * However, $19$ ms is **physically impossible for an LLM multi-agent network**. A single LLM API query over HTTPS takes 500 to 2,500 ms.
   * **Throughput Claim:** The reported throughput of $\approx 52$ projects/sec is mathematically justified **only** for this local procedural engine ($1,000\text{ ms} / 19.01\text{ ms} \approx 52.6\text{ prj/sec}$). It cannot be claimed for LLM-based orchestration.

---

## 7. NLP and Retrieval Audit

**Audit Status:** **WARNING (FALLBACK ENGINES ACTIVE; BENCHMARK TOY CONSTRAINTS)**

### 7.1 NLP Sentiment Component
* **Implementation:** `backend/nlp/sentiment_analyzer.py` contains a try-except block for `transformers.pipeline`. Because `transformers` is not installed in the Python 3.14 environment, it executed under the `Domain Lexicon Engine (Local deterministic)`.
* **Nature:** Analyzes words using predefined positive/negative term sets (`POSITIVE_TERMS`, `NEGATIVE_TERMS`) and regex tokenization.
* **Evaluation Scope:** Tested on 60 handcrafted synthetic sentences. The reported $75\%$ accuracy reflects keyword matching on these 60 sentences, **not** generalizable NLP sentiment analysis on unconstrained text.

### 7.2 Historical Knowledge Retrieval Component
* **Implementation:** `backend/retrieval/historical_store.py` contains a fallback for `chromadb`. Because `chromadb` is not installed, it executed under the `Vector Cosine Similarity (TF-IDF Space)` engine.
* **Corpus Size:** Exactly 15 static case studies (`CASE-001` through `CASE-015`).
* **Evaluation Scope:** Evaluated across 8 predefined queries.
* **Hit Rate @ 1 ($100\%$):** The $100\%$ Hit@1 result is an artifact of having only 15 documents and queries designed with high keyword overlap with specific cases (e.g. "spark query optimization" matching "Data Lakehouse Migration with spark queries"). It is a valid functional sanity check, but cannot be presented as a large-scale semantic retrieval benchmark.

---

## 8. Ablation Study Audit (`evaluation/ablation_study.py`)

**Audit Status:** **PASS**

### 8.1 Experimental Rigor
* **Isolation:** Each variant genuinely disables the intended component (`variant="structured_only"` bypasses NLP and Retrieval; `"structured_nlp"` bypasses Retrieval; `"structured_historical"` bypasses NLP).
* **Test Parity:** All 4 configurations are tested on the **identical 100 evaluation scenarios** (`SCENARIO-001` to `SCENARIO-100`).
* **Ground Truth:** Labels are established prior to evaluation using random seed `2026`.

### 8.2 Investigation: Why Did Structured + NLP Outperform Complete Framework on At-Risk Recall?

| Variant | Accuracy | Macro F1 | At-Risk Recall | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| **1. Structured Only** | 0.3700 | 0.3136 | 0.1429 | 14.03 ms |
| **2. Structured + NLP** | 0.4200 | 0.4243 | **0.4286** | 15.15 ms |
| **3. Structured + Historical** | 0.3800 | 0.3477 | 0.2143 | 15.55 ms |
| **4. Complete Framework** | **0.4700** | **0.4597** | 0.2857 | **12.14 ms** |

**Root Cause Analysis:**
In `backend/agents/risk_scorer.py`:
1. In `Structured + NLP`: Negative sentiment from retrospective notes directly adds up to $+9.0$ points to `base_ml_score` (`final_score = base_ml_score + nlp_adjustment`). This sharp penalty pushes borderline failing projects past the $70.0$ threshold into *At Risk*, achieving **$42.86\%$ recall**.
2. In `Complete Framework`, a dampening linear blend is introduced:
   $$\text{final\_score} = 0.70 \times \text{score} + 0.30 \times \text{internal\_severity}$$
   For many delayed projects, `internal_severity` from the status tracker is moderate ($25–45$). When blended with a high score of $72$, the resulting score is:
   $$0.70 \times 72 + 0.30 \times 35 = 50.4 + 10.5 = 60.9$$
   $60.9$ falls below the $70.0$ threshold, categorizing the project as *In Progress* instead of *At Risk*.
3. **Scientific Validity:** While dampening lowers At-Risk recall from $42.86\%$ to $28.57\%$, it simultaneously eliminates false alarms on normal projects, which is why the **Complete Framework achieves the highest overall accuracy ($47.00\%$) and highest Macro F1 ($0.4597$)**. This trade-off is mathematically explainable and scientifically sound.

---

## 9. End-to-End Evaluation Audit (`evaluation/pipeline_evaluation.py`)

**Audit Status:** **PASS**

* **Execution:** All 100 scenarios execute through the complete multi-stage pipeline.
* **Independence:** Scenarios are generated with independent seed `2026`, distinct from training seed `42`.
* **Output Completeness:** `results/end_to_end_predictions.csv` contains all required fields: `Scenario_ID`, `Ground_Truth`, `Predicted_Category`, `Predicted_Risk_Score`, `Is_Correct`, `Identified_Risk_Factors`, `Mitigation_Recommendations`, and `Execution_Time_ms`.

---

## 10. Results Verification Table

| Specific Metric Claimed | Audited Status | Verified Value | Discrepancy / Audit Comment |
| :--- | :---: | :---: | :--- |
| **68.91% At-Risk Recall** | **MISLABELED** | **66.84%** | $68.91\%$ was On-Track recall due to alphabetical `LabelEncoder` ordering. True At-Risk recall is $66.84\%$. |
| **0.7238 Multiclass ROC-AUC** | **VERIFIED** | **0.7238** | Computed accurately via One-vs-Rest macro average. |
| **19.011 ms Total Agent Latency** | **VERIFIED** | **19.011 ms** | Authentically measured for local procedural Python code. Must not be claimed as LLM latency. |
| **100% Retrieval Hit@1** | **VERIFIED (TOY SET)** | **100.0%** | Accurate for 8 handcrafted queries on 15 cases; not an enterprise-scale benchmark. |
| **75.0% NLP Sentiment Accuracy** | **VERIFIED (TOY SET)** | **75.00%** | Accurate for 60 synthetic sentences on local lexicon; not deep transformer NLP. |
| **47.0% Complete Framework Accuracy** | **VERIFIED** | **47.00%** | Authentically measured across 100 end-to-end evaluation scenarios. |

---

## 11. Critical Issues Requiring Correction

1. **Label Inversion in `evaluate_models.py`:** `target_names` must match `label_encoder.classes_` (`['At Risk', 'In Progress', 'On Track']`) rather than the un-sorted `TARGET_CLASSES` list. Confusion matrix labels and classification report JSON must be updated.
2. **Terminology Divergence:** Terminology claiming active "CrewAI orchestration", "HuggingFace transformer pipeline", and "ChromaDB vector store" in `docs/quantitative_evaluation.md` must be replaced with "procedural multi-agent adapter", "domain keyword lexicon engine", and "vector cosine similarity engine".
3. **Support Mismatch:** In Table 3 of the report, the support for At Risk must be corrected from 238 to 196, and On Track from 196 to 238.

---

## 12. Recommended Corrections

1. **In `evaluation/evaluate_models.py`:**
   Change:
   ```python
   target_classes = data_dict['target_classes']
   ```
   To:
   ```python
   target_classes = list(data_dict['label_encoder'].classes_)
   ```
   This ensures that integer $0$ is labeled *At Risk*, $1$ is *In Progress*, and $2$ is *On Track*, resolving the metric inversion across all JSON and PNG artifacts.
2. **In `docs/quantitative_evaluation.md`:**
   * Update Table 2 and Table 3 with the true audited numbers (At-Risk Recall = $66.84\%$, On-Track Recall = $68.91\%$).
   * Explicitly clarify in Section 10 that latency measurements represent local procedural Python dispatch, and estimate hypothetical LLM cloud latency ($2.0–5.0$ seconds).
   * Note in Section 12 that NLP and Retrieval executed under local deterministic fallbacks due to Python 3.14 binary wheel constraints.

---

## 13. Claims That Are Safe to Include in the Research Paper

* **Methodology:** The design of a 19-feature synthetic IT project-risk benchmark dataset with non-linear boundary disturbance and realistic correlation structures.
* **Cross-Validation Stability:** Logistic Regression achieved $0.5054 \pm 0.0202$ CV accuracy and Random Forest achieved $0.4718 \pm 0.0143$ across stratified 5-fold cross-validation.
* **Discriminative Capacity:** Logistic Regression achieved $0.7238$ multiclass ROC-AUC (OvR) and $66.84\%$ At-Risk recall, demonstrating viable baseline risk screening.
* **Multi-Modal Superiority:** Integrating qualitative communication signals into structured metrics boosted Macro F1 from $0.3136$ to $0.4243$ (+35.3% relative gain) and At-Risk recall from $14.29\%$ to $42.86\%$.
* **Orchestration Benefit:** The complete multi-agent framework achieved the highest overall accuracy ($47.00\%$) and highest Macro F1 ($0.4597$) across 100 evaluation scenarios.
* **Local Harness Efficiency:** The local procedural orchestration pipeline introduces minimal computational overhead ($< 20$ ms per project evaluation).

---

## 14. Claims That Must Be Avoided or Qualified

* **DO NOT CLAIM:** Real-world enterprise validation on historical company repositories. Must be explicitly stated as a synthetic benchmark simulation.
* **DO NOT CLAIM:** That the current runtime benchmarks measure CrewAI or LLM token generation latency. Must be qualified as local algorithmic harness latency.
* **DO NOT CLAIM:** Production-grade enterprise semantic retrieval based on the 8-query test. Must be described as a preliminary vector retrieval proof-of-concept.
* **DO NOT CLAIM:** That Logistic Regression achieved $68.91\%$ At-Risk recall (it achieved $66.84\%$; $68.91\%$ was On-Track recall).
* **DO NOT CLAIM:** That HuggingFace deep learning transformers were deployed in the active environment. Must be acknowledged as a rule-calibrated software engineering sentiment lexicon.
