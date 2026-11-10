"""
run_all.py
==========
Master Execution Script for the Quantitative Evaluation Framework.

Executes the end-to-end evaluation workflow:
1. Synthetic IT Project Risk Dataset Generation (data/synthetic_it_project_risk.csv)
2. Dataset Documentation Generation (data/dataset_documentation.md)
3. Preprocessing & Feature Pipeline (backend/ml/models/preprocessor.pkl)
4. Stratified 5-Fold Cross-Validation (results/cross_validation_results.csv)
5. Supervised Model Training (Logistic Regression & Random Forest)
6. Model Evaluation & Visualization (results/model_comparison.csv, charts)
7. Multi-Agent Runtime Benchmarking (results/agent_runtime.csv, results/agent_runtime_summary.json)
8. End-to-End Multi-Agent Evaluation (results/end_to_end_predictions.csv, results/end_to_end_metrics.json)
9. Historical Knowledge Retrieval Evaluation (results/retrieval_benchmark.csv)
10. NLP Sentiment Analysis Evaluation (results/nlp_evaluation.json)
11. Component-Wise Ablation Study (results/ablation_results.csv)
12. Comprehensive Artifact Verification

Usage:
    python -m evaluation.run_all
"""

import os
import sys
import tempfile
import time

# Ensure sandbox-friendly headless matplotlib and OpenMP thread settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

from evaluation.data_generator import generate_it_project_risk_dataset, save_dataset_and_documentation
from evaluation.preprocessing import prepare_data, save_preprocessing_artifacts
from evaluation.train_models import train_and_cross_validate
from evaluation.evaluate_models import evaluate_models
from evaluation.runtime_benchmark import run_runtime_benchmark
from evaluation.pipeline_evaluation import evaluate_end_to_end_pipeline
from evaluation.retrieval_evaluation import evaluate_retrieval
from evaluation.nlp_evaluation import evaluate_nlp_sentiment
from evaluation.ablation_study import run_ablation_study


def main():
    start_total = time.perf_counter()

    print("=" * 80)
    print("AI-Powered IT Project Risk Management: Quantitative Evaluation Framework")
    print("Multi-Agent Intelligence Framework Benchmark Suite")
    print("=" * 80)

    # Step 1: Generate Synthetic Dataset
    print("\n[STAGE 1/10] Procedurally Generating Synthetic IT Project Risk Dataset...")
    df_data = generate_it_project_risk_dataset(n_samples=3500, random_seed=42)
    save_dataset_and_documentation(
        df=df_data,
        csv_path="data/synthetic_it_project_risk.csv",
        doc_path="data/dataset_documentation.md",
        random_seed=42
    )

    # Step 2: Preprocessing
    print("\n[STAGE 2/10] Executing Preprocessing and Stratified Split (80:20)...")
    data_dict = prepare_data(csv_path="data/synthetic_it_project_risk.csv", test_size=0.20, random_state=42)
    save_preprocessing_artifacts(data_dict['preprocessor'], data_dict['label_encoder'])
    print(f"[+] Transformed {data_dict['X_train'].shape[1]} features across {len(data_dict['X_train'])} training and {len(data_dict['X_test'])} testing samples.")

    # Step 3: Model Training & 5-Fold Cross Validation
    print("\n[STAGE 3/10] Performing Stratified 5-Fold Cross-Validation and Model Fitting...")
    train_res = train_and_cross_validate(data_dict, random_state=42)

    # Step 4: Model Evaluation and Plotting
    print("\n[STAGE 4/10] Evaluating Supervised Models on Independent Test Set (N = 700)...")
    eval_res = evaluate_models(data_dict, train_res['trained_models'])

    # Step 5: Multi-Agent Runtime Benchmark
    print("\n[STAGE 5/10] Benchmarking Agent Execution Latencies (30 Iterations)...")
    runtime_res = run_runtime_benchmark(n_iterations=30)

    # Step 6: End-to-End Pipeline Evaluation
    print("\n[STAGE 6/10] Evaluating End-to-End Multi-Agent Pipeline on 100 Scenarios...")
    e2e_res = evaluate_end_to_end_pipeline(n_scenarios=100)

    # Step 7: Historical Knowledge Retrieval Evaluation
    print("\n[STAGE 7/10] Evaluating Historical Knowledge Retrieval Engine...")
    retrieval_res = evaluate_retrieval(n_iterations_per_query=20)

    # Step 8: NLP Sentiment Analysis Evaluation
    print("\n[STAGE 8/10] Evaluating NLP Sentiment Analysis Component...")
    nlp_res = evaluate_nlp_sentiment()

    # Step 9: Component-Wise Ablation Study
    print("\n[STAGE 9/10] Executing Component-Wise Architectural Ablation Study...")
    ablation_df = run_ablation_study(n_scenarios=100)

    # Step 10: Artifact Verification
    print("\n[STAGE 10/10] Verifying Generated Artifacts...")
    expected_files = [
        "data/synthetic_it_project_risk.csv",
        "data/dataset_documentation.md",
        "backend/ml/models/preprocessor.pkl",
        "backend/ml/models/label_encoder.pkl",
        "backend/ml/models/logistic_regression_model.pkl",
        "backend/ml/models/random_forest_model.pkl",
        "backend/ml/models/it_risk_model.pkl",
        "results/cross_validation_results.csv",
        "results/model_comparison.csv",
        "results/classification_reports.json",
        "results/confusion_matrix_logistic_regression.png",
        "results/confusion_matrix_random_forest.png",
        "results/model_comparison.png",
        "results/class_distribution.png",
        "results/agent_runtime.csv",
        "results/agent_runtime_summary.json",
        "results/end_to_end_predictions.csv",
        "results/end_to_end_metrics.json",
        "results/retrieval_benchmark.csv",
        "results/nlp_evaluation.json",
        "results/ablation_results.csv"
    ]

    all_found = True
    for fpath in expected_files:
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  [OK] {fpath:<48} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {fpath}")
            all_found = False

    elapsed = round(time.perf_counter() - start_total, 2)
    print("\n" + "=" * 80)
    if all_found:
        print(f"SUCCESS: All {len(expected_files)} evaluation artifacts generated and verified in {elapsed}s.")
    else:
        print(f"WARNING: Some artifacts were missing after {elapsed}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
