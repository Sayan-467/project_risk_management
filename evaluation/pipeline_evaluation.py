"""
pipeline_evaluation.py
======================
End-to-End Evaluation of the Multi-Agent Risk Assessment Pipeline.

Evaluates the integrated multi-agent system on 100 independent synthetic project scenarios.
Computes:
- End-to-end classification accuracy, macro/weighted precision, recall, F1.
- Per-class recall with special attention to 'At Risk' critical identification.
- End-to-end confusion matrix.
- Generates structured outputs:
  - results/end_to_end_predictions.csv
  - results/end_to_end_metrics.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from backend.agents.coordinator import ProjectRiskManager
from evaluation.data_generator import generate_evaluation_scenarios


def evaluate_end_to_end_pipeline(
    n_scenarios: int = 100,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run end-to-end pipeline evaluation over 100 independent project scenarios.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Executing End-to-End Pipeline Evaluation (N = {n_scenarios} scenarios)")
    print("=" * 50)

    # Generate independent test scenarios
    scenarios_df = generate_evaluation_scenarios(n_scenarios=n_scenarios, random_seed=2026)
    manager = ProjectRiskManager()

    records = []
    y_true = []
    y_pred = []
    latencies = []

    target_classes = ['On Track', 'In Progress', 'At Risk']

    for idx, row in scenarios_df.iterrows():
        scenario_id = row['scenario_id']
        ground_truth = row['risk_category']
        proj_data = row.drop(['scenario_id', 'risk_category']).to_dict()

        # Generate realistic contextual text based on project features
        comm_health = proj_data.get('communication_health', 'Good')
        delays = proj_data.get('schedule_delay_days', 0)
        bugs = proj_data.get('critical_issue_count', 0)
        
        if delays > 20 or bugs > 5:
            text = f"Sprint retrospective indicates severe frustration with {bugs} blockers and {delays:.0f} days delay."
        elif delays > 8:
            text = f"Moderate delays experienced in release cycle; velocity recovering."
        else:
            text = f"Sprint goals delivered smoothly with high team morale and stakeholder satisfaction."

        # Run pipeline
        res = manager.assess_project(
            project_data=proj_data,
            project_id=scenario_id,
            communication_text=text,
            variant="complete"
        )

        pred_cat = res['scoring']['risk_category']
        score = res['scoring']['risk_score']
        factors = "; ".join(res['report']['identified_risk_factors'])
        mitigations = " | ".join(res['report']['prioritized_mitigations'])
        lat = res['total_latency_ms']

        is_correct = (pred_cat == ground_truth)
        y_true.append(ground_truth)
        y_pred.append(pred_cat)
        latencies.append(lat)

        records.append({
            'Scenario_ID': scenario_id,
            'Ground_Truth': ground_truth,
            'Predicted_Category': pred_cat,
            'Predicted_Risk_Score': score,
            'Is_Correct': is_correct,
            'Execution_Time_ms': lat,
            'Identified_Risk_Factors': factors,
            'Mitigation_Recommendations': mitigations
        })

    # Save detailed prediction CSV
    pred_df = pd.DataFrame(records)
    pred_csv_path = os.path.join(results_dir, "end_to_end_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[+] End-to-end predictions saved to {pred_csv_path}")

    # Compute Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=target_classes, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=target_classes, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=target_classes, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=target_classes, average='weighted', zero_division=0)

    report_dict = classification_report(
        y_true, y_pred,
        labels=target_classes,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=target_classes)

    metrics_summary = {
        "evaluation_type": "Synthetic End-to-End Multi-Agent Assessment",
        "sample_size": n_scenarios,
        "classes": target_classes,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class_metrics": {
            cls_name: {
                "precision": round(float(report_dict[cls_name]['precision']), 4),
                "recall": round(float(report_dict[cls_name]['recall']), 4),
                "f1_score": round(float(report_dict[cls_name]['f1-score']), 4),
                "support": int(report_dict[cls_name]['support'])
            } for cls_name in target_classes
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": target_classes
        },
        "latency_summary_ms": {
            "mean": round(float(np.mean(latencies)), 3),
            "median": round(float(np.median(latencies)), 3),
            "std": round(float(np.std(latencies)), 3),
            "min": round(float(np.min(latencies)), 3),
            "max": round(float(np.max(latencies)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3)
        }
    }

    print("\n--- End-to-End Multi-Agent Assessment Results ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro Precision:  {prec_macro:.4f}")
    print(f"Macro Recall:     {rec_macro:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"At-Risk Recall:   {report_dict['At Risk']['recall']:.4f}")
    print(f"Mean Latency:     {np.mean(latencies):.2f} ms")

    metrics_json_path = os.path.join(results_dir, "end_to_end_metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] End-to-end metrics saved to {metrics_json_path}")

    return {
        "predictions_df": pred_df,
        "metrics": metrics_summary
    }


if __name__ == "__main__":
    evaluate_end_to_end_pipeline(n_scenarios=100)
