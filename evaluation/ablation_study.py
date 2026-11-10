"""
ablation_study.py
=================
Controlled Component-wise Ablation Experiment.

Compares 4 architectural configurations on identical synthetic evaluation scenarios (N = 100):
1. Structured Only: Supervised ML Classifier exclusively (Random Forest).
2. Structured + NLP: ML Classifier + Qualitative Sentiment Analysis Signals.
3. Structured + Historical: ML Classifier + Semantic Historical Case Retrieval.
4. Complete Framework: Coordinated Multi-Agent Synthesis (Structured + NLP + Retrieval + Synthesis).

Metrics Evaluated:
- Overall Accuracy
- Macro F1-Score
- Weighted F1-Score
- At-Risk Class Recall (Primary metric for software project risk detection)
- Mean Execution Latency (ms)

Outputs:
- results/ablation_results.csv
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from backend.agents.coordinator import ProjectRiskManager
from evaluation.data_generator import generate_evaluation_scenarios


def run_ablation_study(
    n_scenarios: int = 100,
    results_dir: str = "results"
) -> pd.DataFrame:
    """
    Execute ablation study comparing 4 framework variants on identical scenarios.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("Executing Component-Wise Ablation Study")
    print("=" * 50)

    scenarios_df = generate_evaluation_scenarios(n_scenarios=n_scenarios, random_seed=2026)
    manager = ProjectRiskManager()

    variants = [
        ("Structured Only", "structured_only", "Supervised ML classifier alone on tabular features"),
        ("Structured + NLP", "structured_nlp", "ML classifier augmented with communication sentiment signals"),
        ("Structured + Historical", "structured_historical", "ML classifier augmented with historical knowledge retrieval"),
        ("Complete Framework", "complete", "Full multi-agent intelligence framework (Structured + NLP + Retrieval + Coordination)")
    ]

    target_classes = ['On Track', 'In Progress', 'At Risk']
    ablation_records = []

    for label, variant_key, description in variants:
        print(f"\n[>] Evaluating Variant: '{label}' ({variant_key})...")
        y_true = []
        y_pred = []
        latencies = []

        for idx, row in scenarios_df.iterrows():
            scenario_id = row['scenario_id']
            ground_truth = row['risk_category']
            proj_data = row.drop(['scenario_id', 'risk_category']).to_dict()

            # Dynamic contextual text based on project features
            delays = proj_data.get('schedule_delay_days', 0)
            bugs = proj_data.get('critical_issue_count', 0)
            if delays > 20 or bugs > 5:
                text = f"Sprint retrospective indicates severe frustration with {bugs} blockers and {delays:.0f} days delay."
            elif delays > 8:
                text = f"Moderate delays experienced in release cycle; velocity recovering."
            else:
                text = f"Sprint goals delivered smoothly with high team morale and stakeholder satisfaction."

            t_start = time.perf_counter()
            res = manager.assess_project(
                project_data=proj_data,
                project_id=scenario_id,
                communication_text=text,
                variant=variant_key
            )
            lat = (time.perf_counter() - t_start) * 1000.0

            pred_cat = res['scoring']['risk_category']

            y_true.append(ground_truth)
            y_pred.append(pred_cat)
            latencies.append(lat)

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, labels=target_classes, average='macro', zero_division=0)
        f1_w = f1_score(y_true, y_pred, labels=target_classes, average='weighted', zero_division=0)
        
        rep = classification_report(y_true, y_pred, labels=target_classes, output_dict=True, zero_division=0)
        at_risk_rec = rep['At Risk']['recall']
        at_risk_f1 = rep['At Risk']['f1-score']
        mean_lat = float(np.mean(latencies))

        print(f"    - Accuracy:        {acc:.4f}")
        print(f"    - Macro F1:        {f1_mac:.4f}")
        print(f"    - Weighted F1:     {f1_w:.4f}")
        print(f"    - At-Risk Recall:  {at_risk_rec:.4f}")
        print(f"    - At-Risk F1:      {at_risk_f1:.4f}")
        print(f"    - Mean Latency:    {mean_lat:.3f} ms")

        ablation_records.append({
            "Framework_Variant": label,
            "Variant_Key": variant_key,
            "Description": description,
            "Accuracy": round(float(acc), 4),
            "Macro_F1": round(float(f1_mac), 4),
            "Weighted_F1": round(float(f1_w), 4),
            "At_Risk_Recall": round(float(at_risk_rec), 4),
            "At_Risk_F1": round(float(at_risk_f1), 4),
            "Mean_Latency_ms": round(mean_lat, 3)
        })

    ablation_df = pd.DataFrame(ablation_records)
    csv_path = os.path.join(results_dir, "ablation_results.csv")
    ablation_df.to_csv(csv_path, index=False)
    print(f"\n[+] Ablation study results saved to {csv_path}")

    return ablation_df


if __name__ == "__main__":
    run_ablation_study(n_scenarios=100)
