"""
evaluate_models.py
==================
Evaluates Machine Learning Models on Test Dataset.

Generates:
- Comprehensive metric comparison (Accuracy, Macro/Weighted Precision, Recall, F1, ROC-AUC)
- Per-class classification reports with focus on 'At Risk' recall
- Confusion matrix heatmaps for each model
- Comparative visualization charts
- Target class distribution visualization

Saves outputs in `results/`.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, Any

from evaluation.preprocessing import prepare_data
from evaluation.train_models import train_and_cross_validate


def evaluate_models(
    data_dict: Dict[str, Any],
    trained_models: Dict[str, Any],
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Compute test set metrics, generate confusion matrices, and save reporting artifacts.
    """
    os.makedirs(results_dir, exist_ok=True)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    label_encoder = data_dict['label_encoder']
    # Dynamically derive class names directly from fitted LabelEncoder to guarantee exact index alignment
    target_classes = list(label_encoder.classes_)  # Guaranteed [0: 'At Risk', 1: 'In Progress', 2: 'On Track']
    class_indices = list(range(len(target_classes)))

    # Get string labels for test set
    y_test_labels = label_encoder.inverse_transform(y_test)

    comparison_records = []
    classification_reports_dict = {}
    confusion_matrices = {}

    print("\n" + "=" * 50)
    print("Evaluating Models on Independent Test Set (N = 700)")
    print(f"Class Index Mapping: {dict(enumerate(target_classes))}")
    print("=" * 50)

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Decode predictions
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        # Overall Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Multiclass ROC-AUC (One-vs-Rest)
        try:
            roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except Exception:
            roc_auc_ovr = np.nan

        # Per-class metrics using explicit labels to ensure exact class mapping
        report_dict = classification_report(
            y_test, y_pred,
            labels=class_indices,
            target_names=target_classes,
            output_dict=True,
            zero_division=0
        )
        classification_reports_dict[model_name] = report_dict

        # Extract per-class metrics dynamically by exact class name
        at_risk_prec = report_dict['At Risk']['precision']
        at_risk_rec = report_dict['At Risk']['recall']
        at_risk_f1 = report_dict['At Risk']['f1-score']

        on_track_prec = report_dict['On Track']['precision']
        on_track_rec = report_dict['On Track']['recall']
        on_track_f1 = report_dict['On Track']['f1-score']

        in_prog_prec = report_dict['In Progress']['precision']
        in_prog_rec = report_dict['In Progress']['recall']
        in_prog_f1 = report_dict['In Progress']['f1-score']

        cm = confusion_matrix(y_test, y_pred, labels=class_indices)
        confusion_matrices[model_name] = cm

        print(f"\n--- {model_name} Test Results ---")
        print(f"Accuracy:                 {acc:.4f}")
        print(f"Macro Precision:          {prec_macro:.4f}")
        print(f"Macro Recall:             {rec_macro:.4f}")
        print(f"Macro F1-Score:           {f1_macro:.4f}")
        print(f"Weighted F1-Score:        {f1_weighted:.4f}")
        print(f"Multiclass ROC-AUC (OvR): {roc_auc_ovr:.4f}")
        print(f"--> At Risk    - Precision: {at_risk_prec:.4f} | Recall: {at_risk_rec:.4f} | F1: {at_risk_f1:.4f} (Support: {report_dict['At Risk']['support']})")
        print(f"--> In Progress- Precision: {in_prog_prec:.4f} | Recall: {in_prog_rec:.4f} | F1: {in_prog_f1:.4f} (Support: {report_dict['In Progress']['support']})")
        print(f"--> On Track   - Precision: {on_track_prec:.4f} | Recall: {on_track_rec:.4f} | F1: {on_track_f1:.4f} (Support: {report_dict['On Track']['support']})")

        comparison_records.append({
            'Model': model_name,
            'Accuracy': round(float(acc), 4),
            'Macro_Precision': round(float(prec_macro), 4),
            'Macro_Recall': round(float(rec_macro), 4),
            'Macro_F1': round(float(f1_macro), 4),
            'Weighted_F1': round(float(f1_weighted), 4),
            'ROC_AUC_OvR': round(float(roc_auc_ovr), 4) if not np.isnan(roc_auc_ovr) else None,
            'At_Risk_Precision': round(float(at_risk_prec), 4),
            'At_Risk_Recall': round(float(at_risk_rec), 4),
            'At_Risk_F1': round(float(at_risk_f1), 4),
            'On_Track_Precision': round(float(on_track_prec), 4),
            'On_Track_Recall': round(float(on_track_rec), 4),
            'On_Track_F1': round(float(on_track_f1), 4)
        })

    # Save Model Comparison CSV
    comp_df = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(results_dir, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\n[+] Model comparison saved to {comp_csv_path}")

    # Save Classification Reports JSON
    reports_json_path = os.path.join(results_dir, "classification_reports.json")
    with open(reports_json_path, 'w', encoding='utf-8') as f:
        json.dump(classification_reports_dict, f, indent=4)
    print(f"[+] Detailed classification reports saved to {reports_json_path}")

    # Generate Confusion Matrix Plots
    plot_confusion_matrices(confusion_matrices, target_classes, results_dir)

    # Generate Model Comparison Bar Chart
    plot_model_comparison(comp_df, results_dir)

    # Generate Class Distribution Chart
    plot_class_distribution(data_dict, results_dir)

    return {
        'comparison_df': comp_df,
        'classification_reports': classification_reports_dict,
        'confusion_matrices': confusion_matrices
    }


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    target_classes: list,
    results_dir: str
):
    """
    Plot and save clean confusion matrix heatmaps.
    """
    for model_name, cm in confusion_matrices.items():
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {model_name}", fontsize=12, pad=12)
        plt.colorbar()

        tick_marks = np.arange(len(target_classes))
        plt.xticks(tick_marks, target_classes, rotation=20, ha='right')
        plt.yticks(tick_marks, target_classes)

        # Print cell counts and percentages
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cell_text = f"{cm[i, j]}"
                plt.text(
                    j, i, cell_text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                    fontweight="bold"
                )

        plt.ylabel('Ground Truth Class', fontsize=10)
        plt.xlabel('Predicted Class', fontsize=10)
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        fig_path = os.path.join(results_dir, f"confusion_matrix_{safe_name}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Confusion matrix plot saved to {fig_path}")


def plot_model_comparison(comp_df: pd.DataFrame, results_dir: str):
    """
    Generate grouped bar chart comparing performance metrics across models.
    """
    metrics = ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1', 'At_Risk_Recall', 'On_Track_Recall']
    metric_labels = ['Accuracy', 'Macro Prec', 'Macro Rec', 'Macro F1', 'At-Risk Recall', 'On-Track Recall']

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))

    for idx, row in comp_df.iterrows():
        model_name = row['Model']
        values = [row[m] for m in metrics]
        offset = (idx - 0.5) * width
        rects = plt.bar(x + offset, values, width, label=model_name)

        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )

    plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
    plt.title('Machine Learning Model Performance Comparison (Synthetic IT Risk Dataset)', fontsize=12)
    plt.xticks(x, metric_labels, fontsize=10)
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Model comparison chart saved to {fig_path}")


def plot_class_distribution(data_dict: Dict[str, Any], results_dir: str):
    """
    Generate bar chart illustrating class distributions in training and testing splits.
    """
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    label_encoder = data_dict['label_encoder']
    classes = list(label_encoder.classes_)

    train_counts = [np.sum(y_train == i) for i in range(len(classes))]
    test_counts = [np.sum(y_test == i) for i in range(len(classes))]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    rects1 = plt.bar(x - width/2, train_counts, width, label='Train Partition (80%)', color='#2b5c8f')
    rects2 = plt.bar(x + width/2, test_counts, width, label='Test Partition (20%)', color='#e27c38')

    for rect in rects1:
        height = rect.get_height()
        plt.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        plt.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.ylabel('Record Count', fontsize=11)
    plt.title('Risk Category Stratified Class Distribution', fontsize=12)
    plt.xticks(x, classes, fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(results_dir, "class_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Class distribution chart saved to {fig_path}")


if __name__ == "__main__":
    data = prepare_data()
    train_res = train_and_cross_validate(data)
    evaluate_models(data, train_res['trained_models'])
