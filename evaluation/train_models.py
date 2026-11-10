"""
train_models.py
===============
Train Machine Learning Classifiers and Perform Stratified 5-Fold Cross-Validation.

Models Evaluated:
1. Logistic Regression (Linear Baseline)
2. Random Forest Classifier (100 Trees - Primary Predictive Model)

Cross-Validation:
- Stratified 5-Fold CV strictly on training partition (X_train).
- Zero data leakage into test set.
- Metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from evaluation.preprocessing import prepare_data, save_preprocessing_artifacts


def train_and_cross_validate(
    data_dict: Dict[str, Any],
    random_state: int = 42,
    results_dir: str = "results",
    models_dir: str = "backend/ml/models"
) -> Dict[str, Any]:
    """
    Perform 5-fold cross-validation, train final models on full training set,
    and persist trained models.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    target_classes = data_dict['target_classes']

    # Initialize models as specified in research setup
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1  # Sandbox safe single-threaded
        )
    }

    # Stratified 5-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }

    cv_records = []
    trained_models = {}

    print("\n" + "=" * 50)
    print("Performing Stratified 5-Fold Cross-Validation")
    print("=" * 50)

    for model_name, model in models.items():
        print(f"\n[>] Running 5-Fold CV for {model_name}...")
        cv_res = cross_validate(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring_metrics,
            return_train_score=False
        )

        acc_mean, acc_std = np.mean(cv_res['test_accuracy']), np.std(cv_res['test_accuracy'])
        prec_mean, prec_std = np.mean(cv_res['test_precision_macro']), np.std(cv_res['test_precision_macro'])
        rec_mean, rec_std = np.mean(cv_res['test_recall_macro']), np.std(cv_res['test_recall_macro'])
        f1_mean, f1_std = np.mean(cv_res['test_f1_macro']), np.std(cv_res['test_f1_macro'])
        f1_w_mean, f1_w_std = np.mean(cv_res['test_f1_weighted']), np.std(cv_res['test_f1_weighted'])

        print(f"    - Accuracy:        {acc_mean:.4f} (+/- {acc_std:.4f})")
        print(f"    - Macro Precision: {prec_mean:.4f} (+/- {prec_std:.4f})")
        print(f"    - Macro Recall:    {rec_mean:.4f} (+/- {rec_std:.4f})")
        print(f"    - Macro F1:        {f1_mean:.4f} (+/- {f1_std:.4f})")
        print(f"    - Weighted F1:     {f1_w_mean:.4f} (+/- {f1_w_std:.4f})")

        cv_records.append({
            'Model': model_name,
            'CV_Accuracy_Mean': round(float(acc_mean), 4),
            'CV_Accuracy_Std': round(float(acc_std), 4),
            'CV_Precision_Macro_Mean': round(float(prec_mean), 4),
            'CV_Precision_Macro_Std': round(float(prec_std), 4),
            'CV_Recall_Macro_Mean': round(float(rec_mean), 4),
            'CV_Recall_Macro_Std': round(float(rec_std), 4),
            'CV_F1_Macro_Mean': round(float(f1_mean), 4),
            'CV_F1_Macro_Std': round(float(f1_std), 4),
            'CV_F1_Weighted_Mean': round(float(f1_w_mean), 4),
            'CV_F1_Weighted_Std': round(float(f1_w_std), 4)
        })

        # Train on full training partition
        print(f"[>] Training final {model_name} on complete training set ({len(X_train)} samples)...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    # Save Cross-Validation Results
    cv_df = pd.DataFrame(cv_records)
    cv_csv_path = os.path.join(results_dir, "cross_validation_results.csv")
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"\n[+] Cross-validation results saved to {cv_csv_path}")

    # Save models
    lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    prod_path = os.path.join(models_dir, "it_risk_model.pkl")  # Primary model for multi-agent system

    joblib.dump(trained_models['Logistic Regression'], lr_path)
    joblib.dump(trained_models['Random Forest'], rf_path)
    joblib.dump(trained_models['Random Forest'], prod_path)

    print(f"[+] Logistic Regression saved to {lr_path}")
    print(f"[+] Random Forest saved to {rf_path}")
    print(f"[+] Production IT Risk Model saved to {prod_path}")

    return {
        'trained_models': trained_models,
        'cv_dataframe': cv_df
    }


if __name__ == "__main__":
    data = prepare_data()
    save_preprocessing_artifacts(data['preprocessor'], data['label_encoder'])
    res = train_and_cross_validate(data)
