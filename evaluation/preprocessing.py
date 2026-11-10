"""
preprocessing.py
================
Robust Preprocessing Pipeline for IT Project-Risk Classification.

Handles:
- Missing value imputation (median for numeric, most frequent for categorical).
- Categorical one-hot encoding with unknown handling.
- Feature standardization (Z-score scaling).
- Stratified 80:20 train-test splitting.
- Zero-leakage Scikit-learn Pipeline / ColumnTransformer architecture.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Explicitly defined feature schemas
CATEGORICAL_FEATURES = [
    'project_complexity',
    'communication_health'
]

NUMERICAL_FEATURES = [
    'project_completion_percentage',
    'planned_completion_percentage',
    'schedule_delay_days',
    'budget_utilization_percentage',
    'budget_variance_percentage',
    'resource_utilization_percentage',
    'team_availability_percentage',
    'unresolved_issue_count',
    'critical_issue_count',
    'bug_count',
    'test_pass_percentage',
    'milestone_completion_percentage',
    'requirement_change_count',
    'team_attrition_rate',
    'stakeholder_satisfaction',
    'project_duration_days',
    'remaining_duration_days'
]

TARGET_COLUMN = 'risk_category'
TARGET_CLASSES = ['At Risk', 'In Progress', 'On Track']


def create_preprocessor() -> ColumnTransformer:
    """
    Build Scikit-learn ColumnTransformer for numerical and categorical preprocessing.
    """
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_data(
    csv_path: str = "data/synthetic_it_project_risk.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Load dataset, split with stratification, fit preprocessor on training partition,
    and return transformed matrices alongside metadata.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run data_generator.py first.")

    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Target label mapping
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_CLASSES)
    y_encoded = label_encoder.transform(y)

    # Stratified Train-Test Split (80:20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # Fit preprocessor strictly on training data
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Retrieve transformed feature names
    cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = list(NUMERICAL_FEATURES) + list(cat_feature_names)

    return {
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'feature_names': all_feature_names,
        'target_classes': list(label_encoder.classes_)
    }


def save_preprocessing_artifacts(
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    save_dir: str = "backend/ml/models"
):
    """
    Persist fitted preprocessor and label encoder using joblib.
    """
    os.makedirs(save_dir, exist_ok=True)
    prep_path = os.path.join(save_dir, "preprocessor.pkl")
    le_path = os.path.join(save_dir, "label_encoder.pkl")
    joblib.dump(preprocessor, prep_path)
    joblib.dump(label_encoder, le_path)
    print(f"[+] Preprocessor saved to {prep_path}")
    print(f"[+] Label Encoder saved to {le_path}")


if __name__ == "__main__":
    data_dict = prepare_data()
    save_preprocessing_artifacts(data_dict['preprocessor'], data_dict['label_encoder'])
    print(f"[+] Training Shape: {data_dict['X_train'].shape}, Testing Shape: {data_dict['X_test'].shape}")
    print(f"[+] Transformed Features ({len(data_dict['feature_names'])}): {data_dict['feature_names']}")
