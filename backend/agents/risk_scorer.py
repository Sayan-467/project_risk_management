"""
risk_scorer.py
==============
Risk Scoring Agent.

Synthesizes multiple intelligence signals:
1. Supervised Machine Learning Classifier predictions (Random Forest / Logistic Regression)
2. Status Tracking internal severity indicators
3. Market Analysis sentiment and stakeholder scores
4. Historical Knowledge similarity factors

Computes a consolidated continuous Risk Score (0-100) and maps it to target categories:
- On Track   (0.0 - 39.9)
- In Progress (40.0 - 69.9)
- At Risk    (70.0 - 100.0)
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class RiskScoringAgent:
    """
    Synthesizes ML model probabilities and multi-agent evidence into a consolidated risk score.
    """
    def __init__(
        self,
        model_path: str = "backend/ml/models/it_risk_model.pkl",
        preprocessor_path: str = "backend/ml/models/preprocessor.pkl",
        label_encoder_path: str = "backend/ml/models/label_encoder.pkl"
    ):
        self.name = "Risk Scoring Agent"
        self.role = "Multi-Signal Quantitative Risk Synthesizer"

        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.target_classes = ['On Track', 'In Progress', 'At Risk']

        # Load ML components
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
            self.target_classes = list(self.label_encoder.classes_)

    def compute_risk_score(
        self,
        project_features: Dict[str, Any],
        status_output: Optional[Dict[str, Any]] = None,
        market_output: Optional[Dict[str, Any]] = None,
        retrieval_output: Optional[Dict[str, Any]] = None,
        variant: str = "complete"
    ) -> Dict[str, Any]:
        """
        Compute consolidated risk score and classification.

        Variants supported for ablation study:
        - 'structured_only': Uses ML model probabilities exclusively.
        - 'structured_nlp': ML probabilities + NLP sentiment modifier.
        - 'structured_historical': ML probabilities + historical precedent risk.
        - 'complete': Full multi-agent synthesis across all 4 components.
        """
        start_t = time.perf_counter()

        # 1. Supervised Machine Learning Prediction
        ml_probs = {'On Track': 0.33, 'In Progress': 0.34, 'At Risk': 0.33}
        ml_pred_class = 'In Progress'

        if self.model is not None and self.preprocessor is not None:
            # Format single record DataFrame
            feat_df = pd.DataFrame([project_features])
            try:
                X_trans = self.preprocessor.transform(feat_df)
                probs = self.model.predict_proba(X_trans)[0]
                pred_idx = np.argmax(probs)
                ml_pred_class = self.target_classes[pred_idx]
                for idx, c in enumerate(self.target_classes):
                    ml_probs[c] = float(probs[idx])
            except Exception as e:
                pass

        # Base ML Risk Score (0-100 scale): weighted expectation
        # P(On Track)*15 + P(In Progress)*55 + P(At Risk)*90
        p_track = ml_probs.get('On Track', 0.33)
        p_prog = ml_probs.get('In Progress', 0.34)
        p_risk = ml_probs.get('At Risk', 0.33)

        base_ml_score = (p_track * 15.0) + (p_prog * 55.0) + (p_risk * 90.0)

        # 2. Component Adjustments based on Variant
        final_score = base_ml_score

        if variant in ("structured_nlp", "complete") and market_output is not None:
            sentiment_data = market_output.get('sentiment_analysis', {})
            polarity = sentiment_data.get('polarity_score', 0.0)
            # Negative sentiment increases score by up to 10 points; positive decreases by up to 8
            nlp_adjustment = - (polarity * 9.0)
            final_score += nlp_adjustment

        if variant in ("structured_historical", "complete") and retrieval_output is not None:
            matches = retrieval_output.get('matches', [])
            if matches:
                # Average similarity of top precedents
                avg_sim = np.mean([m.get('similarity_score', 0.0) for m in matches[:2]])
                # Precedents identify active failure modes
                hist_adjustment = (avg_sim - 0.20) * 8.0
                final_score += hist_adjustment

        if variant == "complete" and status_output is not None:
            internal_severity = status_output.get('internal_severity_score', 0.0)
            # Balance internal operational severity
            final_score = 0.70 * final_score + 0.30 * internal_severity

        # Bound score to [0.0, 100.0]
        final_score = float(np.clip(final_score, 0.0, 100.0))

        # 3. Categorical Thresholding
        if final_score < 40.0:
            assigned_category = 'On Track'
        elif final_score < 70.0:
            assigned_category = 'In Progress'
        else:
            assigned_category = 'At Risk'

        confidence = max(p_track, p_prog, p_risk)
        execution_time_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "agent": self.name,
            "risk_score": round(final_score, 2),
            "risk_category": assigned_category,
            "ml_predicted_class": ml_pred_class,
            "ml_probabilities": {k: round(v, 4) for k, v in ml_probs.items()},
            "confidence": round(float(confidence), 4),
            "variant": variant,
            "execution_time_ms": round(execution_time_ms, 3)
        }
