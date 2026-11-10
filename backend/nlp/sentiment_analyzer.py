"""
sentiment_analyzer.py
======================
Natural Language Processing and Sentiment Analysis Component for IT Project Communications.

Analyzes team communications, stakeholder feedback, and sprint retrospective notes.
Designed with graceful fallback: uses HuggingFace pipeline when available,
and a deterministic rule-calibrated lexicon engine when HuggingFace/transformers is uninstalled.
"""

import time
import re
from typing import Dict, Any, List, Tuple


# Domain-specific sentiment and risk lexicon for software engineering contexts
POSITIVE_TERMS = {
    'ahead', 'stable', 'resolved', 'successful', 'passed', 'excellent', 'on track',
    'velocity', 'delivered', 'smooth', 'optimized', 'confidence', 'aligned', 'approved',
    'efficient', 'healthy', 'productive', 'collaborative', 'reliable', 'improved'
}

NEGATIVE_TERMS = {
    'delayed', 'blocker', 'failing', 'critical', 'overrun', 'buggy', 'attrition',
    'burnout', 'unstable', 'risk', 'deficit', 'behind', 'conflict', 'scope creep',
    'bottleneck', 'resignation', 'escalated', 'missed', 'frustration', 'regression'
}

AMPLIFIERS = {'severe', 'severely', 'extremely', 'very', 'highly', 'massively', 'critically'}


class ProjectSentimentAnalyzer:
    """
    NLP Sentiment Analyzer for project communications.
    """
    def __init__(self, use_hf_if_available: bool = True):
        self.hf_pipeline = None
        self.engine_type = "Domain Lexicon Engine (Local deterministic)"
        self.model_name = "IT-Project-Risk-Lexicon-v1"

        if use_hf_if_available:
            try:
                from transformers import pipeline
                self.hf_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                self.engine_type = "HuggingFace Transformers (DistilBERT)"
                self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            except Exception:
                # Graceful fallback: transformers not installed or no model downloaded
                self.hf_pipeline = None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single project communication snippet.

        Returns
        -------
        dict containing:
            - sentiment: 'Positive', 'Neutral', or 'Negative'
            - polarity_score: float in [-1.0, 1.0]
            - confidence: float in [0.0, 1.0]
            - inference_time_ms: float
            - engine: str
        """
        start_t = time.perf_counter()

        if self.hf_pipeline is not None:
            try:
                res = self.hf_pipeline(text[:512])[0]
                label_map = {'POSITIVE': 'Positive', 'NEGATIVE': 'Negative'}
                raw_label = res['label'].upper()
                sentiment = label_map.get(raw_label, 'Neutral')
                conf = float(res['score'])
                polarity = conf if sentiment == 'Positive' else -conf
                inference_ms = (time.perf_counter() - start_t) * 1000.0
                return {
                    'sentiment': sentiment,
                    'polarity_score': round(polarity, 4),
                    'confidence': round(conf, 4),
                    'inference_time_ms': round(inference_ms, 3),
                    'engine': self.engine_type
                }
            except Exception:
                pass

        # Local deterministic lexicon analysis
        clean_text = text.lower()
        words = re.findall(r'\b[a-z_\-]+\b', clean_text)
        
        pos_score = 0.0
        neg_score = 0.0

        for i, w in enumerate(words):
            amp = 1.5 if i > 0 and words[i-1] in AMPLIFIERS else 1.0
            if w in POSITIVE_TERMS:
                pos_score += 1.0 * amp
            elif w in NEGATIVE_TERMS:
                neg_score += 1.0 * amp

        # Check multi-word phrases
        for phrase in ['on track', 'scope creep']:
            if phrase in clean_text:
                if phrase == 'on track':
                    pos_score += 1.5
                else:
                    neg_score += 1.8

        total = pos_score + neg_score
        if total == 0:
            sentiment = 'Neutral'
            polarity = 0.0
            confidence = 0.50
        else:
            polarity = (pos_score - neg_score) / (pos_score + neg_score)
            if polarity > 0.15:
                sentiment = 'Positive'
                confidence = 0.5 + min(0.45, abs(polarity) * 0.45)
            elif polarity < -0.15:
                sentiment = 'Negative'
                confidence = 0.5 + min(0.45, abs(polarity) * 0.45)
            else:
                sentiment = 'Neutral'
                confidence = 0.60

        inference_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            'sentiment': sentiment,
            'polarity_score': round(float(polarity), 4),
            'confidence': round(float(confidence), 4),
            'inference_time_ms': round(float(inference_ms), 3),
            'engine': self.engine_type
        }
