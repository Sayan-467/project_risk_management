"""
market_analyst.py
=================
Market Analysis Agent.

Monitors external information, stakeholder communications, and team sentiment signals.
Integrates NLP sentiment analysis to detect qualitative warning signs.
"""

import time
from typing import Dict, Any, List
from backend.nlp.sentiment_analyzer import ProjectSentimentAnalyzer


class MarketAnalysisAgent:
    """
    Evaluates qualitative communication sentiment, stakeholder perception,
    and external environmental risk factors.
    """
    def __init__(self):
        self.name = "Market Analysis Agent"
        self.role = "Qualitative NLP and External Context Analyst"
        self.nlp_analyzer = ProjectSentimentAnalyzer()

    def analyze_market_context(
        self,
        communication_text: str = "",
        communication_health: str = "Good",
        stakeholder_satisfaction: float = 4.0
    ) -> Dict[str, Any]:
        """
        Analyze qualitative project communications and stakeholder indicators.
        """
        start_t = time.perf_counter()

        # Run NLP sentiment analysis
        if not communication_text:
            # Fallback contextual text from categorical health
            communication_text = f"Communication health is reported as {communication_health} with stakeholder satisfaction rating of {stakeholder_satisfaction:.1f} out of 5."

        sentiment_res = self.nlp_analyzer.analyze_text(communication_text)

        # External risk modifier calculation
        polarity = sentiment_res['polarity_score']  # -1.0 to +1.0
        
        # Satisfaction factor: 5.0 is best (penalty 0), 1.0 is worst (penalty 25)
        satisfaction_penalty = max(0.0, (4.5 - stakeholder_satisfaction) * 7.0)

        # Sentiment penalty/bonus: negative polarity increases risk, positive reduces risk
        sentiment_modifier = - (polarity * 15.0)

        external_risk_score = max(0.0, min(100.0, 20.0 + satisfaction_penalty + sentiment_modifier))

        observations = []
        if sentiment_res['sentiment'] == 'Negative':
            observations.append(f"Negative communication sentiment detected (Polarity: {polarity:.2f}). Potential team friction or customer dissatisfaction.")
        elif sentiment_res['sentiment'] == 'Positive':
            observations.append(f"Positive communication sentiment observed (Polarity: {polarity:.2f}). Team morale and stakeholder alignment are favorable.")
        else:
            observations.append("Neutral communication sentiment observed across project channels.")

        if stakeholder_satisfaction < 3.0:
            observations.append(f"Low stakeholder satisfaction warning ({stakeholder_satisfaction:.1f} / 5.0).")

        execution_time_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "agent": self.name,
            "sentiment_analysis": sentiment_res,
            "external_risk_score": round(external_risk_score, 2),
            "observations": observations,
            "execution_time_ms": round(execution_time_ms, 3)
        }
