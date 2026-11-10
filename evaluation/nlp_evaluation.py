"""
nlp_evaluation.py
=================
Quantitative Evaluation of NLP Sentiment Analysis Component.

Evaluates sentiment classification on 60 synthetic project communication samples
spanning positive, neutral, and negative real-world development scenarios.

Calculates:
- Sentiment classification accuracy, macro/weighted precision, recall, and F1
- Per-class precision, recall, F1
- Confusion matrix
- Inference latency statistics (Mean, Median, Std, P95)

Outputs:
- results/nlp_evaluation.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from backend.nlp.sentiment_analyzer import ProjectSentimentAnalyzer

# Curated synthetic project communication samples with ground truth sentiment labels
COMMUNICATION_SAMPLES = [
    # Positive Samples (20)
    {"text": "Sprint goals were delivered ahead of schedule with zero critical blockers.", "label": "Positive"},
    {"text": "Great job team! All integration tests passed cleanly and client demo was approved.", "label": "Positive"},
    {"text": "The performance optimization improved database throughput by 40% and latency is minimal.", "label": "Positive"},
    {"text": "Architecture refactor is highly stable, reliable, and our release velocity is accelerating.", "label": "Positive"},
    {"text": "Stakeholder approved milestone deliverables and praised the team's collaborative efficiency.", "label": "Positive"},
    {"text": "CI CD pipelines are running smoothly; unit test coverage reached an impressive 92%.", "label": "Positive"},
    {"text": "Smooth deployment to staging with positive feedback from internal security auditors.", "label": "Positive"},
    {"text": "Excellent progress on the API gateway; cross-functional dependencies were successfully resolved.", "label": "Positive"},
    {"text": "Engineering team feels confident, aligned, and motivated for the final rollout sprint.", "label": "Positive"},
    {"text": "Code reviews are thorough and healthy, reducing technical debt significantly.", "label": "Positive"},
    {"text": "Customer satisfaction survey returned glowing reviews on our streamlined onboarding flow.", "label": "Positive"},
    {"text": "We delivered all contracted scope items on time with zero budget overrun.", "label": "Positive"},
    {"text": "Bug resolution velocity is exceeding ticket incoming rate; QA sign-off is complete.", "label": "Positive"},
    {"text": "The new caching layer works reliably under stress testing, keeping cpu load efficient.", "label": "Positive"},
    {"text": "Clear requirements and proactive communication have kept our development roadmap on track.", "label": "Positive"},
    {"text": "Successful beta user pilot completed with excellent adoption metrics.", "label": "Positive"},
    {"text": "All high-severity vulnerabilities were remediated and verified passed.", "label": "Positive"},
    {"text": "Team capacity is fully optimized and daily standups are crisp and productive.", "label": "Positive"},
    {"text": "Seamless migration to the cloud platform with no unexpected downtime.", "label": "Positive"},
    {"text": "Stakeholders are very pleased with the intuitive UI design and fast query times.", "label": "Positive"},

    # Neutral Samples (20)
    {"text": "The team held the bi-weekly sprint planning meeting and assigned story points as scheduled.", "label": "Neutral"},
    {"text": "Standard sprint retro completed; backlog grooming scheduled for tomorrow morning.", "label": "Neutral"},
    {"text": "Ticket triage is ongoing; 12 routine UI change requests were logged in Jira.", "label": "Neutral"},
    {"text": "We updated third-party dependency versions according to the quarterly maintenance calendar.", "label": "Neutral"},
    {"text": "Architecture documentation was uploaded to the internal team wiki.", "label": "Neutral"},
    {"text": "Database backup verification completed; disk usage is at 62% storage capacity.", "label": "Neutral"},
    {"text": "Standup notes: frontend engineers are implementing designs according to Figma wireframes.", "label": "Neutral"},
    {"text": "Sprint burn-down chart reflects average progress matching baseline velocity estimates.", "label": "Neutral"},
    {"text": "Code freeze scheduled for next Thursday prior to planned maintenance window.", "label": "Neutral"},
    {"text": "Security certificates renewed across internal development and staging endpoints.", "label": "Neutral"},
    {"text": "Sprint review conducted with client product owner; next milestone targets reviewed.", "label": "Neutral"},
    {"text": "API schemas were shared with the mobile integration team for initial review.", "label": "Neutral"},
    {"text": "Log aggregation services rotated indices according to standard retention policy.", "label": "Neutral"},
    {"text": "Two pull requests submitted for peer review in the authentication module.", "label": "Neutral"},
    {"text": "Performance profiling benchmarks recorded for the checkout transaction pipeline.", "label": "Neutral"},
    {"text": "Testing environment refreshed with sanitized production schema snapshot.", "label": "Neutral"},
    {"text": "Weekly status email dispatched to stakeholders summarizing completed backlog items.", "label": "Neutral"},
    {"text": "Daily sync completed; no blockers noted; team working through assigned sprint cards.", "label": "Neutral"},
    {"text": "Dependency scan executed; no novel high priority findings identified.", "label": "Neutral"},
    {"text": "Sprint scope remains unchanged as we approach midpoint of sprint 14.", "label": "Neutral"},

    # Negative Samples (20)
    {"text": "Severe schedule delay of 3 weeks due to critical API blockers and team attrition.", "label": "Negative"},
    {"text": "Testing team uncovered 28 critical bugs in UAT; staging deployment failed completely.", "label": "Negative"},
    {"text": "Massive budget overrun; cloud compute costs escalated far beyond allocated budget limits.", "label": "Negative"},
    {"text": "Key lead architect resigned yesterday, leaving severe knowledge deficits in core subsystems.", "label": "Negative"},
    {"text": "Stakeholder expressed extreme frustration with missed delivery commitments and frequent crashes.", "label": "Negative"},
    {"text": "Uncontrolled scope creep has bloated the backlog, causing engineer burnout and team conflict.", "label": "Negative"},
    {"text": "Production service crashed under peak traffic; load balancer configuration is totally broken.", "label": "Negative"},
    {"text": "Critical data corruption defect identified in customer payment settlement microservice.", "label": "Negative"},
    {"text": "Regression test suite pass rate dropped drastically to 54%; build is currently red.", "label": "Negative"},
    {"text": "Communication breakdown between hardware and firmware groups is halting all testing progress.", "label": "Negative"},
    {"text": "Customer escalations have tripled; client threatens contract termination over unaddressed bugs.", "label": "Negative"},
    {"text": "Overtime burnout is rampant as the team struggles to meet an unrealistic management deadline.", "label": "Negative"},
    {"text": "Third-party vendor API broke backwards compatibility, creating severe operational bottlenecks.", "label": "Negative"},
    {"text": "Memory leak in background workers caused recursive out-of-memory container crashes.", "label": "Negative"},
    {"text": "Project is severely behind schedule with 18 unresolved blocker tickets lingering for weeks.", "label": "Negative"},
    {"text": "Budget deficit is alarming; project burn rate exceeds safe operational thresholds by 45%.", "label": "Negative"},
    {"text": "Disastrous sprint review; client rejected all deliverable milestones due to missing requirements.", "label": "Negative"},
    {"text": "Team morale is deteriorating rapidly due to toxic blame culture and shifting priorities.", "label": "Negative"},
    {"text": "Security audit failed with multiple critical severity compliance vulnerabilities.", "label": "Negative"},
    {"text": "Database deadlock cascade triggered an emergency outage across all customer tenancies.", "label": "Negative"}
]


def evaluate_nlp_sentiment(
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run quantitative evaluation of the sentiment analysis component.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Evaluating NLP Sentiment Analysis Component (N = {len(COMMUNICATION_SAMPLES)} samples)")
    print("=" * 50)

    analyzer = ProjectSentimentAnalyzer()
    print(f"[+] Active NLP Engine: {analyzer.engine_type} (Model: {analyzer.model_name})")

    y_true = []
    y_pred = []
    latencies = []
    sample_outputs = []

    classes = ["Positive", "Neutral", "Negative"]

    for item in COMMUNICATION_SAMPLES:
        t_start = time.perf_counter()
        res = analyzer.analyze_text(item["text"])
        lat = (time.perf_counter() - t_start) * 1000.0

        y_true.append(item["label"])
        y_pred.append(res["sentiment"])
        latencies.append(lat)

        sample_outputs.append({
            "text": item["text"],
            "ground_truth": item["label"],
            "predicted": res["sentiment"],
            "polarity": res["polarity_score"],
            "confidence": res["confidence"],
            "latency_ms": round(lat, 3)
        })

    # Metrics Calculation
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)

    report_dict = classification_report(
        y_true, y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print("\n--- NLP Sentiment Classification Results ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro Precision:  {prec_macro:.4f}")
    print(f"Macro Recall:     {rec_macro:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Negative Recall:  {report_dict['Negative']['recall']:.4f}")
    print(f"Mean Latency:     {np.mean(latencies):.3f} ms")

    summary = {
        "nlp_engine": analyzer.engine_type,
        "model_name": analyzer.model_name,
        "sample_count": len(COMMUNICATION_SAMPLES),
        "target_classes": classes,
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
            } for cls_name in classes
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": classes
        },
        "latency_stats_ms": {
            "mean": round(float(np.mean(latencies)), 3),
            "median": round(float(np.median(latencies)), 3),
            "std": round(float(np.std(latencies)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3)
        },
        "sample_evaluations": sample_outputs[:6]
    }

    json_path = os.path.join(results_dir, "nlp_evaluation.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"[+] NLP evaluation summary saved to {json_path}")

    return summary


if __name__ == "__main__":
    evaluate_nlp_sentiment()
