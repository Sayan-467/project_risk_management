"""
retrieval_evaluation.py
=======================
Evaluation of the Historical Knowledge Retrieval Component.

Evaluates semantic/vector search across representative IT project queries.
Measures:
- Per-query and aggregate retrieval latency (Mean, Median, P95) across 20 iterations
- Retrieval success rate (Hit Rate @ 1, Hit Rate @ 3) against defined relevance benchmarks
- Precision @ 3
- Engine identification (ChromaDB Vector Store vs TF-IDF Vector Cosine Engine)

Outputs:
- results/retrieval_benchmark.csv
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from backend.retrieval.historical_store import HistoricalRiskKnowledgeStore, HISTORICAL_CASES

# Predefined query suite with ground-truth relevant case IDs
BENCHMARK_QUERIES = [
    {
        "query_id": "Q-01",
        "query_text": "severe schedule delay due to scope creep and architect attrition",
        "target_tags": ["scope creep", "attrition", "schedule delay"],
        "relevant_case_ids": ["CASE-001", "CASE-003", "CASE-013"]
    },
    {
        "query_id": "Q-02",
        "query_text": "excessive bug count and test pass failure in healthcare UAT",
        "target_tags": ["defect", "healthcare", "uat"],
        "relevant_case_ids": ["CASE-002", "CASE-007", "CASE-015"]
    },
    {
        "query_id": "Q-03",
        "query_text": "cloud infrastructure budget overrun and spark query optimization",
        "target_tags": ["budget overrun", "cloud cost", "data lake"],
        "relevant_case_ids": ["CASE-009", "CASE-002", "CASE-006"]
    },
    {
        "query_id": "Q-04",
        "query_text": "team burnout and high resource utilization in retail delivery",
        "target_tags": ["burnout", "resource overutilization", "attrition"],
        "relevant_case_ids": ["CASE-003", "CASE-013"]
    },
    {
        "query_id": "Q-05",
        "query_text": "communication breakdown between hardware and cloud teams with unresolved blockers",
        "target_tags": ["communication", "iot", "blocker"],
        "relevant_case_ids": ["CASE-004", "CASE-010"]
    },
    {
        "query_id": "Q-06",
        "query_text": "successful fintech mobile rollout with high test pass rate and automated CI CD",
        "target_tags": ["on track", "fintech", "cicd"],
        "relevant_case_ids": ["CASE-005", "CASE-008", "CASE-012"]
    },
    {
        "query_id": "Q-07",
        "query_text": "regulatory compliance blocker and supply chain delay in aviation",
        "target_tags": ["compliance", "aviation", "blocker"],
        "relevant_case_ids": ["CASE-010", "CASE-006"]
    },
    {
        "query_id": "Q-08",
        "query_text": "AI chatbot hallucination risks and prompt safeguard failures",
        "target_tags": ["ai", "nlp", "hallucination"],
        "relevant_case_ids": ["CASE-011"]
    }
]


def evaluate_retrieval(
    n_iterations_per_query: int = 20,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Execute retrieval latency and relevance benchmark.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("Evaluating Historical Knowledge Retrieval Component")
    print("=" * 50)

    store = HistoricalRiskKnowledgeStore()
    print(f"[+] Active Retrieval Engine: {store.engine_type}")

    query_results = []
    all_latencies = []

    for q in BENCHMARK_QUERIES:
        q_id = q["query_id"]
        q_text = q["query_text"]
        rel_ids = set(q["relevant_case_ids"])

        latencies = []
        top_matches = []

        # Run repeated iterations to benchmark latency
        for _ in range(n_iterations_per_query):
            res = store.query(q_text, top_k=3)
            latencies.append(res["latency_ms"])
            top_matches = res["matches"]

        all_latencies.extend(latencies)

        # Relevance evaluation on top matches
        retrieved_ids = [m["id"] for m in top_matches]
        hit_at_1 = 1 if retrieved_ids and retrieved_ids[0] in rel_ids else 0
        hit_at_3 = 1 if any(rid in rel_ids for rid in retrieved_ids[:3]) else 0
        relevant_in_top3 = sum(1 for rid in retrieved_ids[:3] if rid in rel_ids)
        prec_at_3 = relevant_in_top3 / float(min(3, len(retrieved_ids))) if retrieved_ids else 0.0

        mean_lat = float(np.mean(latencies))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        print(f"[{q_id}] Hit@1: {hit_at_1} | Hit@3: {hit_at_3} | P@3: {prec_at_3:.2f} | Latency: {mean_lat:.3f} ms (P95: {p95_lat:.3f} ms)")

        query_results.append({
            "Query_ID": q_id,
            "Query_Text": q_text,
            "Engine": store.engine_type,
            "Hit_at_1": hit_at_1,
            "Hit_at_3": hit_at_3,
            "Precision_at_3": round(prec_at_3, 3),
            "Mean_Latency_ms": round(mean_lat, 3),
            "Median_Latency_ms": round(med_lat, 3),
            "P95_Latency_ms": round(p95_lat, 3),
            "Top_Retrieved_ID": retrieved_ids[0] if retrieved_ids else "None",
            "Top_Retrieved_Title": top_matches[0]["title"] if top_matches else "None"
        })

    results_df = pd.DataFrame(query_results)
    csv_path = os.path.join(results_dir, "retrieval_benchmark.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Retrieval benchmark saved to {csv_path}")

    # Summary
    hit1_rate = results_df["Hit_at_1"].mean()
    hit3_rate = results_df["Hit_at_3"].mean()
    mean_p3 = results_df["Precision_at_3"].mean()
    overall_mean_lat = np.mean(all_latencies)
    overall_med_lat = np.median(all_latencies)
    overall_p95_lat = np.percentile(all_latencies, 95)

    print("\n--- Retrieval Summary Metrics ---")
    print(f"Hit Rate @ 1:          {hit1_rate * 100:.1f}%")
    print(f"Hit Rate @ 3:          {hit3_rate * 100:.1f}%")
    print(f"Mean Precision @ 3:    {mean_p3:.3f}")
    print(f"Overall Mean Latency:  {overall_mean_lat:.3f} ms")
    print(f"Overall P95 Latency:   {overall_p95_lat:.3f} ms")

    return {
        "results_df": results_df,
        "hit_rate_at_1": round(float(hit1_rate), 4),
        "hit_rate_at_3": round(float(hit3_rate), 4),
        "mean_precision_at_3": round(float(mean_p3), 4),
        "mean_latency_ms": round(float(overall_mean_lat), 3),
        "median_latency_ms": round(float(overall_med_lat), 3),
        "p95_latency_ms": round(float(overall_p95_lat), 3),
        "engine": store.engine_type
    }


if __name__ == "__main__":
    evaluate_retrieval(n_iterations_per_query=20)
