"""
runtime_benchmark.py
====================
Multi-Agent High-Resolution Runtime Benchmark.

Measures latency of each functional agent component using `time.perf_counter()`:
- Project Risk Manager (Coordinator)
- Project Status Tracking Agent
- Market Analysis Agent
- Historical Knowledge Retrieval
- Risk Scoring Agent
- Reporting Agent
- End-to-end workflow total

Executes 30 repeated iterations to compute robust statistical measures:
Mean, Median, Standard Deviation, Minimum, Maximum, and 95th Percentile (P95).

Outputs:
- results/agent_runtime.csv (raw execution timings per iteration)
- results/agent_runtime_summary.json (comprehensive statistical metrics)
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from backend.agents.coordinator import ProjectRiskManager
from evaluation.data_generator import generate_evaluation_scenarios


def run_runtime_benchmark(
    n_iterations: int = 30,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Execute high-resolution runtime benchmark across 30 repeated iterations.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Executing Multi-Agent Runtime Benchmark (N = {n_iterations} iterations)")
    print("=" * 50)

    # 1. Measure Initialization & Model Loading Time
    init_start = time.perf_counter()
    manager = ProjectRiskManager()
    init_duration_ms = round((time.perf_counter() - init_start) * 1000.0, 3)
    print(f"[+] Multi-Agent System & Model Loading Time: {init_duration_ms:.2f} ms")

    # Load representative project scenarios for steady-state evaluation
    scenarios_df = generate_evaluation_scenarios(n_scenarios=n_iterations, random_seed=2026)

    sample_comm_texts = [
        "Team velocity has slowed down due to unpredicted database migration roadblocks.",
        "Sprint goal achieved ahead of schedule with all unit and integration tests passing.",
        "Stakeholder raised serious escalations regarding missing compliance reports and budget burn.",
        "Normal sprint cadence with steady burn-down; minor UI bugs resolved quickly."
    ]

    iteration_records = []

    # Warmup run to avoid JIT/caching skew
    sample_proj = scenarios_df.iloc[0].drop(['scenario_id', 'risk_category']).to_dict()
    manager.assess_project(sample_proj, project_id="WARMUP", communication_text=sample_comm_texts[0])

    print(f"[>] Benchmarking steady-state execution across {n_iterations} runs...")

    for i in range(n_iterations):
        proj_row = scenarios_df.iloc[i]
        scenario_id = proj_row['scenario_id']
        proj_data = proj_row.drop(['scenario_id', 'risk_category']).to_dict()
        comm_text = sample_comm_texts[i % len(sample_comm_texts)]

        res = manager.assess_project(
            project_data=proj_data,
            project_id=scenario_id,
            communication_text=comm_text,
            variant="complete"
        )

        timings = res['component_timings_ms']
        total_lat = res['total_latency_ms']

        record = {
            'Iteration': i + 1,
            'Scenario_ID': scenario_id,
            'Status_Tracking_Agent_ms': timings.get('Project Status Tracking Agent', 0.0),
            'Market_Analysis_Agent_ms': timings.get('Market Analysis Agent', 0.0),
            'Historical_Retrieval_ms': timings.get('Historical Knowledge Retrieval', 0.0),
            'Risk_Scoring_Agent_ms': timings.get('Risk Scoring Agent', 0.0),
            'Reporting_Agent_ms': timings.get('Reporting Agent', 0.0),
            'Coordinator_Overhead_ms': timings.get('Project Risk Manager (Coordinator)', 0.0),
            'Total_End_to_End_Latency_ms': total_lat
        }
        iteration_records.append(record)

    iter_df = pd.DataFrame(iteration_records)
    raw_csv_path = os.path.join(results_dir, "agent_runtime.csv")
    iter_df.to_csv(raw_csv_path, index=False)
    print(f"[+] Raw runtime records saved to {raw_csv_path}")

    # Compute Statistics for each component
    components = [
        ('Project Status Tracking Agent', 'Status_Tracking_Agent_ms'),
        ('Market Analysis Agent', 'Market_Analysis_Agent_ms'),
        ('Historical Knowledge Retrieval', 'Historical_Retrieval_ms'),
        ('Risk Scoring Agent', 'Risk_Scoring_Agent_ms'),
        ('Reporting Agent', 'Reporting_Agent_ms'),
        ('Project Risk Manager (Coordinator)', 'Coordinator_Overhead_ms'),
        ('Total End-to-End Workflow', 'Total_End_to_End_Latency_ms')
    ]

    summary_stats = {
        "benchmark_metadata": {
            "iterations": n_iterations,
            "timing_method": "time.perf_counter()",
            "unit": "milliseconds (ms)",
            "initialization_and_model_loading_ms": init_duration_ms,
            "runtime_environment": "Local Multi-Agent Evaluation Harness (Deterministic Benchmarked)",
            "external_credentials_status": "No external cloud API keys required (local execution)",
            "unmeasured_capabilities": [
                "External commercial LLM API cloud latency (e.g. OpenAI/Anthropic network RTT)"
            ]
        },
        "components": {}
    }

    print("\n" + "=" * 80)
    print(f"{'Component':<38} | {'Mean (ms)':<9} | {'Median':<8} | {'Std':<7} | {'Min':<6} | {'Max':<6} | {'P95':<6}")
    print("-" * 80)

    for comp_name, col in components:
        series = iter_df[col].values
        c_mean = float(np.mean(series))
        c_median = float(np.median(series))
        c_std = float(np.std(series))
        c_min = float(np.min(series))
        c_max = float(np.max(series))
        c_p95 = float(np.percentile(series, 95))

        print(f"{comp_name:<38} | {c_mean:<9.3f} | {c_median:<8.3f} | {c_std:<7.3f} | {c_min:<6.3f} | {c_max:<6.3f} | {c_p95:<6.3f}")

        summary_stats["components"][comp_name] = {
            "mean_ms": round(c_mean, 3),
            "median_ms": round(c_median, 3),
            "std_ms": round(c_std, 3),
            "min_ms": round(c_min, 3),
            "max_ms": round(c_max, 3),
            "p95_ms": round(c_p95, 3)
        }

    print("=" * 80)

    summary_json_path = os.path.join(results_dir, "agent_runtime_summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"[+] Agent runtime summary saved to {summary_json_path}")

    return {
        "raw_dataframe": iter_df,
        "summary": summary_stats
    }


if __name__ == "__main__":
    run_runtime_benchmark(n_iterations=30)
