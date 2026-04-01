"""
llm_monitor.py
LLM Observability — tracks token usage, latency, drift, and response quality.
"""

import time
from collections import defaultdict
from typing import List, Dict


class LLMMonitor:
    """
    Tracks production LLM metrics:
    - Token usage per query
    - Average response latency
    - Hallucination / low-confidence flags
    - Query volume over time
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []

    def log(
        self,
        query: str,
        response: str,
        latency_ms: float,
        token_count: int,
        confidence_score: float = 1.0,
    ):
        self.metrics["latency_ms"].append(latency_ms)
        self.metrics["token_count"].append(token_count)
        self.metrics["confidence_scores"].append(confidence_score)
        self.metrics["queries"].append(query)

        if confidence_score < 0.5:
            self.alerts.append({
                "type": "low_confidence",
                "query": query,
                "score": confidence_score,
                "timestamp": time.time(),
            })

        if latency_ms > 5000:
            self.alerts.append({
                "type": "high_latency",
                "latency_ms": latency_ms,
                "timestamp": time.time(),
            })

    def get_metrics(self) -> Dict:
        latencies = self.metrics["latency_ms"]
        tokens = self.metrics["token_count"]
        scores = self.metrics["confidence_scores"]

        return {
            "total_queries": len(latencies),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
            "avg_tokens_per_query": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "total_tokens_used": sum(tokens),
            "avg_confidence_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "alerts_count": len(self.alerts),
        }

    def drift_report(self, baseline_score: float = 0.85) -> Dict:
        """Compare current confidence scores against a baseline."""
        scores = self.metrics["confidence_scores"]
        if not scores:
            return {"status": "no_data"}

        current_avg = sum(scores[-50:]) / min(len(scores), 50)
        drift = round(baseline_score - current_avg, 4)

        return {
            "baseline_confidence": baseline_score,
            "current_avg_confidence": round(current_avg, 4),
            "drift": drift,
            "status": "drift_detected" if drift > 0.1 else "stable",
        }
