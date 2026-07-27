"""Observability and Cost Tracking module for RouteLLM proxy."""

import time
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from routellm.config import settings


class RequestLogEntry(BaseModel):
    timestamp: float
    incoming_model: str
    target_backend: str
    routed_model: str
    fallback_triggered: bool = False
    decision_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_frontier_usd: float
    estimated_cost_actual_usd: float
    estimated_cost_saved_usd: float


class SystemMetrics(BaseModel):
    uptime_seconds: float
    total_requests: int
    slm_routed_count: int
    frontier_routed_count: int
    fallback_escalations_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_saved_usd: float
    avg_decision_latency_ms: float
    avg_total_latency_ms: float


class MetricsTracker:
    """Thread-safe metrics aggregator for measuring router performance and cost savings."""

    def __init__(self):
        self._start_time = time.time()
        self._lock = asyncio.Lock()
        self.total_requests = 0
        self.slm_routed_count = 0
        self.frontier_routed_count = 0
        self.fallback_escalations_count = 0

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_saved_usd = 0.0

        self._total_decision_latency_ms = 0.0
        self._total_request_latency_ms = 0.0

        self.recent_logs: List[RequestLogEntry] = []
        self._max_recent_logs = 100

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, input_cost_per_m: float, output_cost_per_m: float) -> float:
        input_cost = (prompt_tokens / 1_000_000) * input_cost_per_m
        output_cost = (completion_tokens / 1_000_000) * output_cost_per_m
        return input_cost + output_cost

    async def record_request(
        self,
        incoming_model: str,
        target_backend: str,
        routed_model: str,
        fallback_triggered: bool,
        decision_latency_ms: float,
        total_latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int
    ) -> RequestLogEntry:
        """Record a completed request and calculate cost savings."""
        total_tokens = prompt_tokens + completion_tokens

        # Baseline cost if all queries went to Frontier model
        frontier_cost = self.calculate_cost(
            prompt_tokens, completion_tokens,
            settings.frontier_pricing.input_cost_per_m,
            settings.frontier_pricing.output_cost_per_m
        )

        # Actual cost based on routed model (SLM or Frontier)
        if target_backend == "SLM" and not fallback_triggered:
            actual_cost = self.calculate_cost(
                prompt_tokens, completion_tokens,
                settings.slm_pricing.input_cost_per_m,
                settings.slm_pricing.output_cost_per_m
            )
        else:
            actual_cost = frontier_cost

        cost_saved = max(0.0, frontier_cost - actual_cost)

        entry = RequestLogEntry(
            timestamp=time.time(),
            incoming_model=incoming_model,
            target_backend=target_backend,
            routed_model=routed_model,
            fallback_triggered=fallback_triggered,
            decision_latency_ms=decision_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_frontier_usd=round(frontier_cost, 6),
            estimated_cost_actual_usd=round(actual_cost, 6),
            estimated_cost_saved_usd=round(cost_saved, 6)
        )

        async with self._lock:
            self.total_requests += 1
            if target_backend == "SLM":
                self.slm_routed_count += 1
            else:
                self.frontier_routed_count += 1

            if fallback_triggered:
                self.fallback_escalations_count += 1

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_saved_usd += cost_saved

            self._total_decision_latency_ms += decision_latency_ms
            self._total_request_latency_ms += total_latency_ms

            self.recent_logs.append(entry)
            if len(self.recent_logs) > self._max_recent_logs:
                self.recent_logs.pop(0)

        return entry

    async def get_summary(self) -> SystemMetrics:
        """Get aggregate system performance and savings metrics."""
        async with self._lock:
            total_req = self.total_requests or 1
            return SystemMetrics(
                uptime_seconds=round(time.time() - self._start_time, 2),
                total_requests=self.total_requests,
                slm_routed_count=self.slm_routed_count,
                frontier_routed_count=self.frontier_routed_count,
                fallback_escalations_count=self.fallback_escalations_count,
                total_prompt_tokens=self.total_prompt_tokens,
                total_completion_tokens=self.total_completion_tokens,
                total_tokens=self.total_prompt_tokens + self.total_completion_tokens,
                total_cost_saved_usd=round(self.total_cost_saved_usd, 6),
                avg_decision_latency_ms=round(self._total_decision_latency_ms / total_req, 3),
                avg_total_latency_ms=round(self._total_request_latency_ms / total_req, 3)
            )


metrics_tracker = MetricsTracker()
