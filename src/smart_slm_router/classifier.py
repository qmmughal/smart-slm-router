"""Low-latency Intent Classifier for prompt routing (<15ms target latency)."""

import re
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import tiktoken

from smart_slm_router.config import settings


class ClassificationResult(BaseModel):
    target: str  # "SLM" or "FRONTIER"
    routed_model: str
    complexity_score: float  # 0.0 (very simple) to 1.0 (very complex)
    confidence: float  # 0.0 to 1.0
    prompt_tokens: int
    reason: str
    decision_latency_ms: float


class IntentClassifier:
    """Fast, zero-bloat classifier executing token heuristics and pattern matching under 15ms."""

    def __init__(self):
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

        # Pre-compile regexes for fast sub-millisecond evaluation
        code_patterns = [re.escape(k) for k in settings.code_keywords]
        self._code_regex = re.compile("|".join(code_patterns), re.IGNORECASE)

        reasoning_patterns = [re.escape(k) for k in settings.reasoning_keywords]
        self._reasoning_regex = re.compile("|".join(reasoning_patterns), re.IGNORECASE)

    def count_tokens(self, text: str) -> int:
        """Estimate token count with tiktoken or fallback fast estimate."""
        if self._encoder:
            try:
                return len(self._encoder.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback estimate: ~4 chars per token
        return max(1, len(text) // 4)

    def extract_full_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Extract concatenated text content from OpenAI chat completion messages payload."""
        prompt_parts = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                prompt_parts.append(content)
            elif isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        prompt_parts.append(chunk.get("text", ""))
        return "\n".join(prompt_parts)

    def classify(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        requested_model: Optional[str] = None
    ) -> ClassificationResult:
        """Analyze request complexity and determine whether to route to SLM or Frontier."""
        start_time = time.perf_counter()

        full_prompt = self.extract_full_prompt(messages)
        prompt_tokens = self.count_tokens(full_prompt)

        reasons: List[str] = []
        complexity_score = 0.0

        # Rule 0: Multi-step tool use / functions automatically request Frontier capability
        if tools and len(tools) > 0:
            complexity_score += 0.5
            reasons.append("Tool/function calling active")

        # Rule 1: Length heuristic
        if prompt_tokens > settings.max_slm_tokens:
            complexity_score += 0.4
            reasons.append(f"Prompt length ({prompt_tokens} tokens) exceeds threshold ({settings.max_slm_tokens})")
        else:
            length_ratio = prompt_tokens / settings.max_slm_tokens
            complexity_score += length_ratio * 0.2

        # Rule 2: Code pattern matching
        code_matches = self._code_regex.findall(full_prompt)
        if code_matches:
            match_count = len(code_matches)
            complexity_score += min(0.4, 0.2 + match_count * 0.05)
            reasons.append(f"Detected technical/code keywords ({', '.join(set(code_matches[:3]))})")

        # Rule 3: Deep reasoning matching
        reasoning_matches = self._reasoning_regex.findall(full_prompt)
        if reasoning_matches:
            complexity_score += 0.35
            reasons.append(f"Detected analytical reasoning keywords ({', '.join(set(reasoning_matches[:3]))})")

        # Normalize complexity score between 0.0 and 1.0
        complexity_score = min(1.0, round(complexity_score, 3))

        # Decision threshold (score >= 0.45 escalation)
        if complexity_score >= 0.45:
            target = "FRONTIER"
            routed_model = settings.frontier_model_name
            confidence = min(0.99, 0.6 + complexity_score * 0.4)
            final_reason = "; ".join(reasons) if reasons else "High prompt complexity"
        else:
            target = "SLM"
            routed_model = settings.slm_model_name
            confidence = min(0.99, 0.9 - complexity_score * 0.5)
            final_reason = "Simple/Short query suitable for SLM"

        end_time = time.perf_counter()
        decision_latency_ms = round((end_time - start_time) * 1000, 3)

        return ClassificationResult(
            target=target,
            routed_model=routed_model,
            complexity_score=complexity_score,
            confidence=round(confidence, 3),
            prompt_tokens=prompt_tokens,
            reason=final_reason,
            decision_latency_ms=decision_latency_ms
        )
