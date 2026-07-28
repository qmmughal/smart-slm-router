"""Transparent OpenAI API Proxy with Smart SLM Routing and Fallback Escalation."""

import json
import time
import logging
from typing import AsyncGenerator, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from smart_slm_router.config import settings
from smart_slm_router.classifier import IntentClassifier
from smart_slm_router.metrics import metrics_tracker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("smart_slm_router.proxy")

app = FastAPI(
    title="Smart SLM Router Proxy",
    description="Smart SLM Router & Fallback Proxy for OpenAI Compatible Applications",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = IntentClassifier()

# Global async HTTP client with connection pooling
httpx_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup_event():
    global httpx_client
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.httpx_timeout_seconds),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
    )
    logger.info(f"Smart SLM Router started. SLM Endpoint: {settings.slm_base_url} ({settings.slm_model_name}) | Frontier Endpoint: {settings.frontier_base_url} ({settings.frontier_model_name})")


@app.on_event("shutdown")
async def shutdown_event():
    global httpx_client
    if httpx_client:
        await httpx_client.aclose()
    logger.info("Smart SLM Router proxy shut down successfully.")


def get_backend_params(target: str) -> tuple[str, str, str]:
    """Return (base_url, api_key, model_name) for target backend."""
    if target == "SLM":
        return settings.slm_base_url.rstrip("/"), settings.slm_api_key, settings.slm_model_name
    else:
        return settings.frontier_base_url.rstrip("/"), settings.frontier_api_key, settings.frontier_model_name


async def forward_non_stream_request(
    backend_url: str,
    api_key: str,
    model_name: str,
    payload: Dict[str, Any]
) -> tuple[Dict[str, Any], int]:
    """Forward standard non-streaming payload to target backend."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else ""
    }
    
    # Override model name with target backend model
    forward_payload = {**payload, "model": model_name}

    url = f"{backend_url}/chat/completions"
    response = await httpx_client.post(url, json=forward_payload, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Backend {url} returned status {response.status_code}: {response.text}"
        )
    
    try:
        data = response.json()
        return data, response.status_code
    except Exception as e:
        raise ValueError(f"Failed to parse backend JSON response: {e}")


async def stream_chunk_generator(
    response: httpx.Response,
    incoming_model: str,
    target_backend: str,
    routed_model: str,
    fallback_triggered: bool,
    decision_latency_ms: float,
    start_time: float,
    prompt_tokens: int
) -> AsyncGenerator[bytes, None]:
    """Forward SSE stream chunks to client and record token metrics upon completion."""
    completion_tokens_estimate = 0

    try:
        async for line in response.aiter_lines():
            if not line:
                yield b"\n"
                continue

            yield f"{line}\n\n".encode("utf-8")

            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            completion_tokens_estimate += max(1, len(content) // 4)
                except Exception:
                    pass

    except Exception as err:
        logger.error(f"Error while streaming response: {err}")
    finally:
        await response.aclose()
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        await metrics_tracker.record_request(
            incoming_model=incoming_model,
            target_backend=target_backend,
            routed_model=routed_model,
            fallback_triggered=fallback_triggered,
            decision_latency_ms=decision_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens_estimate
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI compatible chat completions route with intelligent routing and automatic fallback."""
    start_time = time.perf_counter()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request payload")

    messages = body.get("messages", [])
    tools = body.get("tools", None)
    is_stream = body.get("stream", False)
    incoming_model = body.get("model", "default")

    # Step 1: Low-latency intent classification (<15ms)
    classification = classifier.classify(messages=messages, tools=tools, requested_model=incoming_model)
    
    target_backend = classification.target
    routed_model = classification.routed_model
    decision_latency_ms = classification.decision_latency_ms
    prompt_tokens = classification.prompt_tokens

    logger.info(
        f"[{classification.decision_latency_ms}ms] Routed '{incoming_model}' -> {target_backend} ({routed_model}) | Reason: {classification.reason}"
    )

    fallback_triggered = False

    # Step 2: Forward to chosen target backend
    base_url, api_key, backend_model = get_backend_params(target_backend)

    if not is_stream:
        # Non-streaming request with automatic fallback escalation
        try:
            res_data, status_code = await forward_non_stream_request(
                base_url, api_key, backend_model, body
            )
            
            # Check for refusal / empty completion fallback triggers if target was SLM
            choices = res_data.get("choices", [])
            has_content = choices and choices[0].get("message", {}).get("content")
            if target_backend == "SLM" and not has_content and settings.auto_fallback_enabled:
                logger.warning("SLM returned empty or invalid completion. Triggering Frontier fallback...")
                raise ValueError("Empty SLM completion response")

        except Exception as slm_error:
            if target_backend == "SLM" and settings.auto_fallback_enabled:
                logger.warning(f"SLM backend error: {slm_error}. Escalating to Frontier model ({settings.frontier_model_name})...")
                fallback_triggered = True
                target_backend = "FRONTIER"
                base_url, api_key, backend_model = get_backend_params("FRONTIER")
                routed_model = backend_model
                
                try:
                    res_data, status_code = await forward_non_stream_request(
                        base_url, api_key, backend_model, body
                    )
                except Exception as frontier_error:
                    logger.error(f"Frontier backend error: {frontier_error}")
                    raise HTTPException(status_code=502, detail=f"All backends failed. Frontier error: {frontier_error}")
            else:
                raise HTTPException(status_code=502, detail=str(slm_error))

        # Extract completion token usage
        usage = res_data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if not completion_tokens:
            choices = res_data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            completion_tokens = max(1, len(content) // 4)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Record metrics
        log_entry = await metrics_tracker.record_request(
            incoming_model=incoming_model,
            target_backend=target_backend,
            routed_model=routed_model,
            fallback_triggered=fallback_triggered,
            decision_latency_ms=decision_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )

        # Inject Smart SLM Router metadata in response JSON
        res_data["router_meta"] = {
            "routed_backend": target_backend,
            "routed_model": routed_model,
            "fallback_triggered": fallback_triggered,
            "decision_latency_ms": decision_latency_ms,
            "estimated_cost_saved_usd": log_entry.estimated_cost_saved_usd
        }

        return JSONResponse(content=res_data, status_code=status_code)

    else:
        # Streaming SSE request
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
        forward_payload = {**body, "model": backend_model}

        try:
            req = httpx_client.build_request("POST", url, json=forward_payload, headers=headers)
            response = await httpx_client.send(req, stream=True)
            
            if response.status_code != 200 and target_backend == "SLM" and settings.auto_fallback_enabled:
                await response.aclose()
                logger.warning(f"SLM streaming returned status {response.status_code}. Escalating to Frontier model...")
                fallback_triggered = True
                target_backend = "FRONTIER"
                base_url, api_key, backend_model = get_backend_params("FRONTIER")
                routed_model = backend_model
                
                url = f"{base_url}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}" if api_key else ""}
                forward_payload = {**body, "model": backend_model}
                req = httpx_client.build_request("POST", url, json=forward_payload, headers=headers)
                response = await httpx_client.send(req, stream=True)

            return StreamingResponse(
                stream_chunk_generator(
                    response=response,
                    incoming_model=incoming_model,
                    target_backend=target_backend,
                    routed_model=routed_model,
                    fallback_triggered=fallback_triggered,
                    decision_latency_ms=decision_latency_ms,
                    start_time=start_time,
                    prompt_tokens=prompt_tokens
                ),
                media_type="text/event-stream"
            )
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise HTTPException(status_code=502, detail=f"Proxy streaming failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint showing server status and configured endpoints."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "slm_backend": {
            "base_url": settings.slm_base_url,
            "model": settings.slm_model_name
        },
        "frontier_backend": {
            "base_url": settings.frontier_base_url,
            "model": settings.frontier_model_name
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Observability endpoint providing detailed usage and cost savings metrics."""
    summary = await metrics_tracker.get_summary()
    return {
        "metrics": summary.dict(),
        "recent_requests": [log.dict() for log in metrics_tracker.recent_logs[-10:]]
    }
