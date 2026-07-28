"""Sample client script demonstrating how Python applications connect to Smart SLM Router proxy."""

import json
import time
import httpx


PROXY_URL = "http://localhost:8000/v1"
METRICS_URL = "http://localhost:8000/metrics"
API_KEY = "sk-smart-slm-local"


def test_simple_query():
    print("\n--- Test 1: Simple Prompt (Expected: SLM Routing) ---")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    start = time.time()
    try:
        response = httpx.post(f"{PROXY_URL}/chat/completions", json=payload, headers=headers, timeout=10.0)
        elapsed = round((time.time() - start) * 1000, 2)
        
        if response.status_code == 200:
            data = response.json()
            meta = data.get("router_meta", {})
            content = data["choices"][0]["message"]["content"]
            print(f"Status: 200 OK ({elapsed}ms)")
            print(f"Routed Backend : {meta.get('routed_backend')} ({meta.get('routed_model')})")
            print(f"Decision Latency: {meta.get('decision_latency_ms')}ms")
            print(f"Cost Saved USD  : ${meta.get('estimated_cost_saved_usd'):.6f}")
            print(f"Response Content: {content.strip()}")
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection error (Ensure Smart SLM Router proxy is running on localhost:8000): {e}")


def test_complex_code_query():
    print("\n--- Test 2: Complex Code Generation Prompt (Expected: Frontier Escalation) ---")
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a production-ready, thread-safe async bounded queue class in Python "
                    "using `asyncio.Condition` with type annotations, docstrings, and comprehensive "
                    "unit test assertions using `pytest`. Explain step-by-step."
                )
            }
        ]
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    start = time.time()
    try:
        response = httpx.post(f"{PROXY_URL}/chat/completions", json=payload, headers=headers, timeout=30.0)
        elapsed = round((time.time() - start) * 1000, 2)
        
        if response.status_code == 200:
            data = response.json()
            meta = data.get("router_meta", {})
            print(f"Status: 200 OK ({elapsed}ms)")
            print(f"Routed Backend : {meta.get('routed_backend')} ({meta.get('routed_model')})")
            print(f"Decision Latency: {meta.get('decision_latency_ms')}ms")
            print(f"Cost Saved USD  : ${meta.get('estimated_cost_saved_usd'):.6f}")
            print("Response Content Preview:", data["choices"][0]["message"]["content"][:150] + "...")
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection error: {e}")


def test_streaming_query():
    print("\n--- Test 3: Streaming SSE Response (stream=True) ---")
    payload = {
        "model": "llama3:8b",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5 slowly."}
        ],
        "stream": True
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        with httpx.stream("POST", f"{PROXY_URL}/chat/completions", json=payload, headers=headers, timeout=10.0) as response:
            print(f"Streaming status: {response.status_code}")
            print("Chunks received: ", end="", flush=True)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content != "[DONE]":
                        try:
                            chunk = json.loads(content)
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            print(delta, end="", flush=True)
                        except Exception:
                            pass
            print()
    except Exception as e:
        print(f"Streaming error: {e}")


def fetch_metrics():
    print("\n--- Smart SLM Router Metrics Summary ---")
    try:
        res = httpx.get(METRICS_URL, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            print(json.dumps(data["metrics"], indent=2))
        else:
            print(f"Failed to fetch metrics: {res.status_code}")
    except Exception as e:
        print(f"Metrics fetch error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("   Smart SLM Router Proxy Integration Test Suite")
    print("=" * 60)
    test_simple_query()
    test_complex_code_query()
    test_streaming_query()
    fetch_metrics()
