"""CLI and launcher script for Smart SLM Router proxy server."""

import argparse
import uvicorn
from smart_slm_router.config import settings


def cli():
    """CLI entrypoint to run Smart SLM Router ASGI server."""
    parser = argparse.ArgumentParser(description="Smart SLM Router & Fallback Proxy")
    parser.add_argument("--host", type=str, default=settings.host, help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()

    print(f"🚀 Starting Smart SLM Router Proxy on http://{args.host}:{args.port}")
    print(f"📦 SLM Endpoint: {settings.slm_base_url} ({settings.slm_model_name})")
    print(f"⚡ Frontier Endpoint: {settings.frontier_base_url} ({settings.frontier_model_name})")

    uvicorn.run(
        "smart_slm_router.proxy:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    cli()
