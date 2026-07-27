"""Configuration settings for RouteLLM using Pydantic v2."""

from typing import List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseSettings):
    """Cost per 1M tokens in USD."""
    input_cost_per_m: float = 0.15
    output_cost_per_m: float = 0.60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTELLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Proxy server host")
    port: int = Field(default=8000, description="Proxy server port")
    api_key: str = Field(default="sk-routellm-local", description="Master API Key for RouteLLM proxy authentication")

    # SLM (Tier 2 / Cheap Model) Settings
    slm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for local SLM or cheap remote API (e.g. Ollama or vLLM)"
    )
    slm_api_key: str = Field(default="ollama", description="API key for SLM provider")
    slm_model_name: str = Field(default="llama3:8b", description="Model identifier for SLM provider")
    slm_pricing: ModelPricing = Field(
        default_factory=lambda: ModelPricing(input_cost_per_m=0.15, output_cost_per_m=0.60)
    )

    # Frontier (Tier 1 / High-Capability Model) Settings
    frontier_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for Tier-1 Frontier model provider"
    )
    frontier_api_key: str = Field(default="", description="API key for Frontier provider (e.g. OpenAI/Anthropic)")
    frontier_model_name: str = Field(default="gpt-4o", description="Model identifier for Frontier provider")
    frontier_pricing: ModelPricing = Field(
        default_factory=lambda: ModelPricing(input_cost_per_m=2.50, output_cost_per_m=10.00)
    )

    # Router & Intent Classifier Thresholds
    max_slm_tokens: int = Field(
        default=450,
        description="Prompts exceeding this approximate token length route directly to Frontier model"
    )
    
    code_keywords: List[str] = Field(
        default=[
            "def ", "class ", "import ", "async def", "fn ", "struct ", "impl ",
            "SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM",
            "function", "const ", "let ", "var ", "return ",
            "```", "refactor", "debug", "unit test", "stack trace"
        ],
        description="Keywords signaling code generation or technical syntax needing Frontier model"
    )

    reasoning_keywords: List[str] = Field(
        default=[
            "step-by-step", "mathematical proof", "derive", "analyze in detail",
            "compare and contrast", "architectural pattern", "root cause",
            "explain why", "complex problem", "tradeoff analysis"
        ],
        description="Keywords signaling deep analytical reasoning requiring Frontier model"
    )

    # Fallback and Retry settings
    max_slm_retries: int = Field(default=1, description="Number of retries on SLM before failing over to Frontier")
    auto_fallback_enabled: bool = Field(
        default=True,
        description="If True, automatically route requests to Frontier when SLM fails or returns errors/refusal"
    )
    httpx_timeout_seconds: float = Field(default=60.0, description="HTTP request timeout in seconds")


settings = Settings()
