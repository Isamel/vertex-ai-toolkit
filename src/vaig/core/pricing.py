"""Model pricing data and cost calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Pricing per 1M tokens for a model."""

    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    thinking_per_1m: float | None = None  # USD per 1M thinking tokens (None → use output rate)


# Vertex AI pricing (USD per 1M tokens) — updated 2025-Q2.
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
# Models not listed here will show "N/A" for cost.
MODEL_PRICING: dict[str, ModelPricing] = {
    # Gemini 3.x series
    "gemini-3.1-pro-preview": ModelPricing(input_per_1m=1.25, output_per_1m=10.00, thinking_per_1m=10.00),
    "gemini-3.1-pro-preview-customtools": ModelPricing(input_per_1m=1.25, output_per_1m=10.00, thinking_per_1m=10.00),
    "gemini-3-flash-preview": ModelPricing(input_per_1m=0.15, output_per_1m=0.60, thinking_per_1m=0.60),
    "gemini-3.1-flash-lite-preview": ModelPricing(input_per_1m=0.02, output_per_1m=0.10),
    # Gemini 2.5 series
    "gemini-2.5-pro": ModelPricing(input_per_1m=1.25, output_per_1m=10.00, thinking_per_1m=10.00),
    "gemini-2.5-flash": ModelPricing(input_per_1m=0.15, output_per_1m=0.60, thinking_per_1m=0.60),
    # Gemini 2.0 series (deprecated but still available)
    "gemini-2.0-flash": ModelPricing(input_per_1m=0.10, output_per_1m=0.40),
    "gemini-2.0-flash-001": ModelPricing(input_per_1m=0.10, output_per_1m=0.40),
}


def calculate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    thinking_tokens: int = 0,
) -> float | None:
    """Calculate the estimated cost for a request.

    Returns the cost in USD, or None if the model has no pricing data.

    Args:
        model_id: Model identifier (must exist in MODEL_PRICING).
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        thinking_tokens: Number of thinking/reasoning tokens (default 0).
            Priced at ``thinking_per_1m`` if set, otherwise falls back to
            the output rate.
    """
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        return None

    input_cost = (prompt_tokens / 1_000_000) * pricing.input_per_1m
    output_cost = (completion_tokens / 1_000_000) * pricing.output_per_1m
    thinking_rate = pricing.thinking_per_1m if pricing.thinking_per_1m is not None else pricing.output_per_1m
    thinking_cost = (thinking_tokens / 1_000_000) * thinking_rate
    return input_cost + output_cost + thinking_cost


def format_cost(cost: float | None) -> str:
    """Format a cost value for display.

    Returns a string like "$0.0023" or "N/A" if cost is None.
    """
    if cost is None:
        return "N/A"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
