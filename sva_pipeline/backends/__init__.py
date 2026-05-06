"""
LLM backend factory.

Supports three backends:
  - ``local``     : HuggingFace model on local GPU
  - ``openai``    : OpenAI-compatible API (GPT-4, vLLM, ollama)
  - ``anthropic`` : Anthropic Claude API

Usage:
    from sva_pipeline.backends import create_backend
    backend = create_backend(config)
    text, tool_calls = backend.generate(messages, tools)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_backend(config: Any):
    """
    Create the appropriate LLM backend based on ``config.backend``.

    Parameters
    ----------
    config : PipelineConfig
        Must have ``backend`` field ("local", "openai", or "anthropic").

    Returns
    -------
    LLMBackend
        An object implementing the ``generate(messages, tools)`` method.
    """
    backend = config.backend

    if backend == "local":
        logger.info("Using local HuggingFace backend: %s", config.model_id)
        from .local import LocalBackend
        return LocalBackend(config)

    elif backend == "openai":
        logger.info("Using OpenAI API backend: %s", config.model_id)
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(config)

    elif backend == "anthropic":
        logger.info("Using Anthropic Claude backend: %s", config.model_id)
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(config)

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'.\n"
            "Supported backends: 'local', 'openai', 'anthropic'"
        )
