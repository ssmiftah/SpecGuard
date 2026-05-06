"""
OpenAI-compatible API backend.

Supports OpenAI (GPT-4, GPT-4o), Azure OpenAI, vLLM, ollama, and any
other server that implements the OpenAI chat completions API.

Tool definitions are already in OpenAI function-calling format, so they
are passed directly to the API without conversion.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """
    OpenAI-compatible API backend.

    Uses the ``openai`` Python SDK.  Supports custom base URLs for
    vLLM, ollama, or Azure OpenAI endpoints.
    """

    def __init__(self, config: Any):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI backend.\n"
                "Install it with: pip install openai"
            )

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No API key provided for OpenAI backend.\n"
                "Set 'model.api_key' in YAML or the OPENAI_API_KEY env var."
            )

        kwargs = {"api_key": api_key}
        if config.api_base:
            kwargs["base_url"] = config.api_base

        self.client = OpenAI(**kwargs)
        self.model = config.model_id
        self.config = config
        self._grammar: str = ""  # GBNF grammar for constrained generation
        logger.info("OpenAI backend initialised: model=%s", self.model)

    def set_grammar(self, gbnf: str) -> None:
        """
        Set a GBNF grammar for constrained generation.

        When set, the grammar is passed to vLLM/SGLang via ``extra_body``.
        This is a no-op for servers that don't support it (Ollama, OpenAI).
        """
        self._grammar = gbnf
        if gbnf:
            logger.info(
                "Grammar constraints enabled (%d chars). "
                "Requires vLLM or SGLang backend.",
                len(gbnf),
            )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Call the OpenAI chat completions API.

        Tool definitions are passed directly (already in OpenAI format).
        Returns (text, tool_calls, usage) where usage is the prompt /
        completion token counts from ``response.usage``.
        """
        # Clean messages: remove any fields the API doesn't expect.
        clean_messages = self._clean_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": clean_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
        }

        # Only include tools if they're provided.
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Stage 2.5: grammar-constrained generation (vLLM/SGLang only).
        # When a grammar is set AND no tools are active (tool-calling and
        # grammar constraints are mutually exclusive in most engines),
        # pass the GBNF grammar via extra_body.
        if self._grammar and not tools:
            kwargs["extra_body"] = {"guided_grammar": self._grammar}

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = choice.message.content or ""

        # Extract tool calls.
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": args,
                    "tool_call_id": tc.id,
                })

        # Capture usage — Ollama populates this via the OpenAI-compatible
        # endpoint. Default to zeros if the server omits the field.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if getattr(response, "usage", None) is not None:
            u = response.usage
            usage["prompt_tokens"] = getattr(u, "prompt_tokens", 0) or 0
            usage["completion_tokens"] = getattr(u, "completion_tokens", 0) or 0
            usage["total_tokens"] = getattr(u, "total_tokens", 0) or (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )

        return text, tool_calls, usage

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean messages for the OpenAI API.

        - Ensure 'tool' role messages have a 'tool_call_id'.
        - Remove non-standard fields.
        """
        cleaned = []
        for msg in messages:
            m = {"role": msg["role"], "content": msg.get("content", "")}

            if msg["role"] == "tool":
                # OpenAI requires tool_call_id on tool messages.
                m["tool_call_id"] = msg.get("tool_call_id", msg.get("name", "call_0"))
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                m["tool_calls"] = msg["tool_calls"]

            cleaned.append(m)
        return cleaned
