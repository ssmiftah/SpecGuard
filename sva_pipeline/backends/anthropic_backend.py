"""
Anthropic Claude API backend.

Supports Claude models (Sonnet, Opus, Haiku) via the Anthropic Python SDK.
Tool definitions are converted from OpenAI format to Anthropic's tool use
format automatically.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class AnthropicBackend:
    """
    Anthropic Claude API backend.

    Handles the differences between Anthropic's message format and the
    pipeline's OpenAI-style format:
      - System message is a separate parameter (not in messages list).
      - Tool results use ``tool_result`` content blocks.
      - Tool calls appear as ``tool_use`` content blocks in responses.
    """

    def __init__(self, config: Any):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic backend.\n"
                "Install it with: pip install anthropic"
            )

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No API key provided for Anthropic backend.\n"
                "Set 'model.api_key' in YAML or the ANTHROPIC_API_KEY env var."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = config.model_id
        self.config = config
        logger.info("Anthropic backend initialised: model=%s", self.model)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Call the Anthropic messages API.

        Converts OpenAI-format messages and tools to Anthropic format.
        Returns (text, tool_calls, usage) with prompt / completion token
        counts.
        """
        # Separate system message from conversation.
        system = ""
        conv_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            elif msg["role"] == "tool":
                # Anthropic expects tool results as user messages with
                # tool_result content blocks.
                conv_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", msg.get("name", "call_0")),
                        "content": msg.get("content", ""),
                    }],
                })
            elif msg["role"] == "assistant":
                content = msg.get("content", "")
                # If this assistant message had tool calls, reconstruct them
                # as tool_use content blocks.
                if "tool_calls_raw" in msg:
                    blocks = []
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls_raw"]:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("tool_call_id", tc.get("name", "call_0")),
                            "name": tc["name"],
                            "input": tc["arguments"],
                        })
                    conv_messages.append({"role": "assistant", "content": blocks})
                else:
                    conv_messages.append({"role": "assistant", "content": content})
            else:
                conv_messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", ""),
                })

        # Ensure conversation doesn't start with assistant.
        if conv_messages and conv_messages[0]["role"] == "assistant":
            conv_messages.insert(0, {"role": "user", "content": "Continue."})

        # Ensure no consecutive same-role messages.
        conv_messages = self._merge_consecutive_roles(conv_messages)

        # Convert tool definitions from OpenAI to Anthropic format.
        anthropic_tools = _convert_tools_to_anthropic(tools) if tools else []

        kwargs = {
            "model": self.model,
            "messages": conv_messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
        }
        if system.strip():
            kwargs["system"] = system.strip()
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = self.client.messages.create(**kwargs)

        # Extract text and tool calls from response.
        text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input,
                    "tool_call_id": block.id,
                })

        # Anthropic returns input_tokens / output_tokens; normalise to the
        # prompt_tokens / completion_tokens names used by OpenAI.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if getattr(response, "usage", None) is not None:
            u = response.usage
            usage["prompt_tokens"] = getattr(u, "input_tokens", 0) or 0
            usage["completion_tokens"] = getattr(u, "output_tokens", 0) or 0
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return text, tool_calls, usage

    def _merge_consecutive_roles(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge consecutive messages with the same role.

        Anthropic requires alternating user/assistant messages.
        """
        if not messages:
            return messages

        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Merge content.
                prev_content = merged[-1].get("content", "")
                new_content = msg.get("content", "")
                if isinstance(prev_content, str) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + "\n" + new_content
                elif isinstance(prev_content, list) and isinstance(new_content, list):
                    merged[-1]["content"] = prev_content + new_content
                elif isinstance(prev_content, str) and isinstance(new_content, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev_content}] + new_content
                elif isinstance(prev_content, list) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + [{"type": "text", "text": new_content}]
            else:
                merged.append(msg)
        return merged


def _convert_tools_to_anthropic(
    openai_tools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI function-calling tool schema to Anthropic tool schema.

    OpenAI format:
        {"type": "function", "function": {"name": "...", "description": "...",
         "parameters": {"type": "object", "properties": {...}, "required": [...]}}}

    Anthropic format:
        {"name": "...", "description": "...",
         "input_schema": {"type": "object", "properties": {...}, "required": [...]}}
    """
    anthropic_tools = []
    for tool in openai_tools:
        func = tool.get("function", tool)
        anthropic_tools.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools
