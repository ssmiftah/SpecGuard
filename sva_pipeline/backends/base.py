"""
Base protocol for LLM backends.

All backends (local, OpenAI, Anthropic) implement the same ``generate()``
method so the agent loop is backend-agnostic.
"""

from typing import Any, Dict, List, Protocol, Tuple


class LLMBackend(Protocol):
    """
    Protocol that all LLM backends must implement.

    The agent calls ``generate()`` once per ReAct step.  The backend
    handles message formatting, API calls or local inference, and
    response parsing — returning a uniform (text, tool_calls) tuple.
    """

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Generate one model response.

        Parameters
        ----------
        messages : list of dict
            Conversation history in OpenAI message format:
            ``[{"role": "system/user/assistant/tool", "content": "..."}]``
        tools : list of dict
            Tool definitions in OpenAI function-calling format.

        Returns
        -------
        response_text : str
            The model's text response (may be empty if only tool calls).
        tool_calls : list of dict
            Each dict has ``"name"`` (str) and ``"arguments"`` (dict).
            Empty list if the model didn't call any tools.
        usage : dict
            Token counts for this call:
            ``{"prompt_tokens": int, "completion_tokens": int,
              "total_tokens": int}``. All zero if the backend cannot
            report usage.
        """
        ...
