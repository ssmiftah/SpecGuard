"""
Local HuggingFace model backend.

Loads a model (e.g. Qwen3-8B) locally via ``transformers`` and runs
inference on the GPU.  This is the original pipeline backend, extracted
from ``agent.py`` into a standalone class.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Regex to extract <tool_call>…</tool_call> blocks from model output.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class LocalBackend:
    """
    HuggingFace local model backend.

    Loads the model to GPU and runs inference via ``model.generate()``.
    Tool calls are extracted from Qwen3's ``<tool_call>`` XML markers.
    """

    def __init__(self, config: Any):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config

        logger.info("Loading tokenizer for %s …", config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
        )

        # Free any cached embedding models from GPU before loading the LLM.
        try:
            from ..rag import _ENCODER_CACHE
            for enc in _ENCODER_CACHE.values():
                enc.cpu()
        except (ImportError, AttributeError):
            pass

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build load kwargs based on quantization setting.
        load_kwargs = dict(
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )

        quant = getattr(config, "quantization", "none")

        if quant == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            # device_map must be "auto" for quantized models.
            load_kwargs["device_map"] = "auto"
            logger.info("Loading model %s (8-bit quantized) …", config.model_id)

        elif quant == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            logger.info("Loading model %s (4-bit quantized) …", config.model_id)

        else:
            # Full precision — use configured dtype.
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            load_kwargs["dtype"] = dtype_map.get(config.dtype, torch.bfloat16)
            logger.info("Loading model %s (%s) …", config.model_id, config.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, **load_kwargs,
        )
        self.model.eval()
        self._grammar: str = ""
        self._logits_processor = None
        logger.info("Model loaded.")

    def set_grammar(self, gbnf: str) -> None:
        """
        Set a GBNF grammar for constrained generation via outlines.

        Requires ``pip install outlines``. When set, a LogitsProcessor is
        created that constrains token generation to match the grammar.
        """
        self._grammar = gbnf
        if not gbnf:
            self._logits_processor = None
            return

        try:
            from outlines.processors import CFGLogitsProcessor
            self._logits_processor = CFGLogitsProcessor(
                grammar=gbnf,
                tokenizer=self.tokenizer,
            )
            logger.info(
                "Grammar constraints enabled via outlines (%d chars).",
                len(gbnf),
            )
        except ImportError:
            logger.warning(
                "Grammar constraints requested but 'outlines' is not "
                "installed. Install with: pip install outlines. "
                "Falling back to unconstrained generation."
            )
            self._logits_processor = None
        except Exception as exc:
            logger.warning(
                "Failed to create grammar processor: %s. "
                "Falling back to unconstrained generation.",
                exc,
            )
            self._logits_processor = None

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Run one forward pass and return (response_text, tool_calls, usage).
        """
        import torch

        # Apply the chat template with tool definitions.
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.config.enable_thinking,
        )

        # Tokenise.
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_len = inputs.input_ids.shape[1]

        # Stage 2.5: grammar-constrained generation via outlines.
        # Only active when a grammar is set AND no tool calls are expected
        # (grammar and tool-calling are mutually exclusive).
        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=(self.config.temperature > 0),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self._logits_processor is not None and not tools:
            from transformers import LogitsProcessorList
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [self._logits_processor]
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens.
        response_text = self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=False,
        )

        # Strip trailing EOS tokens.
        response_text = response_text.replace(
            self.tokenizer.eos_token or "<|endoftext|>", ""
        ).strip()

        # Parse tool calls from <tool_call> blocks.
        tool_calls = self._parse_tool_calls(response_text)

        completion_len = int(outputs.shape[1] - prompt_len)
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": completion_len,
            "total_tokens": int(prompt_len) + completion_len,
        }

        return response_text, tool_calls, usage

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Qwen3's <tool_call> XML markers."""
        tool_calls = []
        for raw_json in _TOOL_CALL_RE.findall(text):
            try:
                parsed = json.loads(raw_json.strip())
                if "function" in parsed:
                    name = parsed["function"].get("name", "")
                    args = (parsed["function"].get("parameters", {})
                            or parsed["function"].get("arguments", {}))
                else:
                    name = parsed.get("name", "")
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                if name:
                    tool_calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse tool call: %s", exc)
        return tool_calls
