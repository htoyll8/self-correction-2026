"""Provider client: one `complete(prompt, temperature) -> str` routed to either the OpenAI
Responses API or Anthropic (direct, or via Vertex when ANTHROPIC_VERTEX_PROJECT is set).

Prompt wording lives in the strategies, not here — this class only turns a prompt into text.
"""
import os
import threading

import anthropic
from openai import OpenAI


class Model:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float | None = 0):
        self.model_name = model_name
        self.temperature = temperature
        self._lock = threading.Lock()      # complete() is called from worker threads
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if self._is_claude():
            vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT")
            if vertex_project:
                from anthropic import AnthropicVertex
                region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
                self.client = AnthropicVertex(project_id=vertex_project, region=region)
            else:
                self.client = anthropic.Anthropic()
        else:
            self.client = OpenAI()

    def _is_claude(self) -> bool:
        return self.model_name.lower().startswith("claude")

    def _add_usage(self, usage) -> None:
        """Accumulate token counts from a response's usage object (OpenAI Responses and
        Anthropic both expose input_tokens/output_tokens)."""
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "output_tokens", 0) or 0

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """Return a single completion for `prompt`. GPT-5 ignores temperature (unsupported)."""
        temperature = temperature if temperature is not None else self.temperature
        if self._is_claude():
            kwargs = {
                "model": self.model_name,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            msg = self.client.messages.create(**kwargs)
            self._add_usage(msg.usage)
            return msg.content[0].text
        request = {"model": self.model_name, "input": prompt}
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature
        resp = self.client.responses.create(**request)
        self._add_usage(resp.usage)
        return resp.output_text
