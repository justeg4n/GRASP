"""Frozen OpenAI-compatible reader used as the rollout environment."""
from __future__ import annotations


class OpenAIReader:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        return (response.choices[0].message.content or "").strip()
