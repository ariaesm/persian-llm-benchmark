from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMClient(ABC):
    """Abstract interface for benchmark model evaluation targets."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Generates a text completion for a given benchmark prompt."""
        pass


class MockLLMClient(BaseLLMClient):
    """
    Deterministic offline client for benchmarking harness validation and CI environments.
    """
    def __init__(self, accuracy_bias: float = 0.85):
        self.accuracy_bias = accuracy_bias
        self.call_history = []

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.call_history.append(prompt)
        # Deterministic generation for known benchmark patterns
        if "پایتخت ایران" in prompt:
            return "تهران"
        if "۲ + ۲" in prompt or "2 + 2" in prompt:
            return "۴"
        if "تعداد روزهای سال کبیسه" in prompt:
            return "۳۶۶ روز"
        return "پاسخ مدل آزمایشی بر اساس پرامپت ارائه‌شده."


class OpenAILLMClient(BaseLLMClient):
    """Adapter for OpenAI Chat Completions API."""
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        import httpx
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class OllamaLLMClient(BaseLLMClient):
    """Adapter for local Ollama endpoints (e.g. LLaMA-3.1, Qwen-2.5)."""
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        import httpx
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=120.0)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        return response.json()["response"].strip()
