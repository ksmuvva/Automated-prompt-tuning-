"""
LLM interface for testing prompts with various LLM providers.
Supports OpenAI, Anthropic, and local models.
"""

import os
import time
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import json


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def query(self, prompt: str) -> Tuple[str, float]:
        """
        Query the LLM with a prompt.

        Returns:
            (response_text, response_time_seconds)
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def query(self, prompt: str) -> Tuple[str, float]:
        """Query OpenAI API."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analysis assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )

            response_time = time.time() - start_time
            response_text = response.choices[0].message.content

            return response_text, response_time

        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return json.dumps({"above_250": [], "anomalies": [], "error": str(e)}), time.time() - start_time


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def query(self, prompt: str) -> Tuple[str, float]:
        """Query Anthropic API."""
        start_time = time.time()

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_time = time.time() - start_time
            response_text = message.content[0].text

            return response_text, response_time

        except Exception as e:
            print(f"Error querying Anthropic: {e}")
            return json.dumps({"above_250": [], "anomalies": [], "error": str(e)}), time.time() - start_time


class OllamaProvider(LLMProvider):
    """Local Ollama provider for open-source models."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", temperature: float = 0.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")

    def query(self, prompt: str) -> Tuple[str, float]:
        """Query Ollama API."""
        start_time = time.time()

        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                },
                timeout=60
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                response_text = response.json().get("response", "")
                return response_text, response_time
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                return json.dumps({"above_250": [], "anomalies": [], "error": error_msg}), response_time

        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return json.dumps({"above_250": [], "anomalies": [], "error": str(e)}), time.time() - start_time


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API calls."""

    def __init__(self, accuracy: float = 0.8):
        """
        Initialize mock provider.

        Args:
            accuracy: Simulated accuracy (0-1)
        """
        self.accuracy = accuracy
        import random
        self.random = random

    def query(self, prompt: str) -> Tuple[str, float]:
        """Return mock response based on prompt."""
        import time
        start_time = time.time()

        # Simulate processing time
        time.sleep(self.random.uniform(0.1, 0.5))

        # Extract transaction data from prompt (very simple parsing)
        # In real scenario, we'd parse the CSV data properly
        response = {
            "above_250": [],
            "anomalies": []
        }

        # Simulate some detection (for testing purposes)
        # This is intentionally simple for demonstration
        if "TXN" in prompt:
            # Simulate finding some transactions
            response["above_250"] = ["TXN001", "TXN005"] if self.random.random() < self.accuracy else []
            response["anomalies"] = [
                {"transaction_id": "TXN003", "reason": "Mock anomaly detection"}
            ] if self.random.random() < self.accuracy else []

        response_time = time.time() - start_time
        return json.dumps(response), response_time


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(provider_type: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            provider_type: Type of provider ("openai", "anthropic", "ollama", "mock")
            **kwargs: Additional arguments for the provider

        Returns:
            LLMProvider instance
        """
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
            "mock": MockLLMProvider
        }

        if provider_type not in providers:
            raise ValueError(f"Unknown provider type: {provider_type}. "
                           f"Available: {list(providers.keys())}")

        return providers[provider_type](**kwargs)


class LLMTester:
    """Test prompts with LLM providers."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.cache = {}  # Cache results to avoid duplicate API calls

    def test_prompt(self, prompt: str, use_cache: bool = True) -> Tuple[str, float]:
        """
        Test a prompt with the LLM.

        Args:
            prompt: The prompt to test
            use_cache: Whether to use cached results

        Returns:
            (response, response_time)
        """
        # Check cache
        if use_cache and prompt in self.cache:
            return self.cache[prompt]

        # Query LLM
        response, response_time = self.provider.query(prompt)

        # Cache result
        if use_cache:
            self.cache[prompt] = (response, response_time)

        return response, response_time

    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()


def format_transaction_data(df) -> str:
    """
    Format transaction DataFrame for LLM input.

    Args:
        df: pandas DataFrame with transaction data

    Returns:
        Formatted string representation
    """
    # Use CSV format for clarity
    csv_str = df[['transaction_id', 'date', 'amount_gbp', 'category', 'merchant', 'description']].to_csv(index=False)
    return csv_str


if __name__ == "__main__":
    # Example usage
    print("Testing LLM providers...\n")

    # Test with mock provider
    print("1. Testing Mock Provider:")
    mock_provider = LLMFactory.create("mock", accuracy=0.9)
    tester = LLMTester(mock_provider)

    test_prompt = """Analyze these transactions:
TXN001, 2024-01-01, 300.00, Shopping, Store1
TXN002, 2024-01-02, 100.00, Groceries, Store2

Find transactions > 250 GBP and anomalies."""

    response, resp_time = tester.test_prompt(test_prompt)
    print(f"Response: {response}")
    print(f"Time: {resp_time:.2f}s\n")

    # Instructions for real providers
    print("2. To use OpenAI:")
    print("   provider = LLMFactory.create('openai', model='gpt-4', api_key='your-key')")
    print()
    print("3. To use Anthropic:")
    print("   provider = LLMFactory.create('anthropic', model='claude-3-5-sonnet-20241022', api_key='your-key')")
    print()
    print("4. To use Ollama (local):")
    print("   provider = LLMFactory.create('ollama', model='llama2')")
