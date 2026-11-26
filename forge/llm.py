"""Reusable local LLM client using Ollama's /api/chat endpoint."""

import json
from typing import Any, Dict, List, Optional
import requests


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails or times out."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an HTTP error, malformed response, or invalid JSON."""
    pass


class OllamaClient:
    """Reusable client for interacting with a local Ollama instance via /api/chat."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        temperature: float = 0.1,
        num_ctx: int = 4096,
        timeout: float = 300.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.chat_endpoint = f"{self.base_url}/api/chat"

    def _build_options(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_ctx": self.num_ctx,
        }
        token_limit = max_tokens if max_tokens is not None else self.max_tokens
        if token_limit is not None:
            options["num_predict"] = token_limit
        return options

    def _build_messages(self, prompt: str, system_prompt: str = "") -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        format: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(temperature=temperature, max_tokens=max_tokens),
        }
        if format:
            payload["format"] = format

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out after {self.timeout}s: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {self.chat_endpoint}. "
                "Ensure the Ollama service is running."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"HTTP request error: {exc}") from exc

        if response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                f"Malformed JSON response from Ollama API: {response.text}"
            ) from exc

        if not isinstance(data, dict):
            raise OllamaResponseError(f"Unexpected response format from Ollama: {data}")

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise OllamaResponseError(
                f"Missing 'message.content' in Ollama response: {data}"
            )

        content = message["content"]

        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError(
                "Ollama returned empty or invalid message content."
            )

        return content

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a plain text response from the model."""
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)
        return self._call_api(
            messages=messages,
            format=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate and parse a structured JSON response from the model.

        Uses Ollama's format='json' mode and allows at most one bounded retry
        if the response cannot be parsed as valid JSON.
        """
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)

        raw_content = self._call_api(
            messages=messages,
            format="json",
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            parsed = json.loads(raw_content)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            return parsed
        except (json.JSONDecodeError, ValueError) as first_err:
            # Bounded retry: exactly one retry attempt
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": raw_content})
            retry_messages.append({
                "role": "user",
                "content": "Your previous response was not a valid JSON object. Return ONLY a valid JSON object.",
            })

            retry_content = self._call_api(
                messages=retry_messages,
                format="json",
                max_tokens=max_tokens,
                temperature=temperature,
            )

            try:
                parsed_retry = json.loads(retry_content)
                if not isinstance(parsed_retry, dict):
                    raise ValueError(f"Expected JSON object, got {type(parsed_retry).__name__}")
                return parsed_retry
            except (json.JSONDecodeError, ValueError) as retry_err:
                raise OllamaResponseError(
                    f"Failed to parse model response as JSON after 1 retry: {retry_err}. "
                    f"Initial response: {raw_content!r}. Retry response: {retry_content!r}"
                ) from retry_err
