"""Ollama native chat/tool-calling client; no synthetic generation fallback."""
from typing import Protocol

import httpx


class ModelUnavailable(RuntimeError):
    pass


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             format_schema: dict | None = None) -> dict: ...


class OllamaModel:
    def __init__(self, url: str, model: str, timeout: float = 120) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             format_schema: dict | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": 0, "seed": 42, "num_predict": 768}}
        if tools:
            payload["tools"] = tools
        if format_schema:
            payload["format"] = format_schema
        try:
            response = httpx.post(self.url + "/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            message = response.json()["message"]
            if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
                raise ValueError("Malformed model response")
            return message
        except (httpx.HTTPError, KeyError, ValueError) as error:
            # Do not copy server responses, prompts, or environment values into errors/traces.
            raise ModelUnavailable("Ollama unavailable or returned invalid data; check server/model") \
                from error
