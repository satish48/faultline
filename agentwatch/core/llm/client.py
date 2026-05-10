"""
AgentWatch — LLM Client Abstraction
===================================
Provider-agnostic LLM client that normalizes structured tool responses
across Anthropic and Groq/OpenAI-compatible backends.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, Optional

import structlog
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class LLMClient:
    """
    Unified LLM client that hides provider-specific response formats.

    All AgentWatch modules should call:
        client.chat_with_tool(...)

    They should not directly call Anthropic, OpenAI, or Groq SDKs.
    """

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
    ):
        if not api_key:
            raise ValueError(f"Missing API key for provider={provider.value}")

        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self._client = self._init_client(provider, api_key)

    @classmethod
    def from_env(cls) -> "LLMClient":
        """
        Prefer Groq for cheaper/faster development.
        Fall back to Anthropic if no Groq key is found.
        """
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

        if groq_key:
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            logger.info("llm_client_init", provider="groq", model=model)
            return cls(LLMProvider.GROQ, groq_key, model)

        if anthropic_key:
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            logger.info("llm_client_init", provider="anthropic", model=model)
            return cls(LLMProvider.ANTHROPIC, anthropic_key, model)

        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY in .env."
        )

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def chat_with_tool(
        self,
        system: str,
        user: str,
        tool: Dict[str, Any],
        tool_name: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        tokens = max_tokens or self.max_tokens

        if self.provider == LLMProvider.GROQ:
            return self._groq_chat_with_tool(system, user, tool, tool_name, tokens)

        return self._anthropic_chat_with_tool(system, user, tool, tool_name, tokens)

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        tokens = max_tokens or self.max_tokens

        if self.provider == LLMProvider.GROQ:
            return self._groq_chat(system, user, tokens)

        return self._anthropic_chat(system, user, tokens)

    def _groq_chat_with_tool(
        self,
        system: str,
        user: str,
        tool: Dict[str, Any],
        tool_name: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[openai_tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            logger.warning("groq_no_tool_call_returned", tool=tool_name)
            return {}

        raw = tool_calls[0].function.arguments
        return self._safe_json_loads(raw, source="groq_tool")

    def _groq_chat(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def _anthropic_chat_with_tool(
        self,
        system: str,
        user: str,
        tool: Dict[str, Any],
        tool_name: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                return dict(block.input)

        logger.warning("anthropic_no_tool_use_block", tool=tool_name)
        return {}

    def _anthropic_chat(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        texts = [block.text for block in response.content if hasattr(block, "text")]
        return " ".join(texts)

    @staticmethod
    def _safe_json_loads(raw: Any, source: str) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw

        if not isinstance(raw, str):
            logger.error("json_parse_invalid_type", source=source, type=str(type(raw)))
            return {}

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            logger.error("json_parse_failed", source=source, raw=raw[:300])
            return {}

    @staticmethod
    def _init_client(provider: LLMProvider, api_key: str):
        if provider == LLMProvider.GROQ:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install openai package to use Groq: pip install openai") from exc

            return OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic package to use Claude: pip install anthropic") from exc

        return anthropic.Anthropic(api_key=api_key)