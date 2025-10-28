"""
Base LLM Client Interface

Abstract base class for all LLM clients.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """
    Response from LLM

    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls made by the LLM
        usage: Token usage statistics
        model: Model name used for generation
        finish_reason: Reason for completion (stop, length, etc.)
    """
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    })
    model: str = ""
    finish_reason: str = ""

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return len(self.tool_calls) > 0

    def get_total_tokens(self) -> int:
        """Get total token count"""
        return self.usage.get("total_tokens", 0)

    def is_truncated(self) -> bool:
        """Check if response was truncated due to length"""
        return self.finish_reason == "length"


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients

    All LLM providers (OpenAI, Anthropic, etc.) should implement this interface.
    """

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.extra_params = kwargs
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0

    def get_usage_stats(self) -> Dict[str, int]:
        """Get cumulative usage statistics"""
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_requests": self._total_requests
        }

    def _update_usage_stats(self, usage: Dict[str, int]):
        """Update usage statistics"""
        self._total_prompt_tokens += usage.get("prompt_tokens", 0)
        self._total_completion_tokens += usage.get("completion_tokens", 0)
        self._total_requests += 1

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat messages to LLM and get response

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Stream chat response from LLM

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            **kwargs: Additional parameters

        Yields:
            Chunks of LLMResponse
        """
        pass

    def format_system_message(self, content: str) -> Dict[str, str]:
        """Format system message"""
        return {"role": "system", "content": content}

    def format_user_message(self, content: str) -> Dict[str, str]:
        """Format user message"""
        return {"role": "user", "content": content}

    def format_assistant_message(self, content: str) -> Dict[str, str]:
        """Format assistant message"""
        return {"role": "assistant", "content": content}
