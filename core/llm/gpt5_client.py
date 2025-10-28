"""
GPT-5 LLM Client

Implementation for OpenAI GPT-5 API with extended features:
- Reasoning effort control
- Cache control support
- Enhanced error handling and retry
- Token estimation
- OpenRouter provider configuration
"""

import os
import asyncio
import re
import tiktoken
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMClient, LLMResponse

# Load environment variables
load_dotenv()


class ContextLimitError(Exception):
    """Exception raised when context limit is exceeded"""
    pass


class GPT5Client(BaseLLMClient):
    """
    GPT-5 API client with advanced features

    Supports:
    - OpenAI GPT-5 models
    - Reasoning effort control (low/medium/high)
    - Cache control for efficiency
    - Token estimation
    - OpenRouter provider configuration
    """

    def __init__(
        self,
        model: str = "openai/gpt-5",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        reasoning_effort: str = "medium",
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        disable_cache_control: bool = False,
        openrouter_provider: Optional[str] = None,
        async_mode: bool = True,
        **kwargs
    ):
        """
        Initialize GPT-5 client

        Args:
            model: Model name (e.g., "openai/gpt-5", "gpt-5-2025-08-07")
            temperature: Sampling temperature
            api_key: API key (from env if not provided)
            base_url: API base URL (from env if not provided)
            max_tokens: Maximum completion tokens
            reasoning_effort: Reasoning effort level (low/medium/high)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty
            disable_cache_control: Disable cache control
            openrouter_provider: Specific OpenRouter provider
            async_mode: Use async client
            **kwargs: Additional parameters
        """
        super().__init__(model, temperature, **kwargs)

        # API configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_API_BASE")

        # Model parameters
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty

        # Features
        self.disable_cache_control = disable_cache_control
        self.openrouter_provider = openrouter_provider
        self.async_mode = async_mode

        # Initialize client
        if async_mode:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=1800
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=1800
            )

        # Initialize tokenizer for token estimation
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize tiktoken encoder for token estimation"""
        try:
            self.encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            # Fallback to cl100k_base if o200k_base is not available
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count of text"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        # Fallback: ~1 token per 4 characters
        return len(text) // 4

    def _apply_cache_control(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply cache control to messages for efficiency

        NOTE: OpenAI API (including GPT-5 via OpenRouter) does not support
        Anthropic-style cache_control in message content. This method is
        disabled by default for OpenAI-compatible APIs.

        For OpenAI, caching is handled automatically by the API provider.
        """
        # OpenAI API doesn't support Anthropic-style cache control format
        # Always return original messages without modification
        return messages

    def _clean_user_content_from_response(self, text: str) -> str:
        """Remove unwanted content patterns from assistant response"""
        # Remove content between \n\nUser: and <use_mcp_tool>
        pattern = r"\n\nUser:.*?(?=<use_mcp_tool>|$)"
        cleaned_text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
        return cleaned_text

    def _build_extra_body(self) -> Dict[str, Any]:
        """Build extra_body parameters for OpenRouter"""
        extra_body = {}

        # Provider configuration
        provider_config = (self.openrouter_provider or "").strip().lower()
        if provider_config == "google":
            extra_body["provider"] = {
                "only": ["google-vertex/us", "google-vertex/europe", "google-vertex/global"]
            }
        elif provider_config == "anthropic":
            extra_body["provider"] = {"only": ["anthropic"]}
        elif provider_config == "amazon":
            extra_body["provider"] = {"only": ["amazon-bedrock"]}
        elif provider_config:
            extra_body["provider"] = {"only": [provider_config]}

        # Sampling parameters
        if self.top_k != -1:
            extra_body["top_k"] = self.top_k
        if self.min_p != 0.0:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty

        return extra_body

    @retry(
        wait=wait_exponential(multiplier=5, min=5, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_not_exception_type(ContextLimitError),
    )
    async def _create_completion(
        self,
        params: Dict[str, Any],
        is_async: bool = True
    ):
        """
        Create completion with retry logic

        Args:
            params: Completion parameters
            is_async: Whether to use async client

        Returns:
            Completion response
        """
        import time
        retry_count = 0

        while True:
            try:
                if is_async:
                    return await self.client.chat.completions.create(**params)
                else:
                    return self.client.chat.completions.create(**params)
            except Exception as e:
                error_str = str(e)

                # Log full error details for debugging 400 errors
                import logging
                logger = logging.getLogger(__name__)
                if "400" in error_str or "Provider returned error" in error_str:
                    logger.error(f"Full error details: {e}")
                    logger.error(f"Error type: {type(e)}")
                    # Log request details that might help debug
                    logger.error(f"Model: {params.get('model')}")
                    logger.error(f"Messages count: {len(params.get('messages', []))}")
                    logger.error(f"Tools count: {len(params.get('tools', [])) if params.get('tools') else 0}")

                # Check for context limit errors
                context_error_patterns = [
                    "Input is too long for requested model",
                    "input length and `max_tokens` exceed context limit",
                    "maximum context length",
                    "prompt is too long",
                    "exceeds the maximum length",
                    "exceeds the maximum allowed length",
                    "Input tokens exceed the configured limit"
                ]

                if any(pattern in error_str for pattern in context_error_patterns):
                    raise ContextLimitError(f"Context limit exceeded: {error_str}")

                # Retry with exponential backoff
                retry_count += 1
                wait_time = min(2 * (2 ** min(retry_count - 1, 2)), 20)

                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"API call failed (attempt {retry_count}), retrying in {wait_time}s: {error_str}"
                )

                if is_async:
                    await asyncio.sleep(wait_time)
                else:
                    time.sleep(wait_time)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat request to GPT-5

        Args:
            messages: Chat messages
            tools: Tool definitions (OpenAI function calling format)
            **kwargs: Additional parameters

        Returns:
            LLMResponse
        """
        # Apply cache control if enabled
        processed_messages = self._apply_cache_control(messages)

        # Build parameters
        params = {
            "model": self.model,
            "messages": processed_messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": False,
            **self.extra_params,
            **kwargs
        }

        # Add optional parameters
        if self.top_p != 1.0:
            params["top_p"] = self.top_p

        # Add extra_body for OpenRouter
        extra_body = self._build_extra_body()
        if extra_body:
            params["extra_body"] = extra_body

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        # Note: reasoning_effort is removed as it's not supported by OpenRouter
        # OpenRouter automatically handles reasoning based on the model capabilities

        try:
            # Debug: Log API request parameters
            import logging
            logger = logging.getLogger(__name__)

            # Create a safe copy for logging (without large content)
            debug_params = {
                "model": params.get("model"),
                "temperature": params.get("temperature"),
                "max_completion_tokens": params.get("max_completion_tokens"),
                "stream": params.get("stream"),
                "top_p": params.get("top_p"),
                "message_count": len(params.get("messages", [])),
                "tool_count": len(params.get("tools", [])) if params.get("tools") else 0,
                "has_extra_body": bool(params.get("extra_body")),
                "has_tool_choice": "tool_choice" in params
            }
            logger.debug(f"GPT-5 API request params: {debug_params}")

            # If there are tools, log their structure
            if params.get("tools"):
                logger.debug(f"Tool definitions: {[t.get('function', {}).get('name') for t in params['tools']]}")

            # 🔥 DEBUG: Full logging for 400 errors (可通过环境变量控制)
            import json
            import os
            debug_enabled = os.getenv("DEBUG_API_REQUESTS", "false").lower() == "true"

            if debug_enabled and len(params.get('messages', [])) >= 8:  # Only debug problematic requests
                print("\n" + "="*80)
                print(f"🔍 FULL API REQUEST DEBUG (Messages: {len(params.get('messages', []))})")
                print("="*80)
                for i, msg in enumerate(params.get('messages', [])):
                    print(f"\n--- Message {i+1} ---")
                    msg_copy = msg.copy()
                    if 'content' in msg_copy and isinstance(msg_copy['content'], str) and len(msg_copy['content']) > 300:
                        msg_copy['content'] = msg_copy['content'][:300] + f"... (truncated, {len(msg_copy['content'])} chars total)"
                    print(json.dumps(msg_copy, indent=2, ensure_ascii=False))
                print("\n" + "="*80 + "\n")

            # Create completion
            response = await self._create_completion(params, is_async=self.async_mode)

            if not response or not response.choices or len(response.choices) == 0:
                raise Exception(f"LLM returned empty response")

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Check finish reason
            # Note: Only raise ContextLimitError if response is actually empty or truncated
            if finish_reason == "length":
                # Check if we got any usable content
                if not message.content and not (hasattr(message, 'tool_calls') and message.tool_calls):
                    raise ContextLimitError("Response truncated due to maximum context length")
                # Otherwise, log warning but continue
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Response has finish_reason='length' but content exists, continuing...")

            if finish_reason == "stop" and not message.content and not (hasattr(message, 'tool_calls') and message.tool_calls):
                raise Exception("LLM returned empty content with finish_reason='stop'")

            # Extract tool calls if present
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })

            # Clean response content
            content = message.content or ""
            if content:
                content = self._clean_user_content_from_response(content)

            # Prepare usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            # Update cumulative usage stats
            self._update_usage_stats(usage)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                model=response.model,
                finish_reason=finish_reason
            )

        except asyncio.CancelledError:
            # Gracefully handle cancellation (e.g., from Ctrl+C)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("LLM API call was cancelled (likely due to user interrupt)")
            raise
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("LLM API call interrupted by user")
            raise
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"GPT-5 API error: {e}")
            logger.error(f"Model: {self.model}")
            logger.error(f"Message count: {len(messages)}")
            if tools:
                logger.error(f"Tool count: {len(tools)}")
            raise

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Stream chat response from GPT-5

        Args:
            messages: Chat messages
            tools: Tool definitions
            **kwargs: Additional parameters

        Yields:
            LLMResponse chunks
        """
        # Apply cache control if enabled
        processed_messages = self._apply_cache_control(messages)

        params = {
            "model": self.model,
            "messages": processed_messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": True,
            **self.extra_params,
            **kwargs
        }

        if self.top_p != 1.0:
            params["top_p"] = self.top_p

        extra_body = self._build_extra_body()
        if extra_body:
            params["extra_body"] = extra_body

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        # Note: reasoning_effort is removed as it's not supported by OpenRouter

        stream = await self.client.chat.completions.create(**params)

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                yield LLMResponse(
                    content=delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason or ""
                )
