"""
OpenRouter Mini Client - 简化版

使用 GPT-4.1-mini 进行测试的轻量级客户端
- 低成本
- 简单配置
- 快速响应
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

from .base import BaseLLMClient, LLMResponse

# 加载环境变量
load_dotenv()


class OpenRouterMiniClient(BaseLLMClient):
    """
    OpenRouter 简化客户端，使用 GPT-4.1-mini
    
    专为测试设计：
    - 低成本
    - 快速
    - 配置简单
    """

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        async_mode: bool = True,
        **kwargs
    ):
        """
        初始化 OpenRouter Mini 客户端

        Args:
            model: 模型名称，默认 "openai/gpt-4.1-mini"
            temperature: 采样温度
            api_key: API key（从 .env 读取）
            base_url: API base URL（从 .env 读取）
            max_tokens: 最大 token 数
            async_mode: 是否使用异步模式
            **kwargs: 其他参数
        """
        super().__init__(model, temperature, **kwargs)

        # 从环境变量读取配置
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_API_BASE")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL not found in environment variables")

        self.max_tokens = max_tokens
        self.async_mode = async_mode

        # 初始化客户端
        if async_mode:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=300  # 5分钟超时
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=300
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            tools: 工具定义（可选）
            **kwargs: 其他参数

        Returns:
            LLMResponse 对象
        """
        # 构建请求参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_params,
            **kwargs
        }

        # 添加工具（如果有）
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            # 调用 API
            if self.async_mode:
                response = await self.client.chat.completions.create(**params)
            else:
                response = self.client.chat.completions.create(**params)

            # 检查响应
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("LLM 返回空响应")

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # 提取工具调用（如果有）
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

            # 提取内容
            content = message.content or ""

            # 统计 token 使用
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            # 更新累计统计
            self._update_usage_stats(usage)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                model=response.model,
                finish_reason=finish_reason
            )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"OpenRouter Mini API 错误: {e}")
            logger.error(f"模型: {self.model}")
            logger.error(f"消息数量: {len(messages)}")
            raise

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        流式聊天响应

        Args:
            messages: 消息列表
            tools: 工具定义（可选）
            **kwargs: 其他参数

        Yields:
            LLMResponse 块
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **self.extra_params,
            **kwargs
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**params)

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                yield LLMResponse(
                    content=delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason or ""
                )

    def __repr__(self):
        return f"OpenRouterMiniClient(model={self.model}, temperature={self.temperature})"

