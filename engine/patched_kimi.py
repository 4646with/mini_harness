"""Kimi k2.5 专用 LangChain 补丁

解决 Kimi k2.5 (Thinking) 模型在多轮工具调用时，LangChain 静默丢弃
reasoning_content，导致 API 校验报错的问题。

参考 DeerFlow 的 patched_openai.py 架构设计。

使用方式：
    from engine.patched_kimi import PatchedKimiChatOpenAI
    
    llm = PatchedKimiChatOpenAI(
        model="kimi-k2.5",
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
        temperature=1.0,  # Kimi 官方要求 k2.5 的 temperature 必须设为 1.0
    )
"""

import os
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class PatchedKimiChatOpenAI(ChatOpenAI):
    """
    专门为 Kimi k2.5 (Thinking) 修复的 LangChain 补丁类。
    
    防止 LangChain 在多轮工具调用时静默丢弃 reasoning_content，
    导致 API 校验报错: "reasoning_content is missing"
    
    问题背景：
    - Kimi k2.5 是思考模型，输出包含 reasoning_content (思考过程)
    - LangChain 的 ChatOpenAI 在序列化消息时，会丢弃非标准字段
    - 下一轮对话时，服务器发现缺少必需字段，报错
    """
    
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        重写请求 payload 生成，在发送前强行塞回 reasoning_content。
        
        步骤：
        1. 拦截原始 LangChain 消息列表（包含完整的 additional_kwargs）
        2. 调用父类方法，获取标准 payload（此时 reasoning_content 已被阉割）
        3. 遍历 payload，把被丢弃的 reasoning_content 塞回去
        """
        # 1. 获取原始消息列表（包含完整的 additional_kwargs）
        original_messages = self._convert_input(input_).to_messages()
        
        # 2. 调用父类方法，获取标准 payload
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages", [])
        
        # 3. 遍历 Payload，把 Kimi 需要的思考过程塞回去
        if len(payload_messages) == len(original_messages):
            for payload_msg, orig_msg in zip(payload_messages, original_messages):
                # 只处理大模型发出的历史消息
                if payload_msg.get("role") == "assistant" and isinstance(orig_msg, AIMessage):
                    # 尝试从原消息的额外参数中捞回思考过程
                    reasoning = orig_msg.additional_kwargs.get("reasoning_content")
                    
                    if reasoning is not None:
                        # 塞回 reasoning_content
                        payload_msg["reasoning_content"] = reasoning
                    elif payload_msg.get("tool_calls"):
                        # 兜底机制：如果该回合发起了工具调用但没抓到思考过程，
                        # 强行给个空字符串，满足 Kimi API "必须保留该字段" 的严格校验
                        payload_msg["reasoning_content"] = ""
        
        return payload


def get_kimi_llm(model: str = None, temperature: float = None) -> PatchedKimiChatOpenAI:
    """
    工厂函数：获取配置好的 Kimi LLM 实例。
    
    参数:
        model: 模型名称，默认从环境变量 KIMI_MODEL 读取
        temperature: 温度参数，默认 1.0 (k2.5 必须)
        
    返回:
        PatchedKimiChatOpenAI 实例
    """
    model = model or os.getenv("KIMI_MODEL", "moonshot-v1-8k")
    
    # k2.5 必须使用 temperature=1.0
    if temperature is None:
        temperature = 1.0 if "k2.5" in model else 0.7
    
    return PatchedKimiChatOpenAI(
        model=model,
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        temperature=temperature,
    )


def get_summary_llm(config: dict = None) -> PatchedKimiChatOpenAI:
    """
    工厂函数：获取用于摘要生成的轻量级 LLM。
    
    优先使用配置中指定的模型，否则回退到默认轻量模型。
    
    参数:
        config: 摘要配置，包含 model 字段
        
    返回:
        用于摘要的 LLM 实例
    """
    config = config or {}
    model = config.get("model", os.getenv("SUMMARY_MODEL", "moonshot-v1-8k"))
    
    # 摘要模型使用较低温度以保持一致性
    temperature = config.get("temperature", 0.3)
    
    return PatchedKimiChatOpenAI(
        model=model,
        api_key=os.getenv("KIMI_API_KEY") or os.getenv("SUMMARY_API_KEY"),
        base_url=os.getenv("SUMMARY_BASE_URL") or os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        temperature=temperature,
        max_tokens=config.get("max_tokens", 200),
    )
