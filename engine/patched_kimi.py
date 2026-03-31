"""Kimi k2.5 专用 LangChain 补丁"""

import os
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class PatchedKimiChatOpenAI(ChatOpenAI):
    """专门为 Kimi k2.5 (Thinking) 修复的 LangChain 补丁类。"""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        original_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages", [])

        orig_ai_msgs = [m for m in original_messages if isinstance(m, AIMessage)]

        print(f"[KIMI PATCH] 原始消息数: {len(original_messages)}, AI消息数: {len(orig_ai_msgs)}, Payload消息数: {len(payload_messages)}")

        for p_msg in payload_messages:
            if p_msg.get("role") == "assistant":
                orig_msg = orig_ai_msgs.pop(0) if orig_ai_msgs else None

                if orig_msg:
                    reasoning = orig_msg.additional_kwargs.get("reasoning_content")
                    if reasoning is not None:
                        p_msg["reasoning_content"] = reasoning
                        print(f"[KIMI PATCH] 塞入 reasoning_content (len={len(reasoning)})")
                    elif p_msg.get("tool_calls"):
                        p_msg["reasoning_content"] = ""
                        print(f"[KIMI PATCH] 塞入空 reasoning_content (reasoning=None, 有tool_calls)")

                if p_msg.get("tool_calls") and "reasoning_content" not in p_msg:
                    p_msg["reasoning_content"] = ""
                    print(f"[KIMI PATCH] 兜底：强制塞入空 reasoning_content")

        # 调试：打印所有带 tool_calls 的消息
        tool_call_msgs = [(i, msg) for i, msg in enumerate(payload_messages) if msg.get("tool_calls")]
        print(f"[KIMI PATCH] Payload中有 {len(tool_call_msgs)} 条带tool_calls的消息:")
        for i, msg in tool_call_msgs:
            rc = msg.get("reasoning_content")
            print(f"  消息[{i}] 有tool_calls, reasoning_content={repr(rc)[:80]}")

        # 如果有剩余的AI消息没被匹配到，打印警告
        if orig_ai_msgs:
            print(f"[KIMI PATCH] ⚠️ 有 {len(orig_ai_msgs)} 个AI消息没被匹配到!")

        return payload


def get_kimi_llm(model: str = None, temperature: float = None) -> PatchedKimiChatOpenAI:
    model = model or os.getenv("KIMI_MODEL", "moonshot-v1-8k")
    if temperature is None:
        temperature = 1.0 if "k2.5" in model else 0.7
    return PatchedKimiChatOpenAI(
        model=model,
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        temperature=temperature,
    )


def get_summary_llm(config: dict = None) -> PatchedKimiChatOpenAI:
    config = config or {}
    model = config.get("model", os.getenv("SUMMARY_MODEL", "moonshot-v1-8k"))
    temperature = config.get("temperature", 0.3)
    return PatchedKimiChatOpenAI(
        model=model,
        api_key=os.getenv("KIMI_API_KEY") or os.getenv("SUMMARY_API_KEY"),
        base_url=os.getenv("SUMMARY_BASE_URL") or os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        temperature=temperature,
        max_tokens=config.get("max_tokens", 200),
    )