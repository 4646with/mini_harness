"""Stream Bridge - 事件标准化模块

将 LangGraph 复杂的内部状态翻译为标准化的 CLI/UI 事件。

参考 DeerFlow 的 StreamBridge 设计：
- text_reply: 文本回复事件
- tool_call: 工具调用事件
- billing: 计费事件
"""

from typing import Generator, Dict, Any


def extract_standard_events(graph_stream) -> Generator[Dict[str, Any], None, None]:
    """
    将 LangGraph 节点状态翻译为标准化事件。
    
    参数:
        graph_stream: LangGraph stream 输出
        
    生成:
        标准化的事件字典
    """
    for event in graph_stream:
        # 只处理 agent 节点输出
        if "agent" not in event:
            continue
        
        node_state = event.get("agent", {})
        messages = node_state.get("messages", [])
        
        if not messages:
            continue
        
        last_msg = messages[-1]
        
        # 类型检查
        msg_type = getattr(last_msg, "type", None)
        if msg_type != "ai":
            continue
        
        # 1. 工具调用事件
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                yield {
                    "event_type": "tool_call",
                    "tool_name": tc.get("name", "unknown"),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", "")
                }
            # 工具调用消息不输出文本
            continue
        
        # 2. 文本回复事件
        content = getattr(last_msg, "content", "")
        if content:
            yield {
                "event_type": "text_reply",
                "content": content
            }
        
        # 3. 计费事件
        usage_metadata = getattr(last_msg, "usage_metadata", None)
        if usage_metadata:
            yield {
                "event_type": "billing",
                "usage": {
                    "input_tokens": usage_metadata.get("input_tokens", 0),
                    "output_tokens": usage_metadata.get("output_tokens", 0)
                }
            }
