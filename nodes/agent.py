"""调用 LLM 决定下一步操作的 Agent 节点"""

import os
from langchain_core.messages import SystemMessage
from engine.patched_kimi import get_kimi_llm
from engine.state import ThreadState


def agent_node(state: ThreadState) -> dict:
    """处理消息并决定下一步操作的 Agent 节点
    
    DeerFlow 设计原则：
    - Agent 节点在调用 LLM 前自己构建 prompt
    - 从 state.memory_context 读取记忆并注入
    
    参数:
        state: 当前线程状态
        
    返回:
        状态更新 (messages, tool_calls, is_complete)
    """
    # 使用 PatchedKimiChatOpenAI 避免 reasoning_content 丢失问题
    llm = get_kimi_llm()
    
    # 绑定工具以进行函数调用
    # DeerFlow 风格：Agent 自己决定保存什么记忆
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_ppt",
                "description": "创建 PowerPoint 演示文稿。当用户要求创建演示文稿、幻灯片或 PPT 时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "演示文稿的主题"
                        },
                        "slides_count": {
                            "type": "integer",
                            "description": "要创建的幻灯片数量"
                        }
                    },
                    "required": ["topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "将重要信息保存到长期记忆。当用户提到他们的名字、偏好或任何希望你在未来对话中记住的事实时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "要记住的信息（例如：'用户名字是 John'，'用户喜欢中文'）"
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["profile", "preference", "fact", "general"],
                            "description": "记忆类型：profile（用户信息）、preference（用户喜好）、fact（一般事实）、general（其他）"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "置信度 0-1，越高表示越确定"
                        }
                    },
                    "required": ["content"]
                }
            }
        }
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    # 从 state 读取 memory_context 并注入到 prompt
    messages = state.get("messages", [])
    memory_context = state.get("memory_context", [])
    
    # 构建 prompt：如果有记忆则注入
    if memory_context:
        memory_text = "[系统记忆]\n" + "\n".join([
            f"- {m['content']}" for m in memory_context
        ])
        prompt = [SystemMessage(content=memory_text)] + messages
    else:
        prompt = messages
    
    # 调用 LLM
    response = llm_with_tools.invoke(prompt)
    
    # 检查是否有工具调用
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = response.tool_calls
    
    # 判断对话是否完成
    # 完成条件：没有工具调用 且 有内容
    is_complete = len(tool_calls) == 0 and response.content is not None
    
    return {
        "messages": [response],
        "tool_calls": tool_calls,
        "is_complete": is_complete
    }
