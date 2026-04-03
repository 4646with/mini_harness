"""调用 LLM 决定下一步操作的 Agent 节点"""

import os
from langchain_core.messages import SystemMessage
from engine.patched_kimi import get_kimi_llm
from engine.state import ThreadState


def agent_node(state: ThreadState, config: dict = None) -> dict:
    """处理消息并决定下一步操作的 Agent 节点
    
    DeerFlow 设计原则：
    - Agent 节点在调用 LLM 前自己构建 prompt
    - 从 state.memory_context 读取记忆并注入
    
    参数:
        state: 当前线程状态
        config: LLM 配置
        
    返回:
        状态更新 (messages, tool_calls, is_complete)
    """
    # 使用 PatchedKimiChatOpenAI 避免 reasoning_content 丢失问题
    llm = get_kimi_llm(config)
    
    # 绑定唯一的元工具（Meta-Tool），实现 DeerFlow 的 Tool-as-an-Agent 架构
    tools = [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "委派具体任务给执行工兵。当用户要求执行任何需要落地系统的操作（如保存记忆、创建PPT等）时，你必须将所有需求详细写明，委派给此工具由工兵去调用底层业务 API执行。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "向工兵下达的具体任务描述，包含所有的要求和参数信息。例如：'请调用存储记忆工具，记下用户的幸运数字是7'。"
                        }
                    },
                    "required": ["task_description"]
                }
            }
        }
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    # 从 state 读取 memory_context 并注入到 prompt
    messages = state.get("messages", [])
    memory_context = state.get("memory_context", [])
    
    # 构建 prompt：如果有记忆则注入
    # 防呆警告：强制单次批量调用
    memory_rules = """你是一个拥有长期记忆的高级智能中心。

关于任务委派的使用规则：
1. 当用户提及个人信息、偏好、需要创建PPT或需要落地执行任何确切操作时，你必须使用 delegate_task 委派给工兵。
2. ⚠️ 极其重要：工兵（SubAgent）支持批量处理。如果有多条信息需要保存，请务必把它们打包在同一个 delegate_task 的 task_description 中，**只调用一次 delegate_task 工具**。绝不允许为了不同任务连续发起多次委派！
"""
    
    if memory_context:
        memory_text = "[系统记忆]\n" + "\n".join([
            f"- {m['content']}" for m in memory_context
        ])
        prompt = [SystemMessage(content=memory_rules + "\n" + memory_text)] + messages
    else:
        prompt = [SystemMessage(content=memory_rules)] + messages
    
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
