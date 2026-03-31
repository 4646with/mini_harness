"""执行已批准工具调用的工具节点"""

from langchain_core.messages import ToolMessage

from engine.state import ThreadState
from memory.store import get_memory_store


def tool_node(state: ThreadState) -> dict:
    """执行已批准工具调用的工具节点
    
    在第一阶段，模拟工具执行以进行 HITL 测试。
    只执行人类批准的工具。
    
    参数:
        state: 当前线程状态
        
    返回:
        状态更新 (messages, approved_tools cleared)
    """
    approved_tools = state.get("approved_tools", [])
    
    if not approved_tools:
        # 没有已批准的工具，返回空结果
        return {
            "messages": [],
            "approved_tools": []
        }
    
    # 执行工具
    results = []
    for tool_call in approved_tools:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")
        
        # 根据工具名称执行
        if tool_name == "save_memory":
            result_content = _execute_save_memory(tool_args)
        elif tool_name == "create_ppt":
            result_content = f"[模拟] PPT 已创建，主题: {tool_args.get('topic', 'unknown')}"
        else:
            # 默认模拟
            result_content = f"[模拟] 工具 '{tool_name}' 已执行，参数: {tool_args}"
        
        # 使用 ToolMessage 以符合 LangGraph 协议
        results.append(ToolMessage(
            content=result_content,
            tool_call_id=tool_call_id
        ))
    
    return {
        "messages": results,
        "approved_tools": []  # 执行后清除已批准的工具
    }


def _execute_save_memory(args: dict) -> str:
    """执行 save_memory 工具
    
    DeerFlow 风格：Agent 主动决定保存什么记忆
    
    参数:
        args: 工具参数，包含 content, memory_type 等
        
    返回:
        结果消息
    """
    store = get_memory_store()
    
    content = args.get("content", "")
    memory_type = args.get("memory_type", "general")
    confidence = args.get("confidence", 0.9)
    
    if not content:
        return "[错误] 记忆内容不能为空"
    
    # 保存到记忆存储
    memory = store.add(
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        metadata={"source": "agent_tool_call"}
    )
    
    return f"[成功] 记忆已保存: {content[:50]}..."
