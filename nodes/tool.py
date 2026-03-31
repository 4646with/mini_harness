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


def _execute_save_memory(args: dict, config: dict = None) -> str:
    """执行 save_memory 工具（支持批量）"""
    store = get_memory_store()
    memories = args.get("memories", [])
    
    if not memories and "content" in args:
        content = args.get("content")
        memory_type = args.get("memory_type", "general")
        return store.save_memory(content, memory_type, config)
    
    results = []
    for mem in memories:
        content = mem.get("content")
        if content:
            memory_type = mem.get("memory_type", "general")
            res = store.save_memory(content, memory_type, config)
            results.append(res)
    
    if not results:
        return "❌ 错误: 未提供有效的记忆内容。"
        
    return "\n".join(results)
