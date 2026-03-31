"""执行已批准工具调用的工具节点"""

from typing import Dict, Callable, Any
from langchain_core.messages import ToolMessage

from engine.state import ThreadState
from memory.store import get_memory_store
from skills.registry import get_skill_registry


_tool_handlers: Dict[str, Callable] = {}


def register_tool(name: str, handler: Callable):
    """注册工具处理器"""
    _tool_handlers[name] = handler


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
        return {
            "messages": [],
            "approved_tools": []
        }

    results = []
    for tool_call in approved_tools:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        result_content = _execute_by_name(tool_name, tool_args)

        results.append(ToolMessage(
            content=result_content,
            tool_call_id=tool_call_id
        ))

    return {
        "messages": results,
        "approved_tools": []
    }


def _execute_by_name(tool_name: str, tool_args: dict) -> str:
    """根据工具名称执行"""
    if tool_name in _tool_handlers:
        return _tool_handlers[tool_name](tool_args)

    skill_registry = get_skill_registry()
    if skill := skill_registry.get(tool_name):
        return _run_sync(skill.execute, **tool_args)

    return f"[错误] 未知工具: {tool_name}"


def _run_sync(async_func, **kwargs) -> str:
    """安全运行异步函数"""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_func(**kwargs))

    future = loop.create_task(async_func(**kwargs))
    return loop.run_until_complete(future)


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


def execute_tool_call(tool_call: dict, config: dict = None) -> str:
    """外部调用工具执行的接口（供 SubAgent 使用）"""
    tool_name = tool_call.get("name", "unknown")
    tool_args = tool_call.get("args", {})

    return _execute_by_name(tool_name, tool_args)


register_tool("save_memory", _execute_save_memory)
