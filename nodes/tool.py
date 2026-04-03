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


def _execute_delegate_task(args: dict) -> str:
    """工兵内循环：接受委派任务，自动拉起小模型并在节点内闭包执行实际工具"""
    task_desc = args.get("task_description", "")
    if not task_desc:
        return "委派任务失败: 未提供 task_description。"

    from engine.patched_kimi import get_kimi_llm
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    
    # 强制工兵不使用 thinking，采用快速便宜的模型
    sub_config = {"model": "moonshot-v1-8k"}
    llm = get_kimi_llm(sub_config)
    
    # 将原先主帅的粗活工具全盘下放到工兵手中
    business_tools = [
        {
            "type": "function",
            "function": {
                "name": "create_ppt",
                "description": "创建 PowerPoint 演示文稿。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "主题"},
                        "slides_count": {"type": "integer", "description": "数量"}
                    },
                    "required": ["topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "将重要信息保存到长期记忆。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "description": "需要保存的记忆列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string", "description": "记忆内容"},
                                    "memory_type": {
                                        "type": "string",
                                        "enum": ["profile", "preference", "fact"],
                                        "description": "记忆分类"
                                    }
                                },
                                "required": ["content", "memory_type"]
                            }
                        }
                    },
                    "required": ["memories"]
                }
            }
        }
    ]
    
    llm_with_tools = llm.bind_tools(business_tools)
    
    print(f"🤖 [工兵部署] 正在为任务启动专属 8k 工兵: {task_desc[:50]}...")
    
    messages = [
        SystemMessage(content="你是一个专业的执行工兵。你的唯一目标是使用工具准确落实主帅下达的具体指令。请直接调用工具，并在结束时输出一行执行结果摘要。"),
        HumanMessage(content=task_desc)
    ]
    
    # 执行内循环
    for i in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # 检查是否调用了工具
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                t_id = tc.get("id", "")
                
                print(f"🛠️ [工兵行动] 正在执行具体工具: {t_name}")
                try:
                    t_result = _execute_by_name(t_name, t_args)
                except Exception as e:
                    t_result = f"执行报错: {str(e)}"
                    
                messages.append(ToolMessage(tool_call_id=t_id, name=t_name, content=str(t_result)))
        else:
            final_report = f"【工兵报告】: 任务已顺利完成。具体反馈：{response.content}"
            print(f"📡 [工兵回传] {final_report}")
            return final_report
            
    return "【工兵报告】: 警告！循环次数超限，任务可能未完全达成。"


register_tool("save_memory", _execute_save_memory)
register_tool("delegate_task", _execute_delegate_task)
