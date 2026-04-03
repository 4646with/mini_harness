"""FastAPI 网关入口"""
import os
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException

load_dotenv(Path(__file__).parent.parent / ".env")
from api.models import (
    ChatRequest, ChatResponse, SkillInfo,
    ToolApprovalRequest, ToolActionRequest, PendingToolCallsResponse, ToolCall
)
from api.deps import get_graph, get_config
from skills.registry import get_skill_registry
from router import TaskType, get_classifier, LeadAgent, SubAgent, Aggregator
from langchain_core.messages import ToolMessage


class PendingToolsStore:
    """线程安全的待批准工具存储（保留用于兼容性，后续可删除）"""

    def __init__(self):
        self._store: dict[str, list] = {}

    def get(self, thread_id: str) -> list:
        return self._store.get(thread_id, [])

    def set(self, thread_id: str, tools: list):
        self._store[thread_id] = tools

    def clear(self, thread_id: str):
        self._store.pop(thread_id, None)

    def pop(self, thread_id: str) -> list:
        return self._store.pop(thread_id, [])


_pending_tools_store = PendingToolsStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan 上下文管理器 - 应用启动/关闭"""
    yield


app = FastAPI(title="Mini Agent Harness API", version="3.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mini-harness-v3"}


@app.get("/skills", response_model=List[SkillInfo])
def list_skills():
    registry = get_skill_registry()
    return registry.list_skills()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, graph=Depends(get_graph)):
    classifier = get_classifier()
    task_type = classifier.classify(request.message)

    if task_type == TaskType.DIRECT:
        return await _direct_execute(request, graph)
    else:
        return _multi_agent_execute(request)


async def _direct_execute(request, graph):
    """无状态网关：直接执行，不存储任何状态"""
    import uuid
    import asyncio
    from langchain_core.messages import HumanMessage

    thread = {"configurable": {"thread_id": request.thread_id or str(uuid.uuid4())}}
    input_message = {"role": "user", "content": request.message}

    def run_stream():
        result = {"response": "", "tool_calls": [], "has_tool_calls": False}
        for event in graph.stream({"messages": [input_message]}, thread):
            if "agent" in event:
                messages = event["agent"].get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        result["response"] = last_msg.content
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        result["tool_calls"] = last_msg.tool_calls
                        result["has_tool_calls"] = True
        return result

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_stream)

    # 如果有 tool_calls，返回 pending_tools 状态，不返回 response
    if result["has_tool_calls"]:
        return ChatResponse(
            status="pending_tools",
            response="",
            thread_id=thread["configurable"]["thread_id"],
            tool_calls=result["tool_calls"]
        )
    
    # 否则返回 completed 状态
    return ChatResponse(
        status="completed",
        response=result["response"],
        thread_id=thread["configurable"]["thread_id"],
        tool_calls=None
    )


def _multi_agent_execute(request):
    lead = LeadAgent()
    sub_tasks = lead.decompose(request.message)
    print(f"[API] LeadAgent 分解结果: {sub_tasks}")

    results = []
    for task in sub_tasks:
        sub = SubAgent(task["skill"])
        result = sub.execute(task["goal"])
        results.append(result)

    aggregator = Aggregator()
    response = aggregator.aggregate(results, request.message)

    return ChatResponse(
        status="completed",
        response=response,
        thread_id=request.thread_id or "multi",
        tool_calls=None
    )


@app.get("/tools/pending/{thread_id}", response_model=PendingToolCallsResponse)
def get_pending_tools(thread_id: str):
    tools = _pending_tools_store.get(thread_id)
    return PendingToolCallsResponse(
        thread_id=thread_id,
        tool_calls=tools,
        status="pending" if tools else "ready"
    )


@app.post("/tools/approve")
def approve_tools(request: ToolActionRequest, graph=Depends(get_graph)):
    """无状态网关：工具批准/拒绝接口
    
    通过 LangGraph 的 update_state API 实现无状态处理：
    - Approve: 传入 None 唤醒图，让 tool_node 正常执行
    - Reject: 伪造 ToolMessage 并用 update_state 注入，避免 400 错误
    """
    thread = {"configurable": {"thread_id": request.thread_id}}

    # 1. 检查图是否真的处于挂起状态
    state = graph.get_state(thread)
    if not state.next or "tool" not in state.next:
        return {
            "status": "error",
            "message": "当前 Thread 没有挂起的工具任务"
        }

    # 2. 从挂起的状态中获取真实的 tool_call_id
    messages = state.values.get("messages", [])
    if not messages:
        return {
            "status": "error",
            "message": "没有找到消息历史"
        }

    # 找到最后一条带 tool_calls 的消息
    last_tool_call_msg = None
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            last_tool_call_msg = msg
            break

    if not last_tool_call_msg:
        return {
            "status": "error",
            "message": "没有找到 tool_calls"
        }

    # 获取真实的 tool_call_id（如果请求中没有提供）
    tool_call_id = request.tool_call_id or (
        last_tool_call_msg.tool_calls[0].get("id") if hasattr(last_tool_call_msg.tool_calls[0], "get") 
        else last_tool_call_msg.tool_calls[0].id
    )
    tool_name = request.tool_name or (
        last_tool_call_msg.tool_calls[0].get("name") if hasattr(last_tool_call_msg.tool_calls[0], "get")
        else last_tool_call_msg.tool_calls[0].name
    )

    if request.action == "approve":
        # 从挂起的状态中获取所有的 tool_calls 并存入 approved_tools
        approved_tool_calls = state.values.get("tool_calls", [])
        graph.update_state(
            thread,
            {"approved_tools": approved_tool_calls}
        )
        
        # ✅ 【同意执行】：传入 None 唤醒图，让 tool_node 正常执行
        for event in graph.stream(None, thread):
            pass
        
    elif request.action == "reject":
        # ❌ 【拒绝执行】：防御 400 Bad Request 的终极黑客技巧
        # 因为我们跳过了 tool_node，大模型的 tool_calls 会"悬空"
        # 我们必须手动伪造一条 ToolMessage，骗过大模型！
        
        fake_tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            name=tool_name,
            content="用户拒绝了此操作。请向用户解释原因并询问下一步指示。"
        )
        
        # 魔法 API：强行修改状态，并声明我们是作为 tool_node 写入的
        graph.update_state(
            thread,
            {"messages": [fake_tool_message]},
            as_node="tool"
        )
        
        # 状态修改完毕，唤醒大脑继续推理
        for event in graph.stream(None, thread):
            pass

    # 3. 获取最新状态返回给前端
    new_state = graph.get_state(thread)
    final_message = new_state.values["messages"][-1]
    
    return {
        "status": "completed",
        "response": final_message.content
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
