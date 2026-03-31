"""FastAPI 网关入口"""
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from api.models import (
    ChatRequest, ChatResponse, SkillInfo,
    ToolApprovalRequest, PendingToolCallsResponse
)
from api.deps import get_graph, get_config
from skills.registry import get_skill_registry
from router import TaskType, get_classifier, LeadAgent, SubAgent, Aggregator


class PendingToolsStore:
    """线程安全的待批准工具存储"""

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
    import uuid
    import asyncio
    from langchain_core.messages import HumanMessage

    thread = {"configurable": {"thread_id": request.thread_id or str(uuid.uuid4())}}
    input_message = {"role": "user", "content": request.message}

    response_text = ""
    tool_calls = []

    def run_stream():
        for event in graph.stream({"messages": [input_message]}, thread):
            if "agent" in event:
                messages = event["agent"].get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        response_text = last_msg.content
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tool_calls = last_msg.tool_calls

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_stream)

    return ChatResponse(
        response=response_text,
        thread_id=thread["configurable"]["thread_id"],
        tool_calls=tool_calls if tool_calls else None
    )


def _multi_agent_execute(request):
    lead = LeadAgent()
    sub_tasks = lead.decompose(request.message)

    results = []
    for task in sub_tasks:
        sub = SubAgent(task["skill"])
        result = sub.execute(task["goal"])
        results.append(result)

    aggregator = Aggregator()
    response = aggregator.aggregate(results, request.message)

    return ChatResponse(
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
def approve_tools(request: ToolApprovalRequest, graph=Depends(get_graph)):
    thread = {"configurable": {"thread_id": request.thread_id}}

    if request.approved:
        graph.update_state(thread, {"approved_tools": _pending_tools_store.get(request.thread_id)})
        _pending_tools_store.clear(request.thread_id)
    else:
        _pending_tools_store.clear(request.thread_id)
        graph.update_state(thread, {"approved_tools": [], "is_complete": True})

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
