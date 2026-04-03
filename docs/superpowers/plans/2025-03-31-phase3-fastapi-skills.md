# Phase 3: FastAPI & 动态技能加载 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Mini Agent Harness 添加 FastAPI 网关和动态 Skill 加载能力

**Architecture:**
- FastAPI 作为 HTTP 网关，替代当前的 CLI main.py
- Skill 系统：动态加载 Python 模块作为工具技能
- 支持多租户/多线程会话管理
- 保留现有 HITL 机制

**Tech Stack:** FastAPI, uvicorn, Python asyncio, importlib

---

## 文件结构

```
Mini_Harness/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py          # /chat 端点
│   │   └── skills.py        # /skills 端点
│   ├── models.py            # Pydantic 请求/响应模型
│   └── deps.py              # 依赖注入
├── skills/
│   ├── __init__.py
│   ├── base.py              # Skill 基类
│   ├── registry.py          # Skill 注册表
│   └── builtin/
│       ├── __init__.py
│       └── save_memory.py   # 内置记忆技能
├── engine/state.py          # 修改：添加 task_list
├── config/graph.yaml        # 修改：添加 skills 配置段
└── nodes/tool.py            # 修改：动态工具注册
```

---

## 任务清单

### Task 0: 任务路由层 (DeerFlow 风格)

**Files:**
- Create: `router/__init__.py`
- Create: `router/classifier.py`
- Create: `router/agents.py`

- [ ] **Step 1: 创建 router/classifier.py - 任务复杂度分类器**

```python
"""任务复杂度分类器 - DeerFlow 风格"""
from enum import Enum
from typing import List
from engine.patched_kimi import get_kimi_llm


class TaskType(Enum):
    DIRECT = "direct"           # 简单任务，直接执行
    MULTI_AGENT = "multi_agent"  # 复杂任务，多智能体


class TaskClassifier:
    """分析用户输入，决定任务类型"""

    COMPLEX_KEYWORDS = [
        "分析", "拆解", "规划", "研究", "比较",
        "多个", "不同", "哪些", "如何完成",
        "分析一下", "拆解成", "调查", "研究"
    ]

    def classify(self, user_input: str) -> TaskType:
        """判断任务复杂度"""
        # 规则判断：简单关键词匹配
        for kw in self.COMPLEX_KEYWORDS:
            if kw in user_input:
                return TaskType.MULTI_AGENT
        return TaskType.DIRECT

    def classify_with_llm(self, user_input: str) -> tuple[TaskType, str]:
        """用 LLM 辅助判断（更准确但更慢）"""
        llm = get_kimi_llm()
        prompt = f"""分析以下用户输入，判断任务复杂度：

用户输入: {user_input}

规则：
- 简单任务：单个问题、单一操作、直接回答（如"今天天气如何"、"帮我记住我的名字"）
- 复杂任务：需要多步骤、多个子任务、规划研究、多技能协作

直接回答：direct 或 multi_agent""" 

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        
        if "multi" in decision:
            return TaskType.MULTI_AGENT, "LLM 判断为复杂任务"
        return TaskType.DIRECT, "LLM 判断为简单任务"


_classifier = TaskClassifier()


def get_classifier() -> TaskClassifier:
    return _classifier
```

- [ ] **Step 2: 创建 router/agents.py - Lead-Sub 多智能体**

```python
"""Lead Agent 和 Sub Agent - 复杂任务处理"""
from typing import List, Dict, Any
from engine.patched_kimi import get_kimi_llm


class LeadAgent:
    """Lead Agent: 分解复杂任务"""

    def __init__(self):
        self.llm = get_kimi_llm()

    def decompose(self, task: str) -> List[Dict[str, Any]]:
        """将复杂任务分解为子任务"""
        prompt = f"""将以下任务分解为具体的子任务步骤：

任务: {task}

输出格式（JSON数组）：
[{{"skill": "技能名", "goal": "子目标", "priority": 优先级}}]

只输出JSON，不要其他内容。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        # 解析 JSON 子任务
        # ... (简化实现)
        return [{"skill": "unknown", "goal": task, "priority": 1}]


class SubAgent:
    """Sub Agent: 执行单个子任务"""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.llm = get_kimi_llm()

    def execute(self, goal: str) -> str:
        """执行单个子任务"""
        # 调用对应 Skill 执行
        # ...
        return f"[完成] {goal}"


class Aggregator:
    """结果聚合器"""

    def aggregate(self, results: List[str], original_task: str) -> str:
        """聚合多个子任务结果"""
        prompt = f"""原始任务: {original_task}

执行结果:
{chr(10).join(results)}

请用一段话总结以上结果，回复用户。"""

        llm = get_kimi_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

- [ ] **Step 3: 创建 router/__init__.py**

```python
"""任务路由层"""
from router.classifier import TaskType, TaskClassifier, get_classifier
from router.agents import LeadAgent, SubAgent, Aggregator

__all__ = ["TaskType", "TaskClassifier", "get_classifier", "LeadAgent", "SubAgent", "Aggregator"]
```

- [ ] **Step 4: 创建 api/routes/chat.py - 路由集成**

```python
"""聊天路由 - 集成路由层"""
from fastapi import APIRouter, Depends
from api.models import ChatRequest, ChatResponse
from api.deps import get_graph
from router import TaskType, get_classifier, LeadAgent, SubAgent, Aggregator

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, graph=Depends(get_graph)):
    classifier = get_classifier()
    task_type = classifier.classify(request.message)

    if task_type == TaskType.DIRECT:
        # 简单任务：直接走 LangGraph
        return _direct_execute(request, graph)
    else:
        # 复杂任务：Lead-Sub 多智能体
        return _multi_agent_execute(request)


def _direct_execute(request, graph):
    # ... 现有 LangGraph 执行逻辑
    pass


def _multi_agent_execute(request):
    # Lead Agent 分解
    lead = LeadAgent()
    sub_tasks = lead.decompose(request.message)

    # Sub Agent 执行
    results = []
    for task in sub_tasks:
        sub = SubAgent(task["skill"])
        result = sub.execute(task["goal"])
        results.append(result)

    # 聚合结果
    aggregator = Aggregator()
    response = aggregator.aggregate(results, request.message)

    return ChatResponse(response=response, thread_id=request.thread_id or "multi")
```

- [ ] **Step 5: 提交**

```bash
git add router/
git commit -m "feat(router): add DeerFlow-style task classifier and Lead-Sub agents"
```

---

### Task 1: 创建 Skill 基类和注册表

**Files:**
- Create: `skills/base.py`
- Create: `skills/registry.py`
- Create: `skills/__init__.py`

- [ ] **Step 1: 创建 skills/base.py - Skill 基类**

```python
"""Skill 基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Skill(ABC):
    """动态技能基类"""

    name: str
    description: str
    parameters: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行技能"""
        pass
```

- [ ] **Step 2: 创建 skills/registry.py - 技能注册表**

```python
"""Skill 注册表"""
import importlib
from typing import Dict, Optional
from skills.base import Skill


class SkillRegistry:
    """动态技能注册表"""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_skills(self) -> list[Dict]:
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def load_from_module(self, module_path: str):
        """动态加载技能模块"""
        module = importlib.import_module(module_path)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Skill) and attr != Skill:
                self.register(attr())


_registry = SkillRegistry()


def get_skill_registry() -> SkillRegistry:
    return _registry
```

- [ ] **Step 3: 创建 skills/__init__.py**

```python
"""Skill 系统"""
from skills.base import Skill
from skills.registry import SkillRegistry, get_skill_registry

__all__ = ["Skill", "SkillRegistry", "get_skill_registry"]
```

- [ ] **Step 4: 提交**

```bash
git add skills/base.py skills/registry.py skills/__init__.py
git commit -m "feat(skills): add base Skill class and registry"
```

---

### Task 2: 创建 FastAPI 基础结构

**Files:**
- Create: `api/__init__.py`
- Create: `api/main.py`
- Create: `api/models.py`
- Create: `api/deps.py`

- [ ] **Step 1: 创建 api/models.py - Pydantic 模型**

```python
"""API 请求/响应模型"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class SkillInfo(BaseModel):
    name: str
    description: str
```

- [ ] **Step 2: 创建 api/deps.py - 依赖注入**

```python
"""FastAPI 依赖注入"""
from engine.builder import build_graph
from engine.loader import load_config
from config import pathlib


_config_cache = None
_graph_cache = None


def get_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def get_graph():
    global _graph_cache
    if _graph_cache is None:
        config = get_config()
        _graph_cache = build_graph(config)
    return _graph_cache
```

- [ ] **Step 3: 创建 api/main.py - FastAPI 入口**

```python
"""FastAPI 网关入口"""
from fastapi import FastAPI, Depends
from api.models import ChatRequest, ChatResponse, SkillInfo
from api.deps import get_graph, get_config
from skills.registry import get_skill_registry

app = FastAPI(title="Mini Agent Harness API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/skills", response_model=List[SkillInfo])
def list_skills():
    registry = get_skill_registry()
    return registry.list_skills()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, graph=Depends(get_graph)):
    # TODO: 实现聊天逻辑
    return ChatResponse(response="TODO", thread_id=request.thread_id or "default")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: 创建 api/__init__.py**

```python
"""FastAPI 网关"""
```

- [ ] **Step 5: 提交**

```bash
git add api/
git commit -m "feat(api): add FastAPI skeleton with basic endpoints"
```

---

### Task 3: 实现 /chat 端点 - 集成 LangGraph

**Files:**
- Modify: `api/main.py`
- Modify: `engine/state.py` - 添加 task_list 字段

- [ ] **Step 1: 修改 engine/state.py - 添加 task_list**

```python
class ThreadState(TypedDict):
    # ... existing fields ...

    # Phase 3 Extension Fields
    task_list: list
    current_skill: Optional[str]
```

- [ ] **Step 2: 修改 api/main.py - 实现聊天端点**

```python
import uuid
from langgraph.graph import END

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, graph=Depends(get_graph)):
    thread = {"configurable": {"thread_id": request.thread_id or str(uuid.uuid4())}}

    # 检查是否有待处理工具调用需要 HITL
    # 简化处理：自动批准或返回 pending 状态

    input_message = {"role": "user", "content": request.message}

    # 流式响应
    response_text = ""
    tool_calls = []

    for event in graph.stream({"messages": [input_message]}, thread):
        if "agent" in event:
            messages = event["agent"].get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    response_text = last_msg.content
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_calls = last_msg.tool_calls

    return ChatResponse(
        response=response_text,
        thread_id=thread["configurable"]["thread_id"],
        tool_calls=tool_calls if tool_calls else None
    )
```

- [ ] **Step 3: 提交**

```bash
git add api/main.py engine/state.py
git commit -m "feat(api): integrate LangGraph with /chat endpoint"
```

---

### Task 4: 实现动态工具注册到 tool_node

**Files:**
- Modify: `nodes/tool.py` - 添加 tool_registry
- Modify: `config/graph.yaml` - 添加 skills 配置段

- [ ] **Step 1: 修改 nodes/tool.py - 添加动态工具注册**

```python
"""执行已批准工具调用的工具节点"""
from typing import Dict, Callable
from langchain_core.messages import ToolMessage
from engine.state import ThreadState
from memory.store import get_memory_store
from skills.registry import get_skill_registry


# 动态工具注册表
_tool_handlers: Dict[str, Callable] = {
    "save_memory": _execute_save_memory,
}


def register_tool(name: str, handler: Callable):
    """注册工具处理器"""
    _tool_handlers[name] = handler


def tool_node(state: ThreadState) -> dict:
    """执行已批准工具调用的工具节点"""
    approved_tools = state.get("approved_tools", [])
    skill_registry = get_skill_registry()

    if not approved_tools:
        return {"messages": [], "approved_tools": []}

    results = []
    for tool_call in approved_tools:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        # 1. 先查内置工具
        if tool_name in _tool_handlers:
            result_content = _tool_handlers[tool_name](tool_args)
        # 2. 再查 Skill 注册表
        elif skill := skill_registry.get(tool_name):
            import asyncio
            result_content = asyncio.run(skill.execute(**tool_args))
        else:
            result_content = f"[错误] 未知工具: {tool_name}"

        results.append(ToolMessage(
            content=result_content,
            tool_call_id=tool_call_id
        ))

    return {"messages": results, "approved_tools": []}
```

- [ ] **Step 2: 修改 config/graph.yaml - 添加 skills 配置段**

```yaml
graph:
  name: "mini_harness_v2"
  checkpointer: "sqlite"

  # Phase 3: Skills Configuration
  skills:
    enabled: true
    auto_load:
      - "skills.builtin.save_memory"
    registry: "dynamic"  # dynamic or static
```

- [ ] **Step 3: 提交**

```bash
git add nodes/tool.py config/graph.yaml
git commit -m "feat(skills): add dynamic tool registry to tool_node"
```

---

### Task 5: 创建内置 Skill 示例

**Files:**
- Create: `skills/builtin/__init__.py`
- Create: `skills/builtin/save_memory.py`

- [ ] **Step 1: 创建 skills/builtin/save_memory.py**

```python
"""内置记忆技能"""
from skills.base import Skill
from memory.store import get_memory_store


class SaveMemorySkill(Skill):
    """保存记忆技能"""

    name = "save_memory"
    description = "将重要信息保存到长期记忆"
    parameters = {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "memory_type": {"type": "string", "enum": ["profile", "preference", "fact"]}
                    }
                }
            }
        },
        "required": ["memories"]
    }

    def execute(self, memories: list, **kwargs) -> str:
        store = get_memory_store()
        results = []
        for mem in memories:
            content = mem.get("content")
            memory_type = mem.get("memory_type", "general")
            if content:
                res = store.save_memory(content, memory_type)
                results.append(res)
        return "\n".join(results) if results else "无记忆可保存"
```

- [ ] **Step 2: 创建 skills/builtin/__init__.py**

```python
"""内置技能"""
from skills.builtin.save_memory import SaveMemorySkill

__all__ = ["SaveMemorySkill"]
```

- [ ] **Step 3: 提交**

```bash
git add skills/builtin/
git commit -m "feat(skills): add builtin SaveMemorySkill"
```

---

### Task 6: 添加 HITL API 端点

**Files:**
- Modify: `api/main.py` - 添加 HITL 相关端点

- [ ] **Step 1: 修改 api/models.py - 添加 HITL 模型**

```python
class ToolApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


class PendingToolCallsResponse(BaseModel):
    thread_id: str
    tool_calls: List[Dict[str, Any]]
    status: str  # "pending" or "ready"
```

- [ ] **Step 2: 修改 api/main.py - 添加 HITL 端点**

```python
from api.models import ToolApprovalRequest, PendingToolCallsResponse

# 存储待处理的工具调用（生产环境用 Redis）
_pending_tools: Dict[str, List[Dict]] = {}


@app.get("/tools/pending/{thread_id}", response_model=PendingToolCallsResponse)
def get_pending_tools(thread_id: str):
    tools = _pending_tools.get(thread_id, [])
    return PendingToolCallsResponse(
        thread_id=thread_id,
        tool_calls=tools,
        status="pending" if tools else "ready"
    )


@app.post("/tools/approve")
def approve_tools(request: ToolApprovalRequest, graph=Depends(get_graph)):
    thread = {"configurable": {"thread_id": request.thread_id}}

    if request.approved:
        # 恢复执行
        graph.update_state(thread, {"approved_tools": _pending_tools.get(request.thread_id, [])})
        _pending_tools[request.thread_id] = []

        # 继续执行
        # ... 流式响应逻辑
    else:
        # 拒绝
        _pending_tools[request.thread_id] = []
        graph.update_state(thread, {"approved_tools": [], "is_complete": True})

    return {"status": "ok"}
```

- [ ] **Step 3: 提交**

```bash
git add api/main.py api/models.py
git commit -m "feat(api): add HITL approval endpoints"
```

---

## 成功标准

- [ ] FastAPI 服务可以启动并响应 `/health`
- [ ] `GET /skills` 返回已注册技能列表
- [ ] `POST /chat` 可以处理对话并返回响应
- [ ] Skill 可以动态加载和执行
- [ ] 保留原有的 CLI 入口 `main.py` 作为备选

---

## 后续扩展

- WebSocket 支持流式响应
- Redis 会话管理
- Docker 部署配置
