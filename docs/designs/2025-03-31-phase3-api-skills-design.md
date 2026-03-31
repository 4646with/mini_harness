# Phase 3 API & Skill System 文档

> **状态:** 已实现 (Commit: `33646d1`)

## 一、架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ /health │  │  /skills  │  │   /chat      │  │/tools/pending│ │
│  └──────────┘  └──────────┘  └──────────────┘  └─────────────┘  │
└────────┬───────────────────────────┬────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ TaskClassifier  │         │   SkillRegistry  │
│  (任务路由)      │         │   (动态技能)     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Runtime                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Agent   │  │   Tool   │  │Middleware │  │ Checkpoint│   │
│  │  Node    │  │   Node   │  │  Chain   │  │  SQLite  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 二、API 端点

### 2.1 健康检查
```
GET /health
```
**响应:**
```json
{"status": "ok", "service": "mini-harness-v3"}
```

### 2.2 列出技能
```
GET /skills
```
**响应:**
```json
[
  {"name": "save_memory", "description": "将重要信息保存到长期记忆"}
]
```

### 2.3 对话
```
POST /chat
Content-Type: application/json

{
  "message": "用户输入",
  "thread_id": "可选-会话ID"
}
```
**响应:**
```json
{
  "response": "AI 回复内容",
  "thread_id": "会话ID",
  "tool_calls": [{"name": "save_memory", "args": {...}}]
}
```

### 2.4 待批准工具
```
GET /tools/pending/{thread_id}
```
**响应:**
```json
{
  "thread_id": "会话ID",
  "tool_calls": [...],
  "status": "pending"
}
```

### 2.5 批准/拒绝工具
```
POST /tools/approve
Content-Type: application/json

{
  "thread_id": "会话ID",
  "approved": true
}
```

## 三、任务路由 (DeerFlow 风格)

### 3.1 任务分类器
```python
class TaskClassifier:
    COMPLEX_KEYWORDS = [
        "分析", "拆解", "规划", "研究", "比较",
        "多个", "不同", "哪些", "如何完成"
    ]

    def classify(self, user_input: str) -> TaskType:
        # 命中关键词 → MULTI_AGENT
        # 否则 → DIRECT
```

### 3.2 简单任务 (DIRECT)
直接调用 LangGraph stream，实时返回结果。

### 3.3 复杂任务 (MULTI_AGENT)
```
LeadAgent.decompose() → 分解子任务
SubAgent.execute()    → 执行每个子任务
Aggregator.aggregate()→ 聚合结果
```

## 四、Skill 系统

### 4.1 Skill 基类
```python
class Skill(ABC):
    name: str
    description: str
    parameters: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        pass
```

### 4.2 Skill 注册表
```python
class SkillRegistry:
    def register(self, skill: Skill)
    def get(self, name: str) -> Optional[Skill]
    def list_skills(self) -> List[Dict]
    def load_from_module(self, module_path: str)
```

### 4.3 内置技能

#### save_memory
```python
class SaveMemorySkill(Skill):
    name = "save_memory"
    description = "将重要信息保存到长期记忆"
    parameters = {
        "memories": [
            {"content": "记忆内容", "memory_type": "general|preference|fact"}
        ]
    }
```

## 五、动态工具注册

### 5.1 机制
```python
# nodes/tool.py
_tool_handlers: Dict[str, Callable] = {}

def register_tool(name: str, handler: Callable):
    _tool_handlers[name] = handler

def _execute_by_name(tool_name: str, tool_args: dict) -> str:
    if tool_name in _tool_handlers:
        return _tool_handlers[tool_name](tool_args)

    # 尝试从 SkillRegistry 获取
    if skill := get_skill_registry().get(tool_name):
        return asyncio.run(skill.execute(**tool_args))

    return f"[错误] 未知工具: {tool_name}"
```

### 5.2 注册内置工具
```python
# 模块加载时自动注册
register_tool("save_memory", _execute_save_memory)
```

## 六、配置

### 6.1 graph.yaml 技能配置
```yaml
skills:
  enabled: true
  auto_load:
    - "skills.builtin.save_memory"
  registry: "dynamic"
```

## 七、已知问题 (Code Review)

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | `graph.stream()` 同步阻塞 FastAPI | Important | 待修复 |
| 2 | `_pending_tools` 全局状态线程不安全 | Important | 待修复 |
| 3 | SubAgent 忽略 LLM 文本响应 | Important | 待修复 |
| 4 | `asyncio.run()` 在已有 loop 环境崩溃 | Important | 待修复 |

## 八、启动方式

```bash
# 开发模式
python -m api.main

# 生产模式
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

访问 http://localhost:8000/docs 查看 Swagger UI。
