# Phase 1 Design: Core Graph & HITL

## 1. 设计目标

搭建"大脑与刹车"（Core Graph & HITL）：
- 跑通基于 LangGraph 的主循环
- 实现最核心的"断点审批"风控机制
- 建立配置驱动的架构基础
- **为四大核心功能预留扩展接口**

## 2. 完整架构概览（三阶段演进）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Mini Agent Harness 完整架构                          │
│                    阶段一(当前) → 阶段二 → 阶段三                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ 阶段一: Core Graph & HITL (当前实现)                                         │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   config/    │    │    engine/   │    │    nodes/    │                  │
│  │  graph.yaml  │───▶│   builder    │───▶│  agent_node  │                  │
│  │              │    │   (动态建图)  │    │  tool_node   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                             │                                              │
│                             ▼                                              │
│                      ┌──────────────┐                                      │
│                      │  main.py     │                                      │
│                      │ (HITL交互)   │                                      │
│                      └──────────────┘                                      │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ 扩展
┌────────────────────────────────────────────────────────────────────────────┐
│ 阶段二: Context & Memory (护城河)                                           │
├────────────────────────────────────────────────────────────────────────────┤
│  新增组件:                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  middleware/    │  │  memory/        │  │  checkpointer/  │             │
│  │  token_budget   │  │  facts_db       │  │  sqlite         │             │
│  │  memory_inject  │  │  summary_cache  │  │  (持久化)        │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                            │
│  核心功能:                                                                 │
│  • Token 预算控制 (tiktoken)                                               │
│  • 上下文摘要压缩                                                          │
│  • 长期记忆注入                                                            │
│  • 状态持久化 (SQLite)                                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ 扩展
┌────────────────────────────────────────────────────────────────────────────┐
│ 阶段三: Skills & Gateway (工程外壳)                                         │
├────────────────────────────────────────────────────────────────────────────┤
│  新增组件:                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  skills/        │  │  gateway/       │  │  orchestration/ │             │
│  │  skill_loader   │  │  fastapi_app    │  │  lead_agent     │             │
│  │  tool_registry  │  │  sse_stream     │  │  sub_agents     │             │
│  │  markdown_parse │  │  rest_api       │  │  task_router    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                            │
│  核心功能:                                                                 │
│  • Markdown Skill 动态加载                                                 │
│  • Lead-Sub 多智能体并发                                                   │
│  • FastAPI + SSE 网关                                                      │
│  • 工具注册与发现                                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

## 3. 四大核心功能架构

### 3.1 动态 Skills & Tools 加载器 (阶段三实现)

```python
# 设计预留接口 (阶段一)
class SkillLoaderInterface:
    """阶段三实现: 动态 Skills & Tools 加载器"""
    
    def load_skill(self, skill_path: str) -> SkillConfig:
        """从 Markdown 文件加载 Skill
        
        解析 Frontmatter (YAML 头) 获取:
        - name: Skill 名称
        - tools: 所需工具列表
        - prompt: System Prompt 正文
        """
        pass
    
    def register_tools(self, skill_config: SkillConfig) -> list[Tool]:
        """注册 Skill 所需的工具到 ToolRegistry"""
        pass

# YAML 配置预留 (阶段三)
# skills/researcher.md
---
name: "深度研究员"
tools: ["web_search", "pdf_parser", "summarizer"]
model: "kimi"
---

你是一个专业的深度研究员，擅长...
```

**阶段一预留**: `engine/loader.py` 的 `GraphConfig` 类预留 `skills` 字段

### 3.2 Lead-Sub 多智能体并发 (阶段三实现)

```python
# 设计预留接口 (阶段一)
class OrchestrationInterface:
    """阶段三实现: Lead-Sub 多智能体编排"""
    
    def lead_plan(self, task: str) -> TaskList:
        """Lead Agent 拆解任务为子任务列表"""
        pass
    
    def spawn_sub_agents(self, tasks: TaskList) -> list[SubAgent]:
        """并行拉起多个 Sub Agent"""
        pass
    
    def aggregate_results(self, results: list[SubResult]) -> FinalAnswer:
        """汇总子任务结果"""
        pass

# ThreadState 预留字段 (阶段一)
class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: list
    approved_tools: list
    is_complete: bool
    # 阶段三扩展字段:
    # task_list: list[SubTask]      # 子任务列表
    # sub_agents: dict[str, State]  # 子智能体状态
    # aggregation_strategy: str     # 汇总策略
```

**阶段一预留**: `ThreadState` 预留 `task_list` 字段，图结构支持条件边扩展

### 3.3 状态隔离与摘要压缩 (阶段二实现)

```python
# 设计预留接口 (阶段一)
class ContextMiddlewareInterface:
    """阶段二实现: 上下文护城河"""
    
    def token_budget_check(self, messages: list) -> TokenStatus:
        """Token 预算检查 (tiktoken)"""
        pass
    
    def compress_context(self, messages: list, threshold: int) -> list:
        """超过阈值时触发摘要压缩"""
        pass
    
    def inject_summaries(self, state: ThreadState) -> ThreadState:
        """注入历史摘要到 System Prompt"""
        pass

# ThreadState 预留字段 (阶段一)
class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: list
    approved_tools: list
    is_complete: bool
    # 阶段二扩展字段:
    # token_count: int              # 当前 Token 数
    # summary_context: str          # 摘要上下文
    # compression_history: list     # 压缩历史
```

**阶段一预留**: `engine/builder.py` 支持 middleware 链式调用接口

### 3.4 基于置信度的长期记忆库 (阶段二实现)

```python
# 设计预留接口 (阶段一)
class MemoryStoreInterface:
    """阶段二实现: 长期记忆库"""
    
    def store_fact(self, user_id: str, fact: str, confidence: float):
        """存储带置信度的事实"""
        pass
    
    def retrieve_facts(self, user_id: str, query: str, top_k: int = 5) -> list[Fact]:
        """检索相关记忆"""
        pass
    
    def inject_to_prompt(self, facts: list[Fact]) -> str:
        """将记忆注入 System Prompt <memory> 标签"""
        pass

# 数据结构 (阶段二)
# SQLite 表: facts
# - id: INTEGER PRIMARY KEY
# - user_id: TEXT
# - fact: TEXT
# - confidence: FLOAT (0-1)
# - timestamp: DATETIME
# - access_count: INTEGER
```

**阶段一预留**: `ThreadState` 预留 `memory_context` 字段

## 4. 阶段一核心组件设计

### 4.1 ThreadState（状态定义）

```python
class ThreadState(TypedDict):
    """State for a single conversation thread.
    
    阶段一基础字段 + 阶段二/三扩展字段预留
    """
    # 阶段一: Core Graph & HITL
    messages: Annotated[list, add_messages]  # 对话历史
    tool_calls: list                         # 待执行的工具调用
    approved_tools: list                     # 已审批的工具
    is_complete: bool                        # 是否完成
    
    # 阶段二: Context & Memory (预留)
    # token_count: int                       # 当前 Token 数
    # summary_context: str                   # 摘要上下文
    # memory_context: str                    # 注入的记忆
    
    # 阶段三: Lead-Sub Orchestration (预留)
    # task_list: list[SubTask]               # 子任务列表
    # sub_agent_states: dict                 # 子智能体状态
```

### 4.2 图结构（由 YAML 配置驱动）

```yaml
graph:
  name: "mini_harness_v1"
  checkpointer: "memory"     # 阶段一: memory, 阶段二: sqlite
  
  # 阶段一: 简单 Agent-Tool 循环
  nodes:
    - name: "agent"
      type: "agent"
      model: "kimi"
      # 阶段二扩展: middleware: [token_budget, memory_inject]
      
    - name: "tool"
      type: "tool"
      require_approval: true   # HITL 标记
  
  edges:
    - from: "agent"
      to: "tool"
      condition: "has_tool_calls"
      
    - from: "agent"
      to: "END"
      condition: "is_complete"
      
    - from: "tool"
      to: "agent"
  
  # 阶段三扩展: skills 配置
  # skills:
  #   - path: "skills/researcher.md"
  #   - path: "skills/coder.md"
```

### 4.3 HITL 机制

```python
# 使用 LangGraph 的 interrupt
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tool_node"]  # 在工具节点前挂起
)

# 恢复时人工审批
if graph.get_state(thread).next:
    print(f"待审批工具: {state['tool_calls']}")
    approval = input("批准执行? (y/n): ")
```

## 5. 数据流

```
用户输入 ──▶ agent_node ──┬──▶ 需要工具? ──▶ [HITL挂起] ──▶ tool_node ──▶ agent_node
                          │                      ▲
                          └──▶ 直接回答? ──▶ END  │
                                               (人工输入 y/n 恢复)

阶段二扩展 (Middleware 链):
agent_node 前插入:
  token_budget_check ──▶ memory_inject ──▶ agent_node

阶段三扩展 (Lead-Sub):
复杂任务拆解:
  lead_agent ──▶ [并行] sub_agent_1, sub_agent_2, ... ──▶ aggregate
```

## 6. 文件结构（三阶段演进）

```
mini_harness/
├── config/
│   └── graph.yaml              # 图配置 (阶段一)
│   # 阶段三新增: skills/*.md
│
├── engine/
│   ├── __init__.py
│   ├── state.py                # ThreadState (预留扩展字段)
│   ├── loader.py               # YAML加载 (预留 skills 解析)
│   ├── builder.py              # 图构建器 (预留 middleware 接口)
│   # 阶段二新增:
│   # ├── middleware/
│   # │   ├── token_budget.py
│   # │   └── memory_inject.py
│   # └── memory/
│   #     ├── store.py
│   #     └── models.py
│   # 阶段三新增:
│   # └── orchestration/
│   #     ├── lead_agent.py
│   #     └── sub_agent.py
│
├── nodes/
│   ├── __init__.py
│   ├── agent.py                # Agent节点
│   └── tool.py                 # Tool节点
│   # 阶段三新增:
│   # └── skill_node.py
│
├── skills/                     # 阶段三: Skill 定义目录
│   # └── researcher.md
│
├── gateway/                    # 阶段三: FastAPI 网关
│   # ├── __init__.py
│   # ├── app.py
│   # └── sse.py
│
├── main.py                     # 入口 + HITL (阶段一 CLI)
│   # 阶段三: 改为 gateway 启动或保留 CLI 模式
│
├── requirements.txt
└── .env                        # Kimi API Key
```

## 7. 配置 Schema（含扩展预留）

### 7.1 graph.yaml（阶段一 + 预留）

```yaml
graph:
  name: "mini_harness_v1"
  version: "1.0.0"
  checkpointer: "memory"         # 阶段一: memory, 阶段二: sqlite
  
  # 阶段一: 基础节点
  nodes:
    - name: "agent"
      type: "agent"
      model: "kimi"
      temperature: 0.7
      # 阶段二扩展:
      # middleware:
      #   - token_budget
      #   - memory_inject
      
    - name: "tool"
      type: "tool"
      require_approval: true
      # 阶段三扩展:
      # tools: ["web_search", "file_edit"]
  
  edges:
    - from: "agent"
      to: "tool"
      condition: "has_tool_calls"
      
    - from: "agent"
      to: "END"
      condition: "is_complete"
      
    - from: "tool"
      to: "agent"
  
  # 阶段二扩展: 上下文配置
  # context:
  #   token_limit: 8000
  #   compression_threshold: 6000
  #   summary_model: "moonshot-v1-8k"
  
  # 阶段三扩展: Skills 配置
  # skills:
  #   - name: "researcher"
  #     path: "skills/researcher.md"
  #     auto_load: true
```

## 8. 关键设计决策

| 决策 | 选择 | 理由 | 阶段 |
|------|------|------|------|
| Checkpointer | `MemorySaver` | 阶段一只需内存，阶段二换 SQLite | 一 |
| LLM 客户端 | `openai` 库兼容模式 | Kimi API 兼容 OpenAI 格式 | 一 |
| 配置格式 | YAML | 易读易改，支持三阶段扩展 | 一 |
| HITL 方式 | 命令行交互 | 最纯粹，便于调试核心机制 | 一 |
| Token 计算 | `tiktoken` | OpenAI 官方，精准预算控制 | 二 |
| 记忆存储 | SQLite + JSON | 轻量级，足够支撑长期记忆 | 二 |
| Skill 格式 | Markdown + Frontmatter | 业务解耦，非技术人员可编辑 | 三 |
| 网关协议 | FastAPI + SSE | 原生异步，流式输出 | 三 |

## 9. 依赖项（三阶段汇总）

```
# 阶段一
langgraph>=0.2.0
langchain-openai>=0.2.0
pyyaml>=6.0
python-dotenv>=1.0.0

# 阶段二新增
tiktoken>=0.7.0

# 阶段三新增
fastapi>=0.110.0
uvicorn>=0.27.0
sse-starlette>=2.0.0
```

## 10. 成功标准（分阶段）

### 阶段一: Core Graph & HITL
- [ ] 能根据 `graph.yaml` 动态构建 LangGraph
- [ ] Agent 能调用 Kimi API 并返回响应
- [ ] 工具调用前能正确挂起等待审批
- [ ] 输入 `y` 后能恢复执行并返回结果
- [ ] 输入 `n` 后能跳过并返回拒绝信息

### 阶段二: Context & Memory
- [ ] Token 预算超过阈值时触发摘要压缩
- [ ] 长期记忆能正确注入 System Prompt
- [ ] 状态持久化到 SQLite，支持断点恢复
- [ ] 无限轮次对话不崩溃

### 阶段三: Skills & Gateway
- [ ] 能从 Markdown 动态加载 Skill
- [ ] Lead Agent 能拆解任务并行执行
- [ ] FastAPI + SSE 流式输出正常
- [ ] 工具注册与发现机制完善

## 11. 风险与缓解

| 风险 | 缓解措施 | 阶段 |
|------|----------|------|
| Kimi API 额度不足 | 添加请求日志，便于追踪消耗 | 一 |
| YAML 配置错误 | 启动时验证 schema，给出清晰错误 | 一 |
| 状态丢失 | 阶段二使用 SQLite checkpointer | 二 |
| Token 计算不准 | 使用 tiktoken，预留 10% buffer | 二 |
| Skill 加载失败 | 优雅降级，使用默认配置 | 三 |
| 子智能体死锁 | 设置超时和重试机制 | 三 |

## 12. 阶段演进路线图

```
阶段一 (当前) ──▶ 阶段二 ──▶ 阶段三
     │              │            │
     ▼              ▼            ▼
 基础框架       护城河        工程外壳
   Core      Context &      Skills &
   + HITL      Memory        Gateway
     │              │            │
  验证:          验证:         验证:
  HITL工作      长对话稳定     完整产品
  配置驱动      记忆注入       多智能体
```

## 13. 接口预留清单

为确保三阶段平滑演进，阶段一需预留以下接口：

| 接口 | 位置 | 用途 | 实现阶段 |
|------|------|------|----------|
| `middleware_chain` | `engine/builder.py` | 节点前中间件链 | 二 |
| `memory_context` | `engine/state.py` | 记忆注入字段 | 二 |
| `task_list` | `engine/state.py` | 子任务列表 | 三 |
| `skills` | `config/graph.yaml` | Skill 配置段 | 三 |
| `tool_registry` | `nodes/tool.py` | 动态工具注册 | 三 |
