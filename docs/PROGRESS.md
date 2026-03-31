# 当前进度摘要

## 一、完成的工作

### 1.1 SQLite 持久化 (Checkpointer)
- 安装依赖: `pip install langgraph-checkpoint-sqlite`
- 修改 `engine/builder.py`: 添加 SqliteSaver 支持
- 修改 `config/graph.yaml`: checkpointer 从 "memory" 改为 "sqlite"
- 数据存储在: `data/checkpoints.db`

### 1.2 架构调整
- 创建 `engine/stream_bridge.py`: 事件标准化 (text_reply/tool_call/billing)
- 修改 `nodes/middleware.py`: memory_inject_node 只写 state，不修改 messages
- 修改 `nodes/agent.py`: 从 state.memory_context 构建 prompt
- 修改 `main.py`: HITL 恢复逻辑 (update_state + stream)

---

## 二、当前遇到的问题 (BUG)

### 问题描述
```
400 Bad Request: an assistant message with 'tool_calls' must be followed by 
tool messages responding to each 'tool_call_id'. The following tool_call_ids 
did not have response messages: save_memory:0
```

### 问题分析
错误发生在第二轮 `stream(None, thread, ...)` 恢复时：
1. 第一轮：agent 返回 tool_calls → interrupt 暂停 ✅
2. 用户批准 → update_state → 进入 tool 节点
3. tool_node 执行工具，返回 ToolMessage ✅
4. tool → agent (自动继续) → 再次调用 LLM ❌ **400 错误**

### 根本原因
LangGraph 恢复时，消息历史中包含：
- 用户消息 (第一轮)
- AI 消息带 tool_calls
- 但缺少 ToolMessage

这表明 `update_state` 后，tool_node 执行结果没有正确合并到消息历史。

---

## 三、代码架构

### 3.1 文件结构
```
Mini_Harness/
├── main.py                    # CLI 入口，HITL 控制
├── config/graph.yaml         # 图配置
├── engine/
│   ├── builder.py         # 构建图 + SqliteSaver
│   ├── state.py           # ThreadState 定义
│   ├── stream_bridge.py    # 事件标准化
│   └── patched_kimi.py    # Kimi k2.5 兼容
├── nodes/
│   ├── agent.py          # Agent 节点
│   ├── tool.py           # Tool 执行节点
│   └── middleware.py     # token_budget, summarize, memory_inject
└── memory/store.py       # 记忆存储
```

### 3.2 图结构
```
__start__ → token_budget → memory_inject → [needs_s?]→ summarize → agent → [tool_calls?]→ tool
                    ↓                                              ↓
                   end                                           END
```
- `interrupt_before=["tool"]` 当 require_approval=true

### 3.3 ThreadState 定义
```python
class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史 (reducer)
    tool_calls: list
    approved_tools: list
    is_complete: bool
    token_count: int
    token_budget: int
    summary_context: str
    memory_context: list
    needs_summarization: bool
```

---

## 四、调试日志 (debug_test.py)

```
=== Turn 1: First message ===
Event: ['token_budget']
Event: ['memory_inject']
Event: ['agent']
Event: ['__interrupt__']        # ← 中断暂停

State next: ('tool',)           # ← 等待调用 tool
State values keys: ['messages', 'tool_calls', ...]

--- HITL: Approving ---
Tool calls to approve: [{'name': 'save_memory', 'id': 'save_memory:0', ...}]

--- Stream 2: Resume ---
[DEBUG tool_node] state.tool_calls: []          # ← 空！
[DEBUG tool_node] state.approved_tools: [...] # ← 有值

[错误] 400: tool_call_ids did not have response messages
```

---

## 五、待解决的问题

### 关键问题
`update_state` 后，`state.tool_calls` 为空，但 `state.approved_tools` 有值。

这说明：
1. update_state 只更新了 approved_tools
2. 但没有触发 tool 节点执行后的消息合并

### 可能的原因
- LangGraph interrupt 机制与 update_state 配合有问题
- 需要在 update_state 后手动触发下一步

### 尝试过的方案
1. ✅ graph.stream(None, thread, ...) - 恢复执行
2. ❌ graph.update_state(...) + graph.invoke(...) - 参数不匹配

---

## 六、下一步

需要调试 `update_state` 后消息是否正确合并。
或者考虑换一种 HITL 处理方式：
- 不用 interrupt，用条件路由
- 或者在 tool_node 内部处理 approval