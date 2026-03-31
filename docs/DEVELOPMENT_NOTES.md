# Mini Harness 开发笔记

## 项目概述

Mini Harness 是一个基于 LangGraph 的 AI Agent 脚手架，参考工业级项目 [DeerFlow](https://github.com/facebookresearch/deerflow) 的架构设计，实现了完整的对话式 AI Agent 能力。

## 架构演进历程

### Phase 1: 核心 Agent + HITL

**目标**：搭建基础的 Agent 对话框架，支持工具调用和人类审批。

**核心组件**：
- `agent_node`: 调用 LLM 决策下一步操作
- `tool_node`: 执行已批准的工具调用
- HITL (Human-in-the-Loop): 人类审批工作流

**关键代码**：
```python
# nodes/agent.py
def agent_node(state: ThreadState) -> dict:
    llm = get_kimi_llm()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "tool_calls": response.tool_calls}

# nodes/tool.py
def tool_node(state: ThreadState) -> dict:
    for tool_call in state["approved_tools"]:
        result = execute_tool(tool_call)
        results.append(ToolMessage(content=result))
    return {"messages": results}
```

**灵感来源**：
- LangGraph 官方工作流模式
- DeerFlow 的 HITL 设计

---

### Phase 2: Context & Memory

**目标**：解决长对话的上下文溢出问题，实现长期记忆能力。

**核心组件**：
1. `token_budget_node`: 监控 Token 使用量
2. `summarize_context_node`: 上下文压缩
3. `memory_inject_node`: 记忆注入

**架构图**：
```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ token_budget│───▶│memory_inject │───▶│   agent     │
└─────────────┘    └──────────────┘    └──────────────┘
       │                                        │
       ▼                                        ▼
┌─────────────┐                           ┌─────────────┐
│ summarize  │                           │    tool     │
│ _context   │                           └─────────────┘
└─────────────┘
```

---

## 核心技术实现

### 1. DeerFlow 风格上下文压缩（核心亮点）

**问题**：长对话会导致 Token 溢出，需要压缩历史消息。

**解决方案**：1:1 复刻 DeerFlow 的 SummarizationMiddleware

**核心设计**：
```python
# nodes/middleware.py
def summarize_context_node(state: ThreadState, config: dict) -> dict:
    """
    1. Trigger & Keep: 严格划分"被总结区"和"保留区"
    2. AI/Tool Pair Protection: 防止 tool_calls 和结果被拆散
    3. Context Replacement: 使用 RemoveMessage 彻底删除旧消息
    """
    # 保留最近 N 条
    to_keep = messages[-keep_recent:]
    to_summarize = messages[:-keep_recent]
    
    # AI/Tool Pair Protection
    while isinstance(to_keep[0], ToolMessage):
        to_keep = [to_summarize.pop()] + to_keep
    
    # 生成真实摘要（调用 LLM）
    summary = llm.invoke(summary_prompt)
    
    # Context Replacement
    delete_instructions = [RemoveMessage(id=m.id) for m in to_summarize]
    return {"messages": delete_instructions + [summary_msg] + to_keep}
```

**灵感来源**：[DeerFlow backend/docs/summarization.md](https://github.com/facebookresearch/deerflow)

---

### 2. 统一记忆系统（架构演进）

**问题**：之前有独立的"偏好提取"功能，造成功能重叠。

**解决方案**：统一为记忆系统，Agent 主动决定保存什么。

**设计原则**：
- 记忆层只负责存储/检索，不做提取逻辑
- Agent 通过 `save_memory` 工具主动保存记忆

```python
# nodes/agent.py - 注册 save_memory 工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "将重要信息保存到长期记忆..."
        }
    }
]

# nodes/tool.py - 执行记忆保存
def _execute_save_memory(args: dict) -> str:
    store = get_memory_store()
    store.add(
        content=args["content"],
        memory_type=args.get("memory_type", "general"),
        confidence=args.get("confidence", 0.9)
    )
    return f"[成功] 记忆已保存"
```

---

### 3. Kimi k2.5 reasoning_content 补丁（底层拦截）

**问题**：Kimi k2.5 是思考模型，输出包含 `reasoning_content`，但 LangChain 在序列化时会静默丢弃，导致多轮调用时报错。

**解决方案**：参考 DeerFlow 的 `patched_openai.py`，编写底层拦截器。

```python
# engine/patched_kimi.py
class PatchedKimiChatOpenAI(ChatOpenAI):
    """解决 reasoning_content 丢失问题"""
    
    def _get_request_payload(self, input_, **kwargs) -> dict:
        # 1. 获取原始消息（包含 reasoning_content）
        original_messages = self._convert_input(input_).to_messages()
        
        # 2. 调用父类方法（此时 reasoning_content 已被阉割）
        payload = super()._get_request_payload(input_, **kwargs)
        
        # 3. 把被丢弃的 reasoning_content 塞回去
        for payload_msg, orig_msg in zip(payload_messages, original_messages):
            if payload_msg.get("role") == "assistant":
                reasoning = orig_msg.additional_kwargs.get("reasoning_content")
                if reasoning is not None:
                    payload_msg["reasoning_content"] = reasoning
        
        return payload
```

**灵感来源**：Kimi 官方文档 + DeerFlow patched_openai.py

---

### 4. 多维动态触发配置

**问题**：单一触发条件不够灵活。

**解决方案**：支持 OR 逻辑的多维触发。

```python
# config/graph.yaml
token_budget:
  trigger:
    or_logic: true  # 任一条件满足即触发
    conditions:
      - type: "token_count"
        threshold: 4000
      - type: "message_count"
        threshold: 20
      - type: "context_fraction"
        threshold: 0.8  # 达到最大上下文 80%

# nodes/middleware.py
def check_trigger_conditions(messages, config, encoding) -> tuple[bool, str]:
    triggered_reasons = []
    for condition in config["conditions"]:
        if condition["type"] == "token_count":
            if current > threshold:
                triggered_reasons.append(...)
        # ... 其他条件类型
    
    return bool(triggered_reasons), "; ".join(triggered_reasons)
```

---

### 5. Token 精确计算

**问题**：简单字符估算不准确。

**解决方案**：考虑消息结构开销。

```python
def calculate_messages_tokens(messages, encoding) -> int:
    """精确计算考虑：
    - 每条消息基础开销 (+4)
    - Role token 开销
    - Content token
    - Tool call 额外开销
    - 消息列表格式开销 (+3)
    """
    total_tokens = 0
    for msg in messages:
        total_tokens += 4  # 基础开销
        total_tokens += len(encoding.encode(content))
        total_tokens += role_tokens.get(role, 3)
        # Tool call 开销...
    return total_tokens
```

---

## 文件结构

```
Mini_Harness/
├── main.py                    # 主入口
├── engine/
│   ├── state.py              # ThreadState 定义
│   ├── builder.py            # LangGraph 构建
│   ├── loader.py             # 配置加载
│   └── patched_kimi.py       # Kimi k2.5 补丁 ⭐
├── nodes/
│   ├── agent.py              # Agent 节点
│   ├── tool.py               # 工具执行节点
│   └── middleware.py         # Phase 2 中间件 ⭐
├── memory/
│   └── store.py              # 记忆存储
├── config/
│   └── graph.yaml            # 图配置
└── docs/designs/
    └── 2025-03-31-*-design.md  # 设计文档
```

---

## 配置示例

```yaml
# config/graph.yaml
graph:
  name: "mini_harness_v2"
  
context_engineering:
  token_budget:
    trigger:
      or_logic: true
      conditions:
        - type: "token_count"
          threshold: 4000
        - type: "message_count"
          threshold: 20
  
  summarization:
    keep_recent: 4
    model: "moonshot-v1-8k"
    inject_as_human: true  # 摘要伪装为 HumanMessage
  
  memory:
    confidence_threshold: 0.8
    max_memories: 3
```

---

## 环境变量

```bash
# .env
KIMI_API_KEY=sk-xxx
KIMI_MODEL=moonshot-v1-8k
KIMI_BASE_URL=https://api.moonshot.cn/v1
SUMMARY_MODEL=moonshot-v1-8k
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| 使用 PatchedKimiChatOpenAI | 解决 k2.5 reasoning_content 丢失 |
| 统一记忆系统 | 避免偏好提取重复逻辑 |
| 摘要伪装为 HumanMessage | 防止模型对系统提示词敏感 |
| 多维触发 OR 逻辑 | 更灵活的触发条件 |
| Token 精确计算 | 准确的预算控制 |

---

## 依赖

```txt
langgraph
langchain-openai
langchain-core
tiktoken  # 可选，用于精确 Token 计算
pyyaml
```

---

## 如何扩展

### 添加新工具

1. 在 `agent.py` 的 `tools` 列表中添加定义
2. 在 `tool.py` 的 `tool_node` 中添加执行逻辑

### 添加新的触发条件

1. 在 `config/graph.yaml` 的 `conditions` 中添加
2. 在 `check_trigger_conditions` 中添加处理逻辑

### 添加新的记忆类型

1. 在 `save_memory` 工具的 `memory_type` enum 中添加
2. 在 `memory/store.py` 中无需修改（已支持任意类型）

---

## 参考资料

1. [DeerFlow](https://github.com/facebookresearch/deerflow) - 核心架构参考
2. [LangGraph 文档](https://python.langchain.com/docs/langgraph) - 工作流框架
3. [Kimi API 文档](https://platform.moonshot.cn) - 模型 API

---

## 总结

本项目展示了以下核心能力：
1. **LangGraph 工作流构建**：完整的 Agent + Tool + HITL 流程
2. **工业级上下文管理**：DeerFlow 风格的触发/保留/压缩机制
3. **底层 API 适配**：针对特定模型的补丁开发
4. **配置驱动设计**：灵活的多维触发配置

所有实现均参考业界最佳实践（DeerFlow），兼顾工程简洁性和功能完整性。
