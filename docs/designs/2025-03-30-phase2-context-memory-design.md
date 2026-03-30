# Phase 2: Context & Memory 设计文档

## 概述

阶段二在阶段一（Core Graph & HITL）基础上，添加三大核心能力：
1. **Token 预算控制** - 防止上下文溢出
2. **上下文摘要压缩** - 智能压缩历史消息
3. **长期记忆注入** - 基于置信度的记忆检索

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  START  │───▶│ token_budget │───▶│ memory_inject│       │
│  └─────────┘    └──────────────┘    └──────────────┘       │
│                                              │              │
│                                              ▼              │
│  ┌─────────┐    ┌─────────┐    ┌────────┐  ┌──────┐       │
│  │   END   │◀───│  agent  │◀───│  tool  │◀─│agent │       │
│  └─────────┘    └─────────┘    └────────┘  └──────┘       │
│                      │                                      │
│                      ▼                                      │
│               ┌─────────────┐                               │
│               │ summarize   │ (条件触发)                     │
│               │ _context    │                               │
│               └─────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Token 预算控制节点 (token_budget)

**职责**：监控 token 使用量，触发上下文压缩

**触发条件**：
- 当前 token 数 > 预算阈值（默认 4000）
- 或消息数 > 阈值（默认 20 条）

**行为**：
- 如果未超限：直接传递给下一个节点
- 如果超限：标记需要压缩，传递给摘要节点

### 2. 上下文摘要压缩节点 (summarize_context)

**职责**：将历史消息压缩为摘要

**算法**：
1. 保留最近 N 条完整消息（默认 4 条）
2. 将更早的消息送入 LLM 生成摘要
3. 摘要格式："[Summary] 之前对话的主要内容是..."

**状态更新**：
```python
{
    "messages": [summary_message, recent_messages...],
    "summary_context": "压缩后的摘要内容",
    "token_count": new_count
}
```

### 3. 长期记忆注入节点 (memory_inject)

**职责**：从长期记忆库检索相关信息并注入上下文

**流程**：
1. 提取当前查询的关键词/向量
2. 检索记忆库（基于置信度阈值）
3. 将高置信度记忆格式化为系统消息注入

**记忆格式**：
```yaml
memory:
  content: "用户偏好使用中文交流"
  confidence: 0.95
  timestamp: "2025-03-30T10:00:00"
  source: "conversation_001"
```

## ThreadState 扩展

```python
class ThreadState(TypedDict):
    # Phase 1: Core fields
    messages: Annotated[list, add_messages]
    tool_calls: list
    approved_tools: list
    is_complete: bool
    
    # Phase 2: Context & Memory
    token_count: int              # 当前 token 计数
    token_budget: int             # token 预算阈值
    summary_context: str          # 压缩后的摘要
    memory_context: list          # 注入的记忆列表
    needs_summarization: bool     # 是否需要压缩标记
```

## 配置扩展

```yaml
# config/graph.yaml
graph:
  name: mini_harness_v2
  checkpointer: memory
  
  # Phase 2: Context & Memory 配置
  context_engineering:
    token_budget:
      enabled: true
      max_tokens: 4000
      max_messages: 20
    
    summarization:
      enabled: true
      keep_recent: 4           # 保留最近 N 条不压缩
      summary_model: "kimi-k2.5"  # 用于摘要的模型
    
    memory:
      enabled: true
      confidence_threshold: 0.8
      max_memories: 3          # 每次最多注入 N 条记忆
      storage: "json_file"     # 后期可换为 vector_db

nodes:
  - name: token_budget
    type: middleware
    description: "Monitor token usage and trigger summarization"
    
  - name: memory_inject
    type: middleware
    description: "Inject relevant memories into context"
    
  - name: summarize_context
    type: conditional
    description: "Compress old messages when budget exceeded"
    condition: "needs_summarization"
    
  - name: agent
    type: llm
    # ... Phase 1 配置
    
  - name: tool
    type: tool
    # ... Phase 1 配置

edges:
  - from: "__start__"
    to: "token_budget"
    
  - from: "token_budget"
    to: "memory_inject"
    
  - from: "memory_inject"
    to: "agent"
    
  - from: "agent"
    to: "summarize_context"
    condition: "needs_summarization"
    
  - from: "summarize_context"
    to: "agent"
    
  - from: "agent"
    to: "tool"
    condition: "has_tool_calls"
    
  - from: "tool"
    to: "agent"
    
  - from: "agent"
    to: "__end__"
    condition: "is_complete"
```

## 实现顺序

1. **更新 ThreadState** - 添加阶段二字段
2. **实现 token_budget 节点** - 预算监控
3. **实现 summarize_context 节点** - 摘要压缩
4. **实现 memory_inject 节点** - 记忆注入
5. **更新 builder.py** - 集成新节点到图
6. **更新配置** - 添加阶段二配置
7. **测试验证** - 长对话测试、记忆注入测试

## 关键设计决策

### 1. 摘要触发策略
- **方案A**：每次对话前检查（当前采用）
- **方案B**：异步后台压缩
- **方案C**：用户主动触发

选择方案A，简单可靠，易于调试。

### 2. 记忆存储
- **Phase 2**：JSON 文件存储（简单实现）
- **Phase 3+**：向量数据库（性能优化）

### 3. 置信度计算
```python
confidence = base_confidence * decay_factor ^ time_delta
```
- 基础置信度：人工确认 +0.2，多次出现 +0.1
- 时间衰减：每天衰减 5%

## 测试策略

### 测试 1: Token 预算触发
```python
# 模拟 30 条消息，验证是否触发压缩
messages = [HumanMessage(content=f"Message {i}") for i in range(30)]
# 预期：前 26 条被压缩，保留后 4 条
```

### 测试 2: 记忆注入
```python
# 先让系统记住"用户喜欢中文"
# 再开启新对话，验证记忆是否注入
```

### 测试 3: 端到端长对话
```python
# 进行 50 轮对话，验证：
# 1. 没有 token 溢出错误
# 2. 上下文连贯
# 3. 记忆正确应用
```

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 摘要丢失关键信息 | 高 | 保留最近 N 条完整消息 |
| 记忆注入过多 | 中 | 限制注入数量和置信度阈值 |
| Token 计算不准确 | 中 | 使用 tiktoken 精确计算 |
| 性能下降 | 低 | 异步处理摘要任务 |

## 下一阶段衔接

阶段三（Lead-Sub 多智能体）将复用阶段二的上下文管理能力：
- 每个子智能体有自己的上下文预算
- 父智能体可以访问子智能体的摘要
- 记忆库在父子智能体间共享
