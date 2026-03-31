# DeerFlow 摘要功能升级设计文档

## 背景

当前 Mini Harness 的 `summarize_context_node` 已经实现了 DeerFlow 的核心算法逻辑：
- ✅ Trigger & Keep 机制
- ✅ AI/Tool Pair Protection
- ✅ Context Replacement (RemoveMessage)

但与工业级 DeerFlow 实现仍有 4 个关键差距，本文档解决**区别一**：实现真正的摘要生成。

## 当前问题

```python
# 当前代码 - 硬编码模拟文本
except Exception as e:
    summary_text = f"[之前对话包含 {len(to_summarize)} 条消息]"
```

无论删除什么对话，摘要内容都是固定的，无法体现对话实际内容。

## 目标

将 `to_summarize` 区域的消息真实发送给轻量级模型，生成有意义的对话摘要。

---

## 设计方案

### 1. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   摘要生成工作流                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   to_summarize (N条消息)                                    │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────┐     ┌─────────────┐                      │
│   │ 构建摘要    │────▶│ 调用轻量级   │                      │
│   │ 提示词      │     │ LLM (gpt-4o │                      │
│   └─────────────┘     │ -mini 等)   │                      │
│          │            └─────────────┘                      │
│          │                   │                              │
│          ▼                   ▼                              │
│   ┌─────────────────────────────────────────┐              │
│   │  summary_msg (真实摘要内容)              │              │
│   └─────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 提示词设计

```python
SUMMARY_PROMPT_TEMPLATE = """你是一个对话摘要助手。请阅读以下对话历史，用 50-100 字总结核心内容。

对话历史：
{conversation_history}

要求：
1. 保留关键信息（用户意图、已完成操作、重要结论）
2. 忽略重复性客套话
3. 输出格式：纯文本摘要，不要任何格式标记

摘要："""
```

### 3. 模型选择策略

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 生产环境 | gpt-4o-mini | 成本低、速度快 |
| 国内环境 | moonshot-v1-8k | 便宜、本土化 |
| 测试环境 | gpt-4o-mini | 同生产 |

配置优先级：
1. 环境变量 `SUMMARY_MODEL`（用户自定义）
2. 配置文件 `summary.model`
3. 默认值 `gpt-4o-mini`

### 4. 消息格式转换

```python
def build_conversation_for_summary(messages: list) -> str:
    """将消息列表转换为摘要提示词格式"""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            role = "AI"
        elif isinstance(msg, ToolMessage):
            role = f"工具({msg.name})"
        elif isinstance(msg, SystemMessage):
            role = "系统"
        else:
            role = "未知"
        
        content = msg.content if hasattr(msg, "content") else str(msg)
        lines.append(f"{role}: {content}")
    
    return "\n".join(lines)
```

### 5. 摘要角色伪装（区别三）

DeerFlow 将摘要伪装为 `HumanMessage`，让模型认为是"人类在帮助回忆"。

```python
# 当前实现（SystemMessage）
summary_msg = SystemMessage(content=f"[对话摘要] {summary_text}")

# 升级实现（HumanMessage 伪装）
summary_msg = HumanMessage(
    content=f"Here is a summary of the conversation to date: {summary_text}"
)
```

### 6. 错误处理与降级

```python
def summarize_with_fallback(messages: list, config: dict) -> str:
    """带降级方案的摘要生成"""
    
    # 尝试调用 LLM
    try:
        llm = get_summary_llm(config)
        prompt = build_summary_prompt(messages)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.warning(f"摘要生成失败: {e}，使用降级方案")
        
        # 降级方案 1: 简单统计
        return f"[之前对话包含 {len(messages)} 条消息]"
```

---

## 配置设计

### graph.yaml 扩展

```yaml
summarization:
  enabled: true
  keep_recent: 4  # 保留最近 N 条
  
  # 区别二：多维动态触发
  trigger:
    or_logic: true  # 任一条件满足即触发
    
    conditions:
      - type: token_count
        threshold: 4000
      - type: message_count
        threshold: 20
      - type: context_fraction
        threshold: 0.8  # 达到最大上下文 80%
  
  # 区别一：真正的摘要生成
  model:
    provider: "openai"  # 或 "moonshot"
    name: "gpt-4o-mini"
    temperature: 0.3
    max_tokens: 200
  
  # 区别三：摘要伪装
  inject_as_human: true
```

### 环境变量

```bash
# 摘要模型配置
SUMMARY_MODEL=gpt-4o-mini
SUMMARY_API_KEY=sk-xxx
SUMMARY_BASE_URL=https://api.openai.com/v1
```

---

## Token 计算优化（区别四）

### 当前实现问题

```python
# 过于简单的 Token 计算
full_text = "\n".join([str(m.content) for m in messages])
current_tokens = len(encoding.encode(full_text))
```

### 优化方案

```python
def calculate_messages_tokens(messages: list, encoding) -> int:
    """精确计算消息列表的 Token 消耗"""
    
    total_tokens = 0
    
    for msg in messages:
        # 基础：每条消息都有 role 标记的开销
        total_tokens += 4  # 每条消息的 overhead
        
        # Role 名称
        if isinstance(msg, HumanMessage):
            total_tokens += len(encoding.encode("user"))
        elif isinstance(msg, AIMessage):
            total_tokens += len(encoding.encode("assistant"))
        elif isinstance(msg, ToolMessage):
            total_tokens += len(encoding.encode("tool"))
        
        # Content
        content = msg.content if hasattr(msg, "content") else ""
        total_tokens += len(encoding.encode(str(content)))
        
        # Tool call 特殊处理
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                total_tokens += len(encoding.encode(str(tool_call)))
    
    # 消息列表本身的格式开销
    total_tokens += 3  # [ 和 ] 以及逗号
    
    return total_tokens
```

---

## 实现计划

### Phase 1: 核心摘要生成

1. 添加 `get_summary_llm()` 函数
2. 修改 `summarize_context_node` 调用真实 LLM
3. 添加提示词模板
4. 实现错误降级

### Phase 2: 配置驱动

1. 更新 `graph.yaml` 配置结构
2. 实现多维触发逻辑（OR）
3. 添加环境变量支持

### Phase 3: 细节优化

1. 实现 HumanMessage 伪装
2. 优化 Token 计算
3. 添加日志和监控

---

## 预期效果

| 指标 | 当前 | 升级后 |
|------|------|--------|
| 摘要内容 | 固定文本 | 真实对话摘要 |
| 触发条件 | 单一阈值 | 多维动态 OR |
| Token 计算 | 粗略估算 | 精确计算 |
| 摘要注入 | SystemMessage | HumanMessage（伪装） |

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 摘要 LLM 调用失败 | 流程中断 | 降级到简单文本 |
| 摘要超时 | 用户等待久 | 添加 5s 超时 |
| 成本增加 | 费用上升 | 使用 mini 模型 + 缓存 |

---

## 验收标准

1. ✅ 当 `to_summarize` 有内容时，生成的摘要反映真实对话
2. ✅ 摘要 LLM 调用失败时，降级为简单文本
3. ✅ 可通过配置切换模型
4. ✅ 摘要正确伪装为 HumanMessage
5. ✅ Token 计算考虑消息结构开销
