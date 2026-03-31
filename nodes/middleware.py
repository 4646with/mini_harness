"""阶段二中间件节点：Context & Memory

本模块包含在 Agent 处理前处理上下文的节点：
- token_budget: 监控 Token 使用量并触发摘要
- memory_inject: 注入相关记忆到上下文
- summarize_context: 当预算超限时压缩旧消息

参考 DeerFlow: backend/docs/summarization.md
"""

import os
import re
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, 
    RemoveMessage, ToolMessage
)
from engine.patched_kimi import get_summary_llm
from engine.state import ThreadState
from memory.store import get_memory_store


# 摘要提示词模板
SUMMARY_PROMPT_TEMPLATE = """你是一个对话摘要助手。请阅读以下对话历史，用 50-100 字总结核心内容。

对话历史：
{conversation_history}

要求：
1. 保留关键信息（用户意图、已完成操作、重要结论）
2. 忽略重复性客套话
3. 输出格式：纯文本摘要，不要任何格式标记

摘要："""


# ============================================================
# 区别四：Token 精确计算
# ============================================================
def calculate_messages_tokens(messages: list, encoding=None) -> int:
    """精确计算消息列表的 Token 消耗
    
    考虑因素：
    - 每条消息的基础开销（role 标记等）
    - 不同角色的 token 开销
    - Content 长度
    - Tool call 特殊处理
    
    参数:
        messages: 消息列表
        encoding: tiktoken 编码器，如果为 None 则使用降级方案
        
    返回:
        估算的 token 总数
    """
    if not messages:
        return 0
    
    # 如果有 tiktoken，使用精确计算
    if encoding is not None:
        total_tokens = 0
        
        for msg in messages:
            # 每条消息的基础开销
            total_tokens += 4
            
            # Content token
            content = ""
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
            elif hasattr(msg, "content"):
                content = str(msg.content)
            
            if content:
                total_tokens += len(encoding.encode(content))
            
            # Role token (根据消息类型)
            if isinstance(msg, dict):
                role = msg.get("role", "user")
            else:
                role = getattr(msg, "type", "user")
            
            role_tokens = {
                "system": len(encoding.encode("system")),
                "user": len(encoding.encode("user")),
                "assistant": len(encoding.encode("assistant")),
                "human": len(encoding.encode("user")),  # human = user
                "ai": len(encoding.encode("assistant")),  # ai = assistant
                "tool": len(encoding.encode("tool")),
            }
            total_tokens += role_tokens.get(role, 3)
            
            # Tool call 额外开销
            if isinstance(msg, dict) and msg.get("tool_calls"):
                total_tokens += len(encoding.encode(str(msg["tool_calls"])))
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                total_tokens += len(encoding.encode(str(msg.tool_calls)))
        
        # 消息列表格式开销
        total_tokens += 3
        
        return total_tokens
    
    # 降级方案：简单字符估算
    total_chars = sum(
        len(str(msg.get("content", ""))) if isinstance(msg, dict) else len(str(msg.content))
        for msg in messages
    )
    return total_chars // 4


# ============================================================
# 区别二：多维动态触发 (OR Logic)
# ============================================================
def check_trigger_conditions(messages: list, config: dict, encoding=None) -> tuple[bool, str]:
    """检查是否满足任意触发条件
    
    支持多维 OR 逻辑触发：
    - token_count: token 数量超过阈值
    - message_count: 消息数量超过阈值
    - context_fraction: 达到最大上下文的比例
    
    参数:
        messages: 消息列表
        config: 触发配置
        encoding: tiktoken 编码器
        
    返回:
        (是否触发, 触发原因)
    """
    # 获取触发条件配置
    conditions = config.get("conditions", []) if config else []
    or_logic = config.get("or_logic", True) if config else True
    
    # 如果没有配置条件，使用默认行为
    if not conditions:
        max_tokens = config.get("max_tokens", 4000) if config else 4000
        max_messages = config.get("max_messages", 20) if config else 20
        current_tokens = calculate_messages_tokens(messages, encoding)
        
        if current_tokens > max_tokens:
            return True, f"token_count ({current_tokens} > {max_tokens})"
        if len(messages) > max_messages:
            return True, f"message_count ({len(messages)} > {max_messages})"
        return False, ""
    
    # 多维触发检查
    triggered_reasons = []
    
    for condition in conditions:
        condition_type = condition.get("type")
        threshold = condition.get("threshold", 0)
        
        if condition_type == "token_count":
            current = calculate_messages_tokens(messages, encoding)
            if current > threshold:
                triggered_reasons.append(f"token_count ({current} > {threshold})")
                
        elif condition_type == "message_count":
            current = len(messages)
            if current > threshold:
                triggered_reasons.append(f"message_count ({current} > {threshold})")
                
        elif condition_type == "context_fraction":
            # context_fraction 需要知道最大上下文窗口
            max_context = config.get("max_context", 128000)  # 默认 Kimi 128k
            current = calculate_messages_tokens(messages, encoding)
            fraction = current / max_context if max_context > 0 else 0
            if fraction >= threshold:
                triggered_reasons.append(f"context_fraction ({fraction:.1%} >= {threshold:.1%})")
    
    # OR 逻辑：任一条件满足即触发
    if or_logic and triggered_reasons:
        return True, "; ".join(triggered_reasons)
    
    return False, ""


def get_token_encoder():
    """获取 tiktoken 编码器（带错误处理）"""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def token_budget_node(state: ThreadState, config: dict = None) -> dict:
    """监控 Token 使用量并判断是否需要摘要
    
    参数:
        state: 当前线程状态
        config: 节点配置，支持多维触发：
            - max_tokens: 最大 token 数
            - max_messages: 最大消息数
            - trigger.conditions: 触发条件列表
            - trigger.or_logic: OR 逻辑（默认 True）
            - max_context: 最大上下文窗口
        
    返回:
        状态更新，包含 token 计数和摘要标志
    """
    # 获取 tiktoken 编码器
    encoding = get_token_encoder()
    
    messages = state.get("messages", [])
    
    # 使用精确 Token 计算
    estimated_tokens = calculate_messages_tokens(messages, encoding)
    
    # 区别二：多维动态触发检查
    needs_summarization, trigger_reason = check_trigger_conditions(
        messages, config, encoding
    )
    
    # 获取预算阈值（用于显示）
    max_tokens = config.get("max_tokens", 4000) if config else 4000
    
    return {
        "token_count": estimated_tokens,
        "token_budget": max_tokens,
        "needs_summarization": needs_summarization,
        "trigger_reason": trigger_reason
    }


def summarize_context_node(state: ThreadState, config: dict = None) -> dict:
    """
    1:1 复刻 DeerFlow 的 Summarization Middleware
    参考：backend/docs/summarization.md
    
    核心设计：
    1. Trigger & Keep 机制：严格划分"被总结区"和"保留区"
    2. AI/Tool Pair Protection：防止 tool_calls 和 tool 返回结果被拆散
    3. Context Replacement：使用 RemoveMessage 彻底删除旧消息
    
    参数:
        state: 当前线程状态
        config: 节点配置，包含 keep_recent 和 summary_model
        
    返回:
        状态更新，包含 RemoveMessage 指令和摘要
    """
    # === 1. DeerFlow 配置映射 (Trigger & Keep) ===
    keep_recent = config.get("keep_recent", 4) if config else 4
    summary_model = config.get("model", "moonshot-v1-8k") if config else "moonshot-v1-8k"
    
    messages = state.get("messages", [])
    
    # 如果历史太短，直接放行
    if len(messages) <= keep_recent:
        return {"messages": []}
    
    # === 2. 使用精确 Token 计算（区别四）===
    encoding = get_token_encoder()
    current_tokens = calculate_messages_tokens(messages, encoding)
    
    # 从 token_budget_node 获取阈值，或默认 4000
    trigger_tokens = state.get("token_budget", 4000)
    
    print(f"📊 [Summarization] 当前 Tokens: {current_tokens} / 触发阈值: {trigger_tokens}")
    
    # === 3. 触发摘要压缩 ===
    if current_tokens > trigger_tokens or len(messages) > keep_recent + 2:
        print(f"⚠️ [Summarization] 触发 Context 压缩！保留最近 {keep_recent} 条消息。")
        
        # 划分被总结区和保留区
        to_summarize = messages[:-keep_recent]
        to_keep = messages[-keep_recent:]
        
        # ⚠️ DeerFlow 级防呆设计：AI/Tool Pair Protection
        # 如果 to_keep 的第一条是 ToolMessage，说明它的"上文"被切到 to_summarize 里了
        # 我们必须把切分线往前移，防止它们被拆散报错
        while to_keep and isinstance(to_keep[0], ToolMessage):
            to_keep = [to_summarize.pop()] + to_keep
            
        if not to_summarize:
            return {"messages": []}  # 极端情况：全部都是 tool 对，无法压缩
        
        # === 4. 生成摘要 (Summary Generation) ===
        # 使用 PatchedKimiChatOpenAI 避免 reasoning_content 丢失问题
        try:
            llm = get_summary_llm(config)
            
            # 构建对话历史
            conversation_lines = []
            for msg in to_summarize:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "type", "unknown")
                    content = getattr(msg, "content", "")
                
                # 角色映射
                role_map = {"human": "用户", "ai": "AI", "tool": "工具", "system": "系统"}
                role_label = role_map.get(role, role)
                
                if content:
                    conversation_lines.append(f"{role_label}: {content}")
            
            conversation_history = "\n".join(conversation_lines)
            summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
                conversation_history=conversation_history
            )
            
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = summary_response.content
            
        except Exception as e:
            # 降级方案: 简单描述
            summary_text = f"[之前对话包含 {len(to_summarize)} 条消息]"
        
        # === 4.1 摘要注入角色伪装 (区别三) ===
        # DeerFlow 风格：把摘要伪装成人类消息，让模型认为是"人类在帮助回忆"
        inject_as_human = config.get("inject_as_human", True) if config else True
        if inject_as_human:
            summary_msg = HumanMessage(
                content=f"Here is a summary of the conversation to date: {summary_text}"
            )
        else:
            summary_msg = SystemMessage(
                content=f"[对话摘要] {summary_text}"
            )
        
        # === 5. Context Replacement (LangGraph 专属替换法) ===
        # 使用 RemoveMessage 彻底删除旧消息
        delete_instructions = [
            RemoveMessage(id=m.id) for m in to_summarize 
            if hasattr(m, "id") and m.id
        ]
        
        print(f"✂️ [Summarization] 成功清理 {len(delete_instructions)} 条旧消息，注入摘要。")
        
        # 返回：删除指令 + 摘要消息 + 保留的消息
        return {
            "messages": delete_instructions + [summary_msg] + to_keep,
            "summary_context": summary_text,
            "needs_summarization": False,
            "token_count": len(str(to_keep)) // 4  # 重新估算
        }
    
    # 未触发阈值，无事发生
    return {"messages": []}


def memory_inject_node(state: ThreadState, config: dict = None) -> dict:
    """将相关记忆注入到上下文
    
    DeerFlow 设计原则：
    - 记忆层只负责存储和检索，不做任何提取逻辑
    - 让 Agent 自己决定什么值得记住（通过 tool_calls 调用记忆工具）
    - 保持接口简单：add/search/get
    
    参数:
        state: 当前线程状态
        config: 节点配置，包含 confidence_threshold 和 max_memories
        
    返回:
        状态更新，包含注入的记忆
    """
    # 默认配置
    confidence_threshold = config.get("confidence_threshold", 0.8) if config else 0.8
    max_memories = config.get("max_memories", 3) if config else 3
    
    # 获取记忆存储
    store = get_memory_store()
    
    # 获取最后一条用户消息用于上下文检索
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
        elif hasattr(msg, "type") and msg.type == "human":
            last_user_msg = msg.content
            break
    
    # 注意：DeerFlow 风格 - 不在 middleware 里硬编码提取逻辑
    # 偏好提取应该由 Agent 通过工具调用完成（如 save_memory 工具）
    # 这里只做简单的检索注入
    
    # 搜索相关记忆
    retrieved_memories = store.search(last_user_msg, threshold=confidence_threshold)
    
    # 限制数量
    filtered_memories = retrieved_memories[:max_memories]
    
    if not filtered_memories:
        return {"memory_context": []}
    
    # 将记忆格式化为 system 消息
    memory_text = "[相关记忆]\n" + "\n".join([
        f"- {m['content']}"
        for m in filtered_memories
    ])
    
    memory_message = SystemMessage(content=memory_text)
    
    # 将记忆插入到消息开头
    new_messages = [memory_message] + messages
    
    return {
        "messages": new_messages,
        "memory_context": filtered_memories
    }
