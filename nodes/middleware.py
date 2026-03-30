"""Middleware nodes for Phase 2: Context & Memory.

This module contains nodes that process context before it reaches the agent:
- token_budget: Monitor token usage and trigger summarization
- memory_inject: Inject relevant memories into context
- summarize_context: Compress old messages when budget exceeded

DeerFlow Reference: backend/docs/summarization.md
"""

import os
import re
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, 
    RemoveMessage, ToolMessage
)
from langchain_openai import ChatOpenAI
from engine.state import ThreadState
from memory.store import get_memory_store


def token_budget_node(state: ThreadState, config: dict = None) -> dict:
    """Monitor token usage and determine if summarization is needed.
    
    Args:
        state: Current thread state
        config: Node configuration with max_tokens and max_messages
        
    Returns:
        State updates with token count and summarization flag
    """
    # Default configuration
    max_tokens = config.get("max_tokens", 4000) if config else 4000
    max_messages = config.get("max_messages", 20) if config else 20
    
    messages = state.get("messages", [])
    
    # Simple token estimation (4 chars ≈ 1 token for Chinese/English mix)
    total_chars = sum(
        len(str(msg.get("content", ""))) if isinstance(msg, dict) else len(str(msg.content))
        for msg in messages
    )
    estimated_tokens = total_chars // 4
    
    # Check if summarization is needed
    needs_summarization = (
        estimated_tokens > max_tokens or 
        len(messages) > max_messages
    )
    
    return {
        "token_count": estimated_tokens,
        "token_budget": max_tokens,
        "needs_summarization": needs_summarization
    }


def summarize_context_node(state: ThreadState, config: dict = None) -> dict:
    """
    1:1 复刻 DeerFlow 的 Summarization Middleware
    参考：backend/docs/summarization.md
    
    核心设计：
    1. Trigger & Keep 机制：严格划分"被总结区"和"保留区"
    2. AI/Tool Pair Protection：防止 tool_calls 和 tool 返回结果被拆散
    3. Context Replacement：使用 RemoveMessage 彻底删除旧消息
    
    Args:
        state: Current thread state
        config: Node configuration with keep_recent and summary_model
        
    Returns:
        State updates with RemoveMessage instructions and summary
    """
    # === 1. DeerFlow 配置映射 (Trigger & Keep) ===
    keep_recent = config.get("keep_recent", 4) if config else 4
    summary_model = config.get("summary_model", "kimi-k2.5") if config else "kimi-k2.5"
    
    messages = state.get("messages", [])
    
    # 如果历史太短，直接放行
    if len(messages) <= keep_recent:
        return {"messages": []}
    
    # === 2. Token 精准计算 ===
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        full_text = "\n".join([
            str(m.content) for m in messages 
            if hasattr(m, "content") and m.content
        ])
        current_tokens = len(encoding.encode(full_text))
    except ImportError:
        # Fallback: 简单字符估算
        total_chars = sum(
            len(str(m.content)) for m in messages if hasattr(m, "content")
        )
        current_tokens = total_chars // 4
    
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
        try:
            llm = ChatOpenAI(
                model=summary_model,
                api_key=os.getenv("KIMI_API_KEY"),
                base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            )
            
            # Create summary prompt
            summary_prompt = "请用一句话总结以下对话的主要内容（不超过100字）：\n\n"
            for msg in to_summarize:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "type", "unknown")
                    content = getattr(msg, "content", "")
                if content:
                    summary_prompt += f"{role}: {content}\n"
            
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = summary_response.content
            
        except Exception as e:
            # Fallback: simple description
            summary_text = f"[之前对话包含 {len(to_summarize)} 条消息]"
        
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
            "token_count": len(str(to_keep)) // 4  # Re-estimate
        }
    
    # 未触发阈值，无事发生
    return {"messages": []}


def memory_inject_node(state: ThreadState, config: dict = None) -> dict:
    """Inject relevant memories into the context.
    
    Args:
        state: Current thread state
        config: Node configuration with confidence_threshold and max_memories
        
    Returns:
        State updates with injected memories
    """
    # Default configuration
    confidence_threshold = config.get("confidence_threshold", 0.8) if config else 0.8
    max_memories = config.get("max_memories", 3) if config else 3
    
    # Get memory store
    store = get_memory_store()
    
    # Get the last user message for context
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
        elif hasattr(msg, "type") and msg.type == "human":
            last_user_msg = msg.content
            break
    
    # Extract and save new memories from current message
    # Check for language preference
    if any(kw in last_user_msg for kw in ["中文", "Chinese", "用中文"]):
        # Check if we already have this memory
        existing = store.search("中文")
        if not existing:
            store.add(
                content="用户偏好使用中文交流",
                memory_type="preference",
                confidence=0.95,
                metadata={"trigger": "user_explicit_request"}
            )
    
    # Check for name preference (e.g., "以后叫我XXX", "我的名字是XXX")
    name_patterns = [
        r"以后叫我(.+?)(?:，|。|\s|$)",
        r"我的名字是(.+?)(?:，|。|\s|$)",
        r"我是(.+?)(?:，|。|\s|$)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, last_user_msg)
        if match:
            name = match.group(1).strip()
            if name and len(name) <= 10:  # Reasonable name length
                existing = store.search("名字")
                if not existing:
                    store.add(
                        content=f"用户的名字是{name}",
                        memory_type="profile",
                        confidence=0.9,
                        metadata={"trigger": "user_explicit_request", "name": name}
                    )
            break
    
    # Search for relevant memories
    retrieved_memories = store.search(last_user_msg, threshold=confidence_threshold)
    
    # Limit count
    filtered_memories = retrieved_memories[:max_memories]
    
    if not filtered_memories:
        return {"memory_context": []}
    
    # Format memories as system message
    memory_text = "[相关记忆]\n" + "\n".join([
        f"- {m['content']}"
        for m in filtered_memories
    ])
    
    memory_message = SystemMessage(content=memory_text)
    
    # Insert memory at the beginning of messages
    new_messages = [memory_message] + messages
    
    return {
        "messages": new_messages,
        "memory_context": filtered_memories
    }
