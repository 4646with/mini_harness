"""Middleware nodes for Phase 2: Context & Memory.

This module contains nodes that process context before it reaches the agent:
- token_budget: Monitor token usage and trigger summarization
- memory_inject: Inject relevant memories into context
- summarize_context: Compress old messages when budget exceeded
"""

import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from engine.state import ThreadState


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
    """Compress old messages into a summary.
    
    Strategy:
    1. Keep recent N messages intact
    2. Summarize older messages using LLM
    3. Replace old messages with summary
    
    Args:
        state: Current thread state
        config: Node configuration with keep_recent and summary_model
        
    Returns:
        State updates with compressed messages and summary context
    """
    # Default configuration
    keep_recent = config.get("keep_recent", 4) if config else 4
    summary_model = config.get("summary_model", "kimi-k2.5") if config else "kimi-k2.5"
    
    messages = state.get("messages", [])
    
    if len(messages) <= keep_recent:
        # Not enough messages to summarize
        return {
            "needs_summarization": False,
            "summary_context": state.get("summary_context", "")
        }
    
    # Split messages
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]
    
    # Generate summary using LLM
    try:
        llm = ChatOpenAI(
            model=summary_model,
            api_key=os.getenv("KIMI_API_KEY"),
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        )
        
        # Create summary prompt
        summary_prompt = "请用一句话总结以下对话的主要内容（不超过100字）：\n\n"
        for msg in old_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            summary_prompt += f"{role}: {content}\n"
        
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_text = summary_response.content
        
    except Exception as e:
        # Fallback: simple concatenation
        summary_text = f"[之前对话包含 {len(old_messages)} 条消息]"
    
    # Create summary message
    summary_message = SystemMessage(
        content=f"[对话摘要] {summary_text}"
    )
    
    # Rebuild message list: summary + recent messages
    new_messages = [summary_message] + recent_messages
    
    return {
        "messages": new_messages,
        "summary_context": summary_text,
        "needs_summarization": False,
        "token_count": len(str(new_messages)) // 4  # Re-estimate
    }


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
    
    # TODO: Implement actual memory retrieval from storage
    # For Phase 2, we use a simple in-memory approach
    
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
    
    # Simulate memory retrieval (replace with actual implementation)
    # In production, this would query a vector database
    retrieved_memories = []
    
    # Example: Check if we should inject language preference
    # This would normally come from a memory store
    if "中文" in last_user_msg or "Chinese" in last_user_msg:
        retrieved_memories.append({
            "content": "用户偏好使用中文交流",
            "confidence": 0.95,
            "type": "preference"
        })
    
    # Filter by confidence and limit count
    filtered_memories = [
        m for m in retrieved_memories 
        if m.get("confidence", 0) >= confidence_threshold
    ][:max_memories]
    
    if not filtered_memories:
        return {"memory_context": []}
    
    # Format memories as system message
    memory_text = "[相关记忆]\n" + "\n".join([
        f"- {m['content']} (置信度: {m['confidence']:.2f})"
        for m in filtered_memories
    ])
    
    memory_message = SystemMessage(content=memory_text)
    
    # Insert memory at the beginning of messages
    new_messages = [memory_message] + messages
    
    return {
        "messages": new_messages,
        "memory_context": filtered_memories
    }
