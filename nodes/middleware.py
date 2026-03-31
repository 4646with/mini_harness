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
from langchain_openai import ChatOpenAI
from engine.state import ThreadState
from memory.store import get_memory_store


def token_budget_node(state: ThreadState, config: dict = None) -> dict:
    """监控 Token 使用量并判断是否需要摘要
    
    参数:
        state: 当前线程状态
        config: 节点配置，包含 max_tokens 和 max_messages
        
    返回:
        状态更新，包含 token 计数和摘要标志
    """
    # 默认配置
    max_tokens = config.get("max_tokens", 4000) if config else 4000
    max_messages = config.get("max_messages", 20) if config else 20
    
    messages = state.get("messages", [])
    
    # 简单 Token 估算（中英文混合约 4 字符 ≈ 1 token）
    total_chars = sum(
        len(str(msg.get("content", ""))) if isinstance(msg, dict) else len(str(msg.content))
        for msg in messages
    )
    estimated_tokens = total_chars // 4
    
    # 检查是否需要摘要
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
    
    参数:
        state: 当前线程状态
        config: 节点配置，包含 keep_recent 和 summary_model
        
    返回:
        状态更新，包含 RemoveMessage 指令和摘要
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
        # 降级方案: 简单字符估算
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
            
            # 创建摘要提示词
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
            # 降级方案: 简单描述
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
