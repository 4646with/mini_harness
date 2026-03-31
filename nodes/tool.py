"""Tool node that executes approved tool calls."""

from langchain_core.messages import ToolMessage

from engine.state import ThreadState
from memory.store import get_memory_store


def tool_node(state: ThreadState) -> dict:
    """Tool node that executes approved tool calls.
    
    In Phase 1, this simulates tool execution for HITL testing.
    Only executes tools that have been approved by human.
    
    Args:
        state: Current thread state
        
    Returns:
        State updates (messages, approved_tools cleared)
    """
    approved_tools = state.get("approved_tools", [])
    
    if not approved_tools:
        # No approved tools, return empty result
        return {
            "messages": [],
            "approved_tools": []
        }
    
    # Execute tools
    results = []
    for tool_call in approved_tools:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")
        
        # Execute based on tool name
        if tool_name == "save_memory":
            result_content = _execute_save_memory(tool_args)
        elif tool_name == "create_ppt":
            result_content = f"[Simulated] PPT created with topic: {tool_args.get('topic', 'unknown')}"
        else:
            # Default simulation
            result_content = f"[Simulated] Tool '{tool_name}' executed with args: {tool_args}"
        
        # Use ToolMessage for proper LangGraph protocol
        results.append(ToolMessage(
            content=result_content,
            tool_call_id=tool_call_id
        ))
    
    return {
        "messages": results,
        "approved_tools": []  # Clear approved tools after execution
    }


def _execute_save_memory(args: dict) -> str:
    """Execute save_memory tool.
    
    DeerFlow 风格：Agent 主动决定保存什么记忆
    
    Args:
        args: Tool arguments with content, memory_type, etc.
        
    Returns:
        Result message
    """
    store = get_memory_store()
    
    content = args.get("content", "")
    memory_type = args.get("memory_type", "general")
    confidence = args.get("confidence", 0.9)
    
    if not content:
        return "[Error] Memory content cannot be empty"
    
    # Save to memory store
    memory = store.add(
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        metadata={"source": "agent_tool_call"}
    )
    
    return f"[Success] Memory saved: {content[:50]}..."
