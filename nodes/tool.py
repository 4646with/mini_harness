"""Tool node that executes approved tool calls."""

from langchain_core.messages import ToolMessage

from engine.state import ThreadState


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
    
    # Simulate tool execution
    results = []
    for tool_call in approved_tools:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")
        
        # Simulate execution (Phase 1)
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
