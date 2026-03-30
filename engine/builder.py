"""Graph builder that constructs LangGraph from YAML configuration."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from engine.state import ThreadState
from engine.loader import GraphConfig
from nodes import agent_node, tool_node


def has_tool_calls(state: ThreadState) -> str:
    """Condition: Check if agent returned tool calls."""
    tool_calls = state.get("tool_calls", [])
    return "tool" if len(tool_calls) > 0 else "end"


def is_complete(state: ThreadState) -> str:
    """Condition: Check if conversation is complete."""
    return END if state.get("is_complete", False) else "agent"


def build_graph(config: GraphConfig):
    """Build LangGraph from configuration.
    
    Phase 1: Basic agent-tool loop with HITL
    Phase 2 Extension: Add middleware_chain before agent_node
    Phase 3 Extension: Add Lead-Sub orchestration nodes
    
    Args:
        config: Loaded graph configuration
        
    Returns:
        Compiled graph with checkpointer
    """
    # Initialize graph with state schema
    workflow = StateGraph(ThreadState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)
    
    # Phase 2 Extension: Middleware chain
    # workflow.add_node("token_budget", token_budget_node)
    # workflow.add_node("memory_inject", memory_inject_node)
    
    # Phase 3 Extension: Lead-Sub orchestration
    # workflow.add_node("lead_plan", lead_plan_node)
    # workflow.add_node("sub_agent", sub_agent_node)
    # workflow.add_node("aggregate", aggregate_node)
    
    # Add edges
    workflow.set_entry_point("agent")
    
    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        has_tool_calls,
        {
            "tool": "tool",
            "end": END
        }
    )
    
    # Edge from tool back to agent
    workflow.add_edge("tool", "agent")
    
    # Configure checkpointer
    if config.checkpointer == "memory":
        checkpointer = MemorySaver()
    else:
        raise ValueError(f"Unsupported checkpointer: {config.checkpointer}")
    
    # Compile with interrupt_before for HITL
    # Interrupt before tool node to require human approval
    tool_node_config = config.get_node("tool")
    interrupt_before = ["tool"] if tool_node_config and tool_node_config.get("require_approval") else []
    
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before
    )
    
    return graph
