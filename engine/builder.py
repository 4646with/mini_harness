"""Graph builder that constructs LangGraph from YAML configuration."""

import os
import sqlite3
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from engine.state import ThreadState
from engine.loader import GraphConfig
from nodes import (
    agent_node,
    tool_node,
    token_budget_node,
    summarize_context_node,
    memory_inject_node,
)


def has_tool_calls(state: ThreadState) -> str:
    """Condition: Check if agent returned tool calls."""
    tool_calls = state.get("tool_calls", [])
    return "tool" if len(tool_calls) > 0 else "end"


def is_complete(state: ThreadState) -> str:
    """Condition: Check if conversation is complete."""
    return END if state.get("is_complete", False) else "agent"


def needs_summarization(state: ThreadState) -> str:
    """Condition: Check if context needs to be summarized."""
    return "summarize" if state.get("needs_summarization", False) else "agent"


def build_graph(config: GraphConfig):
    """Build LangGraph from configuration.
    
    Phase 1: Basic agent-tool loop with HITL
    Phase 2: Add middleware chain (token_budget, memory_inject, summarize_context)
    Phase 3 Extension: Add Lead-Sub orchestration nodes
    
    Args:
        config: Loaded graph configuration
        
    Returns:
        Compiled graph with checkpointer
    """
    # Initialize graph with state schema
    workflow = StateGraph(ThreadState)
    
    # Get context engineering config
    token_budget_config = config.get_context_config("token_budget")
    summarization_config = config.get_context_config("summarization")
    
    # Merge LLM config into summarization config for factory pattern
    llm_config = config.get_llm_config()
    if llm_config:
        summarization_config = {**llm_config, **summarization_config}
    
    # Add nodes
    workflow.add_node("agent", partial(agent_node, config=llm_config))
    workflow.add_node("tool", tool_node)
    
    # Phase 2: Middleware chain nodes with config
    workflow.add_node("token_budget", partial(token_budget_node, config=token_budget_config))
    workflow.add_node("memory_inject", partial(memory_inject_node, config={}))
    workflow.add_node("summarize_context", partial(summarize_context_node, config=summarization_config))
    
    # Phase 3 Extension: Lead-Sub orchestration
    # workflow.add_node("lead_plan", lead_plan_node)
    # workflow.add_node("sub_agent", sub_agent_node)
    # workflow.add_node("aggregate", aggregate_node)
    
    # Add edges - Phase 2 middleware chain
    workflow.set_entry_point("token_budget")
    
    # Token budget -> Memory inject
    workflow.add_edge("token_budget", "memory_inject")
    
    # Memory inject -> Summarize (conditional) or Agent
    workflow.add_conditional_edges(
        "memory_inject",
        needs_summarization,
        {
            "summarize": "summarize_context",
            "agent": "agent"
        }
    )
    
    # Summarize -> Agent
    workflow.add_edge("summarize_context", "agent")
    
    # Agent -> Tool (conditional) or END
    workflow.add_conditional_edges(
        "agent",
        has_tool_calls,
        {
            "tool": "tool",
            "end": END
        }
    )
    
    # Tool back to agent (for next turn)
    workflow.add_edge("tool", "agent")
    
    # Configure checkpointer
    if config.checkpointer == "memory":
        checkpointer = MemorySaver()
    elif config.checkpointer == "sqlite":
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
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
