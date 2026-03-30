"""State definitions for the agent harness.

Extension interfaces for Phase 2 & 3:
- Phase 2: token_count, summary_context, memory_context
- Phase 3: task_list, sub_agent_states
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class ThreadState(TypedDict):
    """State for a single conversation thread.
    
    Phase 1 (Current): Core Graph & HITL
    Phase 2 (Extension): Context & Memory
    Phase 3 (Extension): Lead-Sub Orchestration
    
    Fields:
        messages: Conversation history with add_messages reducer
        tool_calls: List of pending tool calls from agent
        approved_tools: List of tool calls approved by human
        is_complete: Whether the conversation is finished
        
        # Phase 2 Extension Fields
        token_count: Current token count for budget control
        token_budget: Token budget threshold
        summary_context: Compressed summary of old messages
        memory_context: Injected facts from long-term memory
        needs_summarization: Flag to trigger context compression
        
        # Phase 3 Extension Fields (预留)
        # task_list: Sub-tasks for Lead-Sub orchestration
        # sub_agent_states: States of spawned sub-agents
    """
    # Phase 1: Core fields
    messages: Annotated[list, add_messages]
    tool_calls: list
    approved_tools: list
    is_complete: bool
    
    # Phase 2: Context & Memory fields
    token_count: int
    token_budget: int
    summary_context: str
    memory_context: list
    needs_summarization: bool
