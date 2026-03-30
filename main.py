"""Main entry point for Mini Agent Harness with HITL."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from engine.loader import load_graph_config
from engine.builder import build_graph
from memory.store import get_memory_store


def main():
    """Run the agent harness with HITL."""
    # Load environment variables
    load_dotenv()
    
    # Verify API key
    if not os.getenv("KIMI_API_KEY"):
        print("Error: KIMI_API_KEY not set in .env file")
        print("Please copy .env.example to .env and add your API key")
        return
    
    # Load configuration
    print("Loading configuration...")
    config = load_graph_config("config/graph.yaml")
    print(f"Graph: {config.name}")
    print(f"Checkpointer: {config.checkpointer}")
    
    # Build graph
    print("Building graph...")
    graph = build_graph(config)
    
    # Initialize thread
    thread_id = "test_thread_001"
    thread = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'='*50}")
    print("Mini Agent Harness - Phase 1")
    print(f"{'='*50}\n")
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Check if there's existing state
        current_state = graph.get_state(thread)
        
        # Load memories and format as system message
        memory_store = get_memory_store()
        memories = memory_store.get_all()
        
        # Build memory context
        memory_messages = []
        if memories:
            memory_text = "[系统记忆]\n" + "\n".join([
                f"- {m['content']}" for m in memories[-3:]  # Last 3 memories
            ])
            memory_messages.append({"role": "system", "content": memory_text})
        
        if current_state.values and current_state.values.get("messages"):
            # Continue existing conversation - use dict format for consistency
            stream_input = {
                "messages": memory_messages + [{"role": "user", "content": user_input}]
            }
        else:
            # First message - create initial state with Phase 2 fields
            stream_input = {
                "messages": memory_messages + [{"role": "user", "content": user_input}],
                "tool_calls": [],
                "approved_tools": [],
                "is_complete": False,
                # Phase 2: Context & Memory initial values
                "token_count": 0,
                "token_budget": 4000,
                "summary_context": "",
                "memory_context": [],
                "needs_summarization": False
            }
        
        # Run graph with updates mode - only get incremental changes
        print("\n🤖 Agent: ", end="", flush=True)
        
        for event in graph.stream(stream_input, thread, stream_mode="updates"):
            for node_name, node_state in event.items():
                # Only care about agent node output
                if node_name == "agent":
                    messages = node_state.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        # Strict check: must be AI message with actual content
                        if hasattr(last_msg, "type") and last_msg.type == "ai":
                            if hasattr(last_msg, "content") and last_msg.content:
                                # Skip if this message has tool_calls (intermediate step)
                                if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                                    print(last_msg.content)
        
        # Check if interrupted (waiting for HITL)
        current_state = graph.get_state(thread)
        
        if current_state.next:
            # Graph is interrupted before tool node
            print(f"\n⏸️ [HITL] Agent 申请调用工具，是否允许？")
            
            # Get pending tool calls from state
            state_values = current_state.values
            pending_tools = state_values.get("tool_calls", [])
            
            if pending_tools:
                print(f"待执行工具: {len(pending_tools)} 个")
                for i, tool in enumerate(pending_tools, 1):
                    # Handle both dict and ToolCall object formats
                    if isinstance(tool, dict):
                        tool_name = tool.get("name", "unknown")
                        tool_args = tool.get("args", {})
                    else:
                        # ToolCall object
                        tool_name = getattr(tool, 'name', 'unknown')
                        tool_args = getattr(tool, 'args', {})
                    print(f"  {i}. {tool_name}({tool_args})")
                
                # Request approval
                approval = input("\nApprove execution? (y/n): ").strip().lower()
                
                if approval == "y":
                    # Mark tools as approved
                    print("[HITL] 授权通过，继续执行...")
                    
                    # Convert ToolCall objects to dicts for storage
                    approved_tools_converted = []
                    for tool in pending_tools:
                        if isinstance(tool, dict):
                            approved_tools_converted.append(tool)
                        else:
                            # Convert ToolCall object to dict
                            approved_tools_converted.append({
                                "id": getattr(tool, 'id', ''),
                                "name": getattr(tool, 'name', 'unknown'),
                                "args": getattr(tool, 'args', {})
                            })
                    
                    # Update state with approved tools
                    graph.update_state(
                        thread,
                        {"approved_tools": approved_tools_converted}
                    )
                    
                    # Resume execution
                    print("\n🤖 Agent: ", end="", flush=True)
                    ai_response_printed = False
                    
                    for event in graph.stream(None, thread, stream_mode="updates"):
                        for node_name, node_state in event.items():
                            if node_name == "agent":
                                messages = node_state.get("messages", [])
                                if messages:
                                    last_msg = messages[-1]
                                    if hasattr(last_msg, "type") and last_msg.type == "ai":
                                        if hasattr(last_msg, "content") and last_msg.content:
                                            # Skip if has tool_calls
                                            if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                                                print(last_msg.content)
                                                ai_response_printed = True
                    
                    if not ai_response_printed:
                        print("(No response generated)")
                
                else:
                    # Denied
                    print("🚫 [HITL] 已拒绝，跳过工具执行")
                    
                    # Clear tool calls and mark complete
                    graph.update_state(
                        thread,
                        {"approved_tools": [], "tool_calls": [], "is_complete": True}
                    )
                    
                    # Resume to get final response
                    print("\n🤖 Agent: ", end="", flush=True)
                    for event in graph.stream(None, thread, stream_mode="updates"):
                        for node_name, node_state in event.items():
                            if node_name == "agent":
                                messages = node_state.get("messages", [])
                                if messages:
                                    last_msg = messages[-1]
                                    if hasattr(last_msg, "type") and last_msg.type == "ai":
                                        if hasattr(last_msg, "content") and last_msg.content:
                                            print(last_msg.content)


if __name__ == "__main__":
    main()
