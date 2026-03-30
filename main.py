"""Main entry point for Mini Agent Harness with HITL."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from engine.loader import load_graph_config
from engine.builder import build_graph


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
        
        if current_state.values and current_state.values.get("messages"):
            # Continue existing conversation - just add user message
            graph.update_state(
                thread,
                {"messages": [HumanMessage(content=user_input)]}
            )
            # Stream with None to continue from current state
            stream_input = None
        else:
            # First message - create initial state with Phase 2 fields
            stream_input = {
                "messages": [{"role": "user", "content": user_input}],
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
        
        # Run graph
        print("\nAgent: ", end="", flush=True)
        
        for event in graph.stream(stream_input, thread, stream_mode="values"):
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    print(last_message.content)
        
        # Check if interrupted (waiting for HITL)
        current_state = graph.get_state(thread)
        
        if current_state.next:
            # Graph is interrupted before tool node
            print(f"\n[HITL] Tool execution requested")
            
            # Get pending tool calls from state
            state_values = current_state.values
            pending_tools = state_values.get("tool_calls", [])
            
            if pending_tools:
                print(f"Pending tools: {len(pending_tools)}")
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
                    print("[HITL] Approved, continuing...")
                    
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
                        {"approved_tools": approved_tools_converted, "tool_calls": []}
                    )
                    
                    # Resume execution
                    print("\nAgent: ", end="", flush=True)
                    ai_response_printed = False
                    for event in graph.stream(None, thread, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            # Only print AI messages with actual content (final response)
                            if hasattr(last_message, "type") and last_message.type == "ai":
                                if hasattr(last_message, "content") and last_message.content:
                                    print(last_message.content)
                                    ai_response_printed = True
                    
                    if not ai_response_printed:
                        print("(No response generated)")
                
                else:
                    # Denied
                    print("[HITL] Denied, skipping tool execution")
                    
                    # Clear tool calls and mark complete
                    graph.update_state(
                        thread,
                        {"approved_tools": [], "tool_calls": [], "is_complete": True}
                    )
                    
                    # Resume to get final response
                    print("\nAgent: ", end="", flush=True)
                    for event in graph.stream(None, thread, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            if hasattr(last_message, "type") and last_message.type == "ai":
                                if hasattr(last_message, "content") and last_message.content:
                                    print(last_message.content)


if __name__ == "__main__":
    main()
