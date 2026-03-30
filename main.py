"""Main entry point for Mini Agent Harness with HITL."""

import os
from dotenv import load_dotenv
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
        
        # Add user message to state
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "tool_calls": [],
            "approved_tools": [],
            "is_complete": False
        }
        
        # Run graph
        print("\nAgent: ", end="", flush=True)
        
        for event in graph.stream(initial_state, thread, stream_mode="values"):
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
                    tool_name = tool.get("name", "unknown")
                    tool_args = tool.get("args", {})
                    print(f"  {i}. {tool_name}({tool_args})")
                
                # Request approval
                approval = input("\nApprove execution? (y/n): ").strip().lower()
                
                if approval == "y":
                    # Mark tools as approved
                    print("[HITL] Approved, continuing...")
                    
                    # Update state with approved tools
                    graph.update_state(
                        thread,
                        {"approved_tools": pending_tools, "tool_calls": []}
                    )
                    
                    # Resume execution
                    for event in graph.stream(None, thread, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            if isinstance(last_message, dict) and "content" in last_message:
                                print(f"\nTool result: {last_message['content']}")
                            elif hasattr(last_message, "content") and last_message.content:
                                print(f"\nAgent: {last_message.content}")
                
                else:
                    # Denied
                    print("[HITL] Denied, skipping tool execution")
                    
                    # Clear tool calls and mark complete
                    graph.update_state(
                        thread,
                        {"approved_tools": [], "tool_calls": [], "is_complete": True}
                    )
                    
                    # Resume to get final response
                    for event in graph.stream(None, thread, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            if hasattr(last_message, "content") and last_message.content:
                                print(f"\nAgent: {last_message.content}")


if __name__ == "__main__":
    main()
