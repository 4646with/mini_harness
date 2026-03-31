"""Main entry point for Mini Agent Harness with HITL."""

import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from engine.loader import load_graph_config
from engine.builder import build_graph
from engine.stream_bridge import extract_standard_events
from memory.store import get_memory_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """Run the agent harness with HITL."""
    load_dotenv()
    
    if not os.getenv("KIMI_API_KEY"):
        logger.error("KIMI_API_KEY not set in .env file")
        logger.info("Please copy .env.example to .env and add your API key")
        return
    
    logger.info("Loading configuration...")
    config = load_graph_config("config/graph.yaml")
    logger.info(f"Graph: {config.name}")
    logger.info(f"Checkpointer: {config.checkpointer}")
    
    logger.info("Building graph...")
    graph = build_graph(config)
    
    logger.info("Build complete")
    
    print(f"\n{'='*50}")
    print("Mini Agent Harness - Phase 1")
    print(f"{'='*50}\n")
    
    thread = {"configurable": {"thread_id": "default"}}
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Check if there's existing state
        current_state = graph.get_state(thread)
        
        # Load memories
        memory_store = get_memory_store()
        memories = memory_store.get_all()
        
        # Build memory context (will be injected by agent node)
        memory_context = []
        if memories:
            memory_context = [
                {"role": "system", "content": "[系统记忆]\n" + "\n".join([
                    f"- {m['content']}" for m in memories[-3:]
                ])}
            ]
        
        if current_state.values and current_state.values.get("messages"):
            stream_input = {
                "messages": [{"role": "user", "content": user_input}]
            }
        else:
            stream_input = {
                "messages": [{"role": "user", "content": user_input}],
                "tool_calls": [],
                "approved_tools": [],
                "is_complete": False,
                "token_count": 0,
                "token_budget": 4000,
                "summary_context": "",
                "memory_context": memory_context,
                "needs_summarization": False
            }
        
        # Run graph with stream bridge
        print("\n🤖 Agent: ", end="", flush=True)
        
        raw_stream = graph.stream(stream_input, thread, stream_mode="updates")
        
        for event in extract_standard_events(raw_stream):
            if event["event_type"] == "text_reply":
                print(event["content"])
                
            elif event["event_type"] == "billing":
                usage = event["usage"]
                logger.info(f"💰 [计费] 输入: {usage['input_tokens']}, 输出: {usage['output_tokens']}")
        
        # Check if interrupted (waiting for HITL)
        current_state = graph.get_state(thread)
        
        if current_state.next:
            logger.warning("⏸️ HITL: Agent 申请调用工具，是否允许？")
            
            state_values = current_state.values
            pending_tools = state_values.get("tool_calls", [])
            
            if pending_tools:
                logger.info(f"待执行工具: {len(pending_tools)} 个")
                for i, tool in enumerate(pending_tools, 1):
                    if isinstance(tool, dict):
                        tool_name = tool.get("name", "unknown")
                        tool_args = tool.get("args", {})
                    else:
                        tool_name = getattr(tool, 'name', 'unknown')
                        tool_args = getattr(tool, 'args', {})
                    logger.info(f"  {i}. {tool_name}({tool_args})")
            
            approval = input("\nApprove execution? (y/n): ").strip().lower()
            
            if approval == "y":
                logger.info("✅ HITL: 授权通过，继续执行...")
                
                approved_tool_calls = state_values.get("tool_calls", [])
                graph.invoke(
                    None,
                    thread,
                    {"approved_tools": approved_tool_calls}
                )
                
                # Resume with stream bridge
                print("\n🤖 Agent: ", end="", flush=True)
                raw_stream = graph.stream(None, thread, stream_mode="updates")
                for event in extract_standard_events(raw_stream):
                    if event["event_type"] == "text_reply":
                        print(event["content"])
                    elif event["event_type"] == "billing":
                        usage = event["usage"]
                        logger.info(f"💰 [计费] 输入: {usage['input_tokens']}, 输出: {usage['output_tokens']}")
            else:
                logger.warning("🚫 HITL: 已拒绝，跳过工具执行")
                
                graph.invoke(
                    None,
                    thread,
                    {"approved_tools": [], "tool_calls": [], "is_complete": True}
                )
                
                print("\n🤖 Agent: ", end="", flush=True)
                raw_stream = graph.stream(None, thread, stream_mode="updates")
                for event in extract_standard_events(raw_stream):
                    if event["event_type"] == "text_reply":
                        print(event["content"])
                    elif event["event_type"] == "billing":
                        usage = event["usage"]
                        logger.info(f"💰 [计费] 输入: {usage['input_tokens']}, 输出: {usage['output_tokens']}")


if __name__ == "__main__":
    main()
