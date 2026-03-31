"""Main entry point for Mini Agent Harness with HITL."""

import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from engine.loader import load_graph_config
from engine.builder import build_graph
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
        
        # Load memories and format as system message
        memory_store = get_memory_store()
        memories = memory_store.get_all()
        
        # Build memory context
        memory_messages = []
        if memories:
            memory_text = "[系统记忆]\n" + "\n".join([
                f"- {m['content']}" for m in memories[-3:]
            ])
            memory_messages.append({"role": "system", "content": memory_text})
        
        if current_state.values and current_state.values.get("messages"):
            # Continue existing conversation
            stream_input = {
                "messages": memory_messages + [{"role": "user", "content": user_input}]
            }
        else:
            # First message - create initial state
            stream_input = {
                "messages": memory_messages + [{"role": "user", "content": user_input}],
                "tool_calls": [],
                "approved_tools": [],
                "is_complete": False,
                "token_count": 0,
                "token_budget": 4000,
                "summary_context": "",
                "memory_context": [],
                "needs_summarization": False
            }
        
        # Run graph
        print("\n🤖 Agent: ", end="", flush=True)
        
        for event in graph.stream(stream_input, thread, stream_mode="updates"):
            for node_name, node_state in event.items():
                if node_name == "agent":
                    messages = node_state.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "type") and last_msg.type == "ai":
                            if hasattr(last_msg, "content") and last_msg.content:
                                if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                                    print(last_msg.content)
                                    
                                    if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                                        usage = last_msg.usage_metadata
                                        input_tokens = usage.get("input_tokens", 0)
                                        output_tokens = usage.get("output_tokens", 0)
                                        logger.info(f"💰 [计费] 输入: {input_tokens}, 输出: {output_tokens}")
        
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
                
                # Resume graph with approval
                approved_tool_calls = state_values.get("tool_calls", [])
                graph.invoke(
                    None,
                    thread,
                    {"approved_tools": approved_tool_calls}
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
                                        
                                        if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                                            usage = last_msg.usage_metadata
                                            input_tokens = usage.get("input_tokens", 0)
                                            output_tokens = usage.get("output_tokens", 0)
                                            logger.info(f"💰 [计费] 输入: {input_tokens}, 输出: {output_tokens}")
                                else:
                                    print("(No response generated)")
            else:
                logger.warning("🚫 HITL: 已拒绝，跳过工具执行")
                
                # Resume graph without approval
                graph.invoke(
                    None,
                    thread,
                    {"approved_tools": [], "tool_calls": [], "is_complete": True}
                )
                
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
                                        
                                        if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                                            usage = last_msg.usage_metadata
                                            input_tokens = usage.get("input_tokens", 0)
                                            output_tokens = usage.get("output_tokens", 0)
                                            logger.info(f"💰 [计费] 输入: {input_tokens}, 输出: {output_tokens}")


if __name__ == "__main__":
    main()
