"""Agent node that calls LLM to decide next action."""

import os
from langchain_openai import ChatOpenAI
from engine.state import ThreadState


def agent_node(state: ThreadState) -> dict:
    """Agent node that processes messages and decides next action.
    
    Args:
        state: Current thread state
        
    Returns:
        State updates (messages, tool_calls, is_complete)
    """
    # Initialize LLM client (Kimi API compatible with OpenAI format)
    # Note: Some models don't support temperature parameter
    llm = ChatOpenAI(
        model=os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
    )
    
    # Bind tools for function calling
    # 使用更具体的工具描述，引导 Agent 在需要时调用工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_ppt",
                "description": "Create a PowerPoint presentation. Use this tool when the user asks to create a presentation, slides, or PPT about any topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic of the presentation"
                        },
                        "slides_count": {
                            "type": "integer",
                            "description": "Number of slides to create"
                        }
                    },
                    "required": ["topic"]
                }
            }
        }
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    # Call LLM
    response = llm_with_tools.invoke(state["messages"])
    
    # Check if tool calls are present
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = response.tool_calls
    
    # Determine if conversation is complete
    # Complete if: no tool calls AND content is present
    is_complete = len(tool_calls) == 0 and response.content is not None
    
    return {
        "messages": [response],
        "tool_calls": tool_calls,
        "is_complete": is_complete
    }
