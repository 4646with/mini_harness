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
    llm = ChatOpenAI(
        model=os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        temperature=0.7,
    )
    
    # Bind tools for function calling
    # DeerFlow 风格：Agent 自己决定保存什么记忆
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
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "Save important information to long-term memory. Use this when the user mentions their name, preferences, or any fact they want you to remember for future conversations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember (e.g., 'User's name is John', 'User prefers Chinese language')"
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["profile", "preference", "fact", "general"],
                            "description": "Type of memory: profile (user info), preference (user likes), fact (general info), general (other)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level 0-1, higher means more certain"
                        }
                    },
                    "required": ["content"]
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
