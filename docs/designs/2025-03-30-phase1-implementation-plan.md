# Phase 1 Implementation Plan: Core Graph & HITL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configuration-driven LangGraph agent harness with HITL (Human-in-the-Loop) approval mechanism using Kimi API.

**Architecture:** YAML-configured state graph with separate engine (builder/loader), nodes (agent/tool), and CLI-based HITL using LangGraph's interrupt mechanism. **Includes extension interfaces for Phase 2 (Context & Memory) and Phase 3 (Skills & Gateway).**

**Tech Stack:** Python 3.10+, LangGraph 0.2+, LangChain-OpenAI, PyYAML, python-dotenv

**Extension Interfaces:**
- Phase 2: `middleware_chain` in builder, `memory_context` in state
- Phase 3: `skills` config, `task_list` in state, `tool_registry`

---

## File Structure

```
mini_harness/
├── config/
│   └── graph.yaml              # Graph configuration
├── engine/
│   ├── __init__.py             # Package init
│   ├── state.py                # ThreadState TypedDict
│   ├── loader.py               # YAML config loader
│   └── builder.py              # Graph builder from config
├── nodes/
│   ├── __init__.py             # Package init
│   ├── agent.py                # Agent node implementation
│   └── tool.py                 # Tool node implementation
├── main.py                     # Entry point with HITL
├── requirements.txt            # Dependencies
└── .env                        # Environment variables (Kimi API key)
```

---

## Task 1: Project Setup and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`

- [ ] **Step 1: Create requirements.txt**

```txt
langgraph>=0.2.0
langchain-openai>=0.2.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Create .env.example**

```bash
# Kimi API Configuration
KIMI_API_KEY=your_kimi_api_key_here
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=moonshot-v1-8k
```

- [ ] **Step 3: Create .gitignore**

```gitignore
# Environment
.env
.venv/
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "chore: add project dependencies and environment setup"
```

---

## Task 2: ThreadState Definition (with Extension Interfaces)

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/state.py`

- [ ] **Step 1: Create engine/__init__.py**

```python
"""Engine package for Mini Agent Harness."""

from .state import ThreadState

__all__ = ["ThreadState"]
```

- [ ] **Step 2: Create engine/state.py**

```python
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
        
        # Phase 2 Extension Fields (预留)
        # token_count: Current token count for budget control
        # summary_context: Compressed summary of old messages
        # memory_context: Injected facts from long-term memory
        
        # Phase 3 Extension Fields (预留)
        # task_list: Sub-tasks for Lead-Sub orchestration
        # sub_agent_states: States of spawned sub-agents
    """
    # Phase 1: Core fields
    messages: Annotated[list, add_messages]
    tool_calls: list
    approved_tools: list
    is_complete: bool
```

- [ ] **Step 3: Commit**

```bash
git add engine/
git commit -m "feat: add ThreadState definition with messages, tool_calls, approved_tools, is_complete"
```

---

## Task 3: YAML Configuration Loader

**Files:**
- Create: `engine/loader.py`
- Create: `config/graph.yaml`

- [ ] **Step 1: Create engine/loader.py**

```python
"""YAML configuration loader for graph definition."""

import yaml
from pathlib import Path
from typing import Any


class GraphConfig:
    """Loaded graph configuration."""
    
    def __init__(self, config_dict: dict):
        self.name = config_dict["graph"]["name"]
        self.checkpointer = config_dict["graph"]["checkpointer"]
        self.nodes = config_dict["graph"]["nodes"]
        self.edges = config_dict["graph"]["edges"]
    
    def get_node(self, name: str) -> dict | None:
        """Get node configuration by name."""
        for node in self.nodes:
            if node["name"] == name:
                return node
        return None
    
    def get_edges_from(self, node_name: str) -> list[dict]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e["from"] == node_name]


def load_graph_config(path: str | Path = "config/graph.yaml") -> GraphConfig:
    """Load graph configuration from YAML file.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        GraphConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # Basic validation
    if "graph" not in config_dict:
        raise ValueError("Config missing 'graph' root key")
    
    required_graph_keys = ["name", "checkpointer", "nodes", "edges"]
    for key in required_graph_keys:
        if key not in config_dict["graph"]:
            raise ValueError(f"Config missing 'graph.{key}'")
    
    return GraphConfig(config_dict)
```

- [ ] **Step 2: Create config/graph.yaml**

```yaml
graph:
  name: "mini_harness_v1"
  checkpointer: "memory"
  
  nodes:
    - name: "agent"
      type: "agent"
      model: "kimi"
      
    - name: "tool"
      type: "tool"
      require_approval: true
      # Phase 3 Extension: tools registry for dynamic loading
      # tools: ["web_search", "file_edit", "code_execute"]
  
  edges:
    - from: "agent"
      to: "tool"
      condition: "has_tool_calls"
      
    - from: "agent"
      to: "END"
      condition: "is_complete"
      
    - from: "tool"
      to: "agent"
  
  # Phase 2 Extension: Context configuration
  # context:
  #   token_limit: 8000
  #   compression_threshold: 6000
  #   summary_model: "moonshot-v1-8k"
  
  # Phase 3 Extension: Skills configuration
  # skills:
  #   - name: "researcher"
  #     path: "skills/researcher.md"
  #     auto_load: true
```

- [ ] **Step 3: Commit**

```bash
git add engine/loader.py config/graph.yaml
git commit -m "feat: add YAML config loader and initial graph.yaml"
```

---

## Task 4: Agent Node Implementation

**Files:**
- Create: `nodes/__init__.py`
- Create: `nodes/agent.py`

- [ ] **Step 1: Create nodes/__init__.py**

```python
"""Node implementations for the agent harness."""

from .agent import agent_node
from .tool import tool_node

__all__ = ["agent_node", "tool_node"]
```

- [ ] **Step 2: Create nodes/agent.py**

```python
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
    tools = [
        {
            "type": "function",
            "function": {
                "name": "simulate_tool",
                "description": "Simulate a tool execution for testing HITL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to simulate"
                        }
                    },
                    "required": ["action"]
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
```

- [ ] **Step 3: Commit**

```bash
git add nodes/
git commit -m "feat: add agent node with Kimi API integration and tool binding"
```

---

## Task 5: Tool Node Implementation

**Files:**
- Create: `nodes/tool.py`

- [ ] **Step 1: Create nodes/tool.py**

```python
"""Tool node that executes approved tool calls."""

from engine.state import ThreadState


def tool_node(state: ThreadState) -> dict:
    """Tool node that executes approved tool calls.
    
    In Phase 1, this simulates tool execution for HITL testing.
    Only executes tools that have been approved by human.
    
    Args:
        state: Current thread state
        
    Returns:
        State updates (messages, approved_tools cleared)
    """
    approved_tools = state.get("approved_tools", [])
    
    if not approved_tools:
        # No approved tools, return empty result
        return {
            "messages": [],
            "approved_tools": []
        }
    
    # Simulate tool execution
    results = []
    for tool_call in approved_tools:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        
        # Simulate execution (Phase 1)
        result_content = f"[Simulated] Tool '{tool_name}' executed with args: {tool_args}"
        
        results.append({
            "role": "tool",
            "content": result_content,
            "tool_call_id": tool_call.get("id", "")
        })
    
    return {
        "messages": results,
        "approved_tools": []  # Clear approved tools after execution
    }
```

- [ ] **Step 2: Commit**

```bash
git add nodes/tool.py
git commit -m "feat: add tool node with simulated execution for HITL testing"
```

---

## Task 6: Graph Builder

**Files:**
- Create: `engine/builder.py`

- [ ] **Step 1: Create engine/builder.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add engine/builder.py
git commit -m "feat: add graph builder with conditional edges and HITL interrupt"
```

---

## Task 7: Main Entry Point with HITL

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add main entry point with HITL CLI interaction loop"
```

---

## Task 8: Integration Test

**Files:**
- Create: `test_harness.py` (simple manual test script)

- [ ] **Step 1: Create test_harness.py**

```python
"""Simple test to verify harness components load correctly."""

import os
from dotenv import load_dotenv

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from engine.state import ThreadState
        print("✓ engine.state")
        
        from engine.loader import load_graph_config, GraphConfig
        print("✓ engine.loader")
        
        from engine.builder import build_graph
        print("✓ engine.builder")
        
        from nodes import agent_node, tool_node
        print("✓ nodes")
        
        print("\nAll imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting config loading...")
    
    try:
        from engine.loader import load_graph_config
        
        config = load_graph_config("config/graph.yaml")
        print(f"✓ Graph name: {config.name}")
        print(f"✓ Checkpointer: {config.checkpointer}")
        print(f"✓ Nodes: {len(config.nodes)}")
        print(f"✓ Edges: {len(config.edges)}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_graph_building():
    """Test graph building."""
    print("\nTesting graph building...")
    
    try:
        from engine.loader import load_graph_config
        from engine.builder import build_graph
        
        config = load_graph_config("config/graph.yaml")
        graph = build_graph(config)
        
        print(f"✓ Graph compiled successfully")
        print(f"✓ Nodes: {list(graph.nodes.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Graph building failed: {e}")
        return False


def main():
    """Run all tests."""
    load_dotenv()
    
    print("="*50)
    print("Mini Agent Harness - Component Tests")
    print("="*50)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Graph Building", test_graph_building()))
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run main.py")
    else:
        print("\n✗ Some tests failed. Please fix issues before running.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

- [ ] **Step 2: Run tests**

Run: `python test_harness.py`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add test_harness.py
git commit -m "test: add component integration tests"
```

---

## Task 9: Final Verification

- [ ] **Step 1: Verify file structure**

Run: `tree -I '__pycache__|*.pyc' -L 2`
Expected:
```
.
├── config/
│   └── graph.yaml
├── engine/
│   ├── __init__.py
│   ├── builder.py
│   ├── loader.py
│   └── state.py
├── nodes/
│   ├── __init__.py
│   ├── agent.py
│   └── tool.py
├── main.py
├── requirements.txt
├── test_harness.py
└── .env (user creates from .env.example)
```

- [ ] **Step 2: Run component tests**

Run: `python test_harness.py`
Expected: All 3 tests pass

- [ ] **Step 3: Create .env and test main (manual)**

Instructions for user:
1. Copy `.env.example` to `.env`
2. Add your Kimi API key to `.env`
3. Run: `python main.py`
4. Test conversation flow
5. When tool is requested, test both `y` (approve) and `n` (deny)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete phase 1 - core graph with HITL

- Configuration-driven graph builder from YAML
- Agent node with Kimi API integration
- Tool node with simulated execution
- CLI-based HITL with interrupt/resume
- Component integration tests"
```

---

## Success Criteria Verification

| Criteria | How to Verify |
|----------|---------------|
| Dynamic graph from YAML | Run `test_harness.py`, check "Config Loading" |
| Kimi API calls | Run `main.py`, verify responses |
| HITL suspend | Trigger tool call, verify pause |
| Approve (y) resume | Input `y`, verify tool executes |
| Deny (n) skip | Input `n`, verify skip message |

---

## Notes for Implementer

1. **API Costs**: Each test call consumes Kimi API quota. Use short prompts during testing.

2. **State Persistence**: Phase 1 uses `MemorySaver`. State is lost when process exits. Phase 2 will add SQLite persistence.

3. **Error Handling**: Basic validation is in place. Edge cases (network errors, invalid API keys) will show stack traces.

4. **Extensibility**: The YAML config structure supports adding new node types and edges for Phase 2/3.
