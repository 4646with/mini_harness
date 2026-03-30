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
