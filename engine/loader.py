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
        # Phase 2: Context & Memory configuration
        self.context_config = config_dict["graph"].get("context_engineering", {})
    
    def get_node(self, name: str) -> dict | None:
        """Get node configuration by name."""
        for node in self.nodes:
            if node["name"] == name:
                return node
        return None
    
    def get_edges_from(self, node_name: str) -> list[dict]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e["from"] == node_name]
    
    def get_context_config(self, section: str) -> dict:
        """Get Phase 2 context engineering configuration.
        
        Args:
            section: Configuration section (token_budget, summarization, memory)
            
        Returns:
            Configuration dict for the section, or empty dict if not found
        """
        return self.context_config.get(section, {})


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
