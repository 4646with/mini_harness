"""YAML configuration loader for graph definition."""

import os
import yaml
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from engine.patched_kimi import PatchedKimiChatOpenAI

load_dotenv()

LLM_PROVIDER_REGISTRY = {
    "openai": ChatOpenAI,
    "moonshot": PatchedKimiChatOpenAI,
}


def get_llm(config: dict = None) -> Any:
    config = config or {}
    
    provider = config.get("provider", "openai").lower()
    model_name = config.get("model", "moonshot-v1-32k-vision-preview")
    temperature = config.get("temperature", 0.7)
    
    llm_class = LLM_PROVIDER_REGISTRY.get(provider, ChatOpenAI)
    
    print(f"🤖 [Loader] 根据 provider '{provider}' 路由，使用类: {llm_class.__name__}")
    
    if provider == "moonshot":
        return llm_class(
            model=model_name,
            api_key=os.getenv("KIMI_API_KEY"),
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            temperature=temperature,
        )
    else:
        return llm_class(
            model=model_name,
            temperature=temperature,
        )


def get_summary_llm(config: dict = None) -> Any:
    config = config or {}
    
    provider = config.get("provider", "openai").lower()
    model_name = config.get("model", "moonshot-v1-8k")
    temperature = config.get("temperature", 0.3)
    max_tokens = config.get("max_tokens", 200)
    
    llm_class = LLM_PROVIDER_REGISTRY.get(provider, ChatOpenAI)
    
    print(f"🤖 [Loader] 摘要使用 provider '{provider}' 路由，使用类: {llm_class.__name__}")
    
    if provider == "moonshot":
        return llm_class(
            model=model_name,
            api_key=os.getenv("KIMI_API_KEY") or os.getenv("SUMMARY_API_KEY"),
            base_url=os.getenv("SUMMARY_BASE_URL") or os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return llm_class(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class GraphConfig:
    """Loaded graph configuration."""
    
    def __init__(self, config_dict: dict):
        self.name = config_dict["graph"]["name"]
        self.checkpointer = config_dict["graph"]["checkpointer"]
        self.nodes = config_dict["graph"]["nodes"]
        self.edges = config_dict["graph"]["edges"]
        self.context_config = config_dict["graph"].get("context_engineering", {})
        self.llm_config = config_dict.get("llm", {})
    
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
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration."""
        return self.llm_config


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
    
    if "graph" not in config_dict:
        raise ValueError("Config missing 'graph' root key")
    
    required_graph_keys = ["name", "checkpointer", "nodes", "edges"]
    for key in required_graph_keys:
        if key not in config_dict["graph"]:
            raise ValueError(f"Config missing 'graph.{key}'")
    
    return GraphConfig(config_dict)
