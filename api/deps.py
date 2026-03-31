"""FastAPI 依赖注入"""
from functools import lru_cache
from engine.loader import load_graph_config, GraphConfig
from engine.builder import build_graph


@lru_cache
def get_config() -> GraphConfig:
    return load_graph_config("config/graph.yaml")


@lru_cache
def get_graph():
    config = get_config()
    return build_graph(config)
