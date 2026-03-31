"""简单的基于 JSON 的记忆存储模块"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class MemoryStore:
    """简单的基于文件的记忆存储"""
    
    def __init__(self, storage_path: str = "data/memories.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.memories = []
        self._load()
    
    def _load(self):
        """从文件加载记忆"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except Exception:
                self.memories = []
        else:
            self.memories = []
    
    def _save(self):
        """保存记忆到文件"""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def add(self, content: str, memory_type: str = "preference", 
            confidence: float = 0.9, metadata: Dict = None):
        """添加新记忆"""
        memory = {
            "id": len(self.memories) + 1,
            "content": content,
            "type": memory_type,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.memories.append(memory)
        self._save()
        return memory
    
    def search(self, query: str, threshold: float = 0.5) -> List[Dict]:
        """基于关键词的简单记忆搜索"""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories:
            content = memory.get("content", "").lower()
            # 简单关键词匹配
            if any(word in content for word in query_lower.split()):
                if memory.get("confidence", 0) >= threshold:
                    results.append(memory)
        
        return sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    
    def get_all(self) -> List[Dict]:
        """获取所有记忆"""
        return self.memories
    
    def clear(self):
        """清除所有记忆"""
        self.memories = []
        self._save()


# 全局单例
_memory_store = None


def get_memory_store() -> MemoryStore:
    """获取单例记忆存储实例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
