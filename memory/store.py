"""Simple JSON-based memory storage."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class MemoryStore:
    """Simple file-based memory storage."""
    
    def __init__(self, storage_path: str = "data/memories.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.memories = []
        self._load()
    
    def _load(self):
        """Load memories from file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except Exception:
                self.memories = []
        else:
            self.memories = []
    
    def _save(self):
        """Save memories to file."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def add(self, content: str, memory_type: str = "preference", 
            confidence: float = 0.9, metadata: Dict = None):
        """Add a new memory."""
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
        """Simple keyword-based memory search."""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories:
            content = memory.get("content", "").lower()
            # Simple keyword matching
            if any(word in content for word in query_lower.split()):
                if memory.get("confidence", 0) >= threshold:
                    results.append(memory)
        
        return sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    
    def get_all(self) -> List[Dict]:
        """Get all memories."""
        return self.memories
    
    def clear(self):
        """Clear all memories."""
        self.memories = []
        self._save()


# Global instance
_memory_store = None


def get_memory_store() -> MemoryStore:
    """Get singleton memory store instance."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
