"""内置记忆技能"""
from skills.base import Skill
from memory.store import get_memory_store


class SaveMemorySkill(Skill):
    """保存记忆技能"""

    name = "save_memory"
    description = "将重要信息保存到长期记忆"
    parameters = {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "memory_type": {"type": "string", "enum": ["profile", "preference", "fact"]}
                    }
                }
            }
        },
        "required": ["memories"]
    }

    async def execute(self, memories: list, **kwargs) -> str:
        store = get_memory_store()
        results = []
        for mem in memories:
            content = mem.get("content")
            memory_type = mem.get("memory_type", "general")
            if content:
                res = store.save_memory(content, memory_type)
                results.append(res)
        return "\n".join(results) if results else "无记忆可保存"
