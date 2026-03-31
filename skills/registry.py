"""Skill 注册表"""
import importlib
from typing import Dict, Optional, List
from skills.base import Skill


class SkillRegistry:
    """动态技能注册表"""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_skills(self) -> List[Dict]:
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def load_from_module(self, module_path: str):
        """动态加载技能模块"""
        module = importlib.import_module(module_path)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Skill) and attr != Skill:
                self.register(attr())


_registry = SkillRegistry()


def get_skill_registry() -> SkillRegistry:
    return _registry
