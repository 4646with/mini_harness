"""Skill 系统"""
from skills.base import Skill
from skills.registry import SkillRegistry, get_skill_registry

__all__ = ["Skill", "SkillRegistry", "get_skill_registry"]
