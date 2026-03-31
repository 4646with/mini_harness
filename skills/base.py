"""Skill 基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Skill(ABC):
    """动态技能基类"""

    name: str
    description: str
    parameters: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行技能"""
        pass
