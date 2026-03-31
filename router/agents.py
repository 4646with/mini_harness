"""Lead Agent 和 Sub Agent - 复杂任务处理"""
import json
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from engine.patched_kimi import get_kimi_llm


class LeadAgent:
    """Lead Agent: 分解复杂任务"""

    def __init__(self):
        self.llm = get_kimi_llm()

    def decompose(self, task: str) -> List[Dict[str, Any]]:
        """将复杂任务分解为子任务"""
        prompt = f"""将以下任务分解为具体的子任务步骤：

任务: {task}

输出格式（JSON数组）：
[{{"skill": "技能名", "goal": "子目标", "priority": 优先级数字}}]

技能名目前只支持：save_memory, web_search
只输出JSON，不要其他内容。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return [{"skill": "unknown", "goal": task, "priority": 1}]


class SubAgent:
    """Sub Agent: 执行单个子任务"""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.llm = get_kimi_llm()

    def execute(self, goal: str) -> str:
        """执行单个子任务"""
        if self.skill_name == "save_memory":
            prompt = f"""将以下信息保存到记忆：

{goal}

请调用 save_memory 工具，memories 数组格式：
[{{"content": "具体记忆内容", "memory_type": "general"}}]"""

            response = self.llm.invoke([HumanMessage(content=prompt)])

            tool_calls = []
            if hasattr(response, "tool_calls"):
                tool_calls = response.tool_calls

            for tc in tool_calls:
                if tc.get("name") == "save_memory":
                    from nodes.tool import execute_tool_call
                    return execute_tool_call(tc)

            if hasattr(response, "content") and response.content:
                return response.content

        return f"[完成] {goal}"


class Aggregator:
    """结果聚合器"""

    def aggregate(self, results: List[str], original_task: str) -> str:
        """聚合多个子任务结果"""
        prompt = f"""原始任务: {original_task}

执行结果:
{chr(10).join(results)}

请用一段话总结以上结果，回复用户。"""

        llm = get_kimi_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
