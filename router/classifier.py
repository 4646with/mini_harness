"""任务复杂度分类器 - DeerFlow 风格"""
from enum import Enum
from typing import Tuple


class TaskType(Enum):
    DIRECT = "direct"           # 简单任务，直接执行
    MULTI_AGENT = "multi_agent"  # 复杂任务，多智能体


class TaskClassifier:
    """分析用户输入，决定任务类型"""

    COMPLEX_KEYWORDS = [
        "分析", "拆解", "规划", "研究", "比较",
        "多个", "不同", "哪些", "如何完成",
        "分析一下", "拆解成", "调查", "研究",
        "详细说明", "全面", "系统", "步骤",
        "帮我了解", "深入", "探索"
    ]

    def classify(self, user_input: str) -> TaskType:
        """判断任务复杂度 - 规则匹配"""
        for kw in self.COMPLEX_KEYWORDS:
            if kw in user_input:
                return TaskType.MULTI_AGENT
        return TaskType.DIRECT

    def classify_with_llm(self, user_input: str) -> Tuple[TaskType, str]:
        """用 LLM 辅助判断（更准确但更慢）"""
        from langchain_core.messages import HumanMessage
        from engine.patched_kimi import get_kimi_llm

        llm = get_kimi_llm()
        prompt = f"""分析以下用户输入，判断任务复杂度：

用户输入: {user_input}

规则：
- 简单任务：单个问题、单一操作、直接回答（如"今天天气如何"、"帮我记住我的名字"）
- 复杂任务：需要多步骤、多个子任务、规划研究、多技能协作

直接回答：direct 或 multi_agent"""

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()

        if "multi" in decision:
            return TaskType.MULTI_AGENT, "LLM 判断为复杂任务"
        return TaskType.DIRECT, "LLM 判断为简单任务"


_classifier = TaskClassifier()


def get_classifier() -> TaskClassifier:
    return _classifier
