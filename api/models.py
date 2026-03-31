"""API 请求/响应模型"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class SkillInfo(BaseModel):
    name: str
    description: str


class ToolApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


class PendingToolCallsResponse(BaseModel):
    thread_id: str
    tool_calls: List[Dict[str, Any]]
    status: str
