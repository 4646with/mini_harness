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


class ToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any]


class ChatResponse(BaseModel):
    status: str  # "completed" 或 "pending_tools"
    response: str
    thread_id: str
    tool_calls: Optional[List[ToolCall]] = None


class SkillInfo(BaseModel):
    name: str
    description: str


class ToolApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


class ToolActionRequest(BaseModel):
    """工具批准/拒绝请求（用于无状态网关）"""
    thread_id: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    action: str  # "approve" 或 "reject"


class PendingToolCallsResponse(BaseModel):
    thread_id: str
    tool_calls: List[ToolCall]
    status: str
