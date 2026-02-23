# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Pydantic models for AI chat."""
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    """Types of streaming events."""
    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    DONE = "done"


class StreamEvent(BaseModel):
    """A single streaming event emitted during chat."""
    type: StreamEventType
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None


# Type alias for the stream callback
StreamCallback = Callable[["StreamEvent"], None]


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=100000)


class ChatRequest(BaseModel):
    """Chat request body."""
    message: str = Field(..., min_length=1, max_length=50000)
    session_id: Optional[str] = Field(None, max_length=64)
    template_context: Optional[Dict[str, Any]] = None
    history: Optional[List[ChatMessage]] = Field(None, max_length=50)


class UsageStats(BaseModel):
    """Token usage statistics from a chat interaction."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ChatResponse(BaseModel):
    """Chat response."""
    ok: bool
    message: str
    session_id: str
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    execution_results: List[Dict[str, Any]] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    rounds_used: int = 0
    usage: Optional[UsageStats] = None
