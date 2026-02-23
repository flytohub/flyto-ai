# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Pydantic models for AI chat."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class ChatResponse(BaseModel):
    """Chat response."""
    ok: bool
    message: str
    session_id: str
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
