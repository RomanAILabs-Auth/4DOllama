# schemas.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    stream: bool = True
    options: dict[str, Any] | None = None
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    raw: bool | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[str] | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    format: Literal["", "json"] | None = None
    keep_alive: str | None = None


class ShowRequest(BaseModel):
    model: str
    verbose: bool | None = None


class ModelTag(BaseModel):
    name: str
    model: str = ""
    modified_at: str = ""
    size: int = 0
    digest: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class TagsResponse(BaseModel):
    models: list[ModelTag]
