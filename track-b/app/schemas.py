from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    document: str
    chunk_id: str


class AssistantAnswer(BaseModel):
    answer: str = Field(..., description="The direct answer to the user's question")
    sources: list[SourceRef] = Field(default_factory=list, description="Chunks used to answer, if any")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's self-rated confidence 0-1")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[Union[ChatMessage, dict[str, Any]]]] = None
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    max_tokens: Optional[int] = Field(default=800, ge=1, le=4096, description="Max tokens to generate")

    def get_normalized_history(self) -> list[dict[str, str]] | None:
        """Safely normalizes history to a list of role/content dicts."""
        if not self.history:
            return None
        normalized = []
        for item in self.history:
            if isinstance(item, ChatMessage):
                normalized.append({"role": item.role, "content": item.content})
            elif isinstance(item, dict):
                normalized.append({
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                })
        return normalized


class StructuredChatRequest(BaseModel):
    message: str
    schema_definition: Optional[dict[str, Any]] = None
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=800, ge=1, le=4096)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = []
    confidence: Optional[float] = None
    structured_data: Optional[dict[str, Any]] = None


class ResearchRequest(BaseModel):
    """Request for the agentic research endpoint (/chat/research)."""
    message: str = Field(..., min_length=1, description="Question for the research agent")
    mode: Literal["single", "multi"] = Field(
        default="multi",
        description="single = one agent with in-context self-check; multi = researcher + isolated verifier",
    )
    max_steps: int = Field(default=6, ge=1, le=10, description="Hard cap on researcher iterations")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=1200, ge=1, le=4096)
    inject_failure: Optional[str] = Field(
        default=None,
        description="TEST ONLY: force a tool failure (kb_down | malformed_retrieval | timeout)",
    )


class TokenUsageModel(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0
    calls: int = 0


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = []
    confidence: Optional[float] = None
    mode: str
    steps: int = Field(..., description="Number of researcher iterations")
    stop_reason: str = Field(..., description="submitted | clarification | max_steps")
    clarification: Optional[str] = None
    verified: Optional[bool] = None
    verification: Optional[dict[str, Any]] = None
    tools_used: list[str] = []
    trajectory: list[dict[str, Any]] = []
    fabricated_citations: list[dict[str, Any]] = []
    tool_errors: list[str] = []
    tokens: TokenUsageModel
    latency_ms: float = 0.0
    injection: str = ""
    max_steps: int = 0
