from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    filename: str
    kind: str
    characters: int


class SuggestedStep(BaseModel):
    step: str
    rationale: str | None = None


class SupportAnalysis(BaseModel):
    issue_summary: str = Field(..., description="Concise summary of the customer or system issue.")
    likely_cause: str | None = Field(None, description="Most likely cause inferred from the ticket and logs.")
    severity: str | None = Field(None, description="Suggested severity or priority.")
    affected_components: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    debugging_steps: list[SuggestedStep] = Field(default_factory=list)
    customer_response: str | None = Field(
        None,
        description="Optional support-friendly response that can be sent back to the requester.",
    )
    raw_model_output: dict[str, Any] | str | None = None


class AnalysisResponse(BaseModel):
    response_id: str | None = None
    model: str
    sources: list[SourceDocument]
    analysis: SupportAnalysis
