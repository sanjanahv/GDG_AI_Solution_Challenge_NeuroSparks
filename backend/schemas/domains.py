"""
Pydantic schemas for BiasScope API.
Defines request/response models for all domains and API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────

class DomainEnum(str, Enum):
    job = "job"
    loan = "loan"
    college = "college"


class AttributeClassification(str, Enum):
    NORMAL = "NORMAL"
    AMBIGUOUS = "AMBIGUOUS"
    REDUNDANT = "REDUNDANT"
    PROTECTED = "PROTECTED"


class LetterGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# ─── Request Models ──────────────────────────────────────────────

class DecisionRequest(BaseModel):
    """Request body for POST /api/decide"""
    domain: DomainEnum
    profile: dict = Field(..., description="Applicant profile as key-value pairs")
    decision: str = Field(..., description="AI decision: Hire/Reject, Approve/Deny, Admit/Waitlist")
    weighted_attributes: list[dict] = Field(
        ...,
        description="List of {attribute, weight, reasoning} dicts from the AI"
    )


class FeedbackRequest(BaseModel):
    """Request body for POST /api/reward, /api/penalize, /api/flag-proxy"""
    domain: DomainEnum
    attribute: str = Field(..., description="The attribute name to give feedback on")


class BiasAnalysisRequest(BaseModel):
    """Request body for POST /api/analyze-bias"""
    domain: DomainEnum
    records: list[dict] = Field(
        ...,
        description="List of decision records, each containing profile attributes + 'decision' field"
    )
    protected_attribute: str = Field(
        ...,
        description="The protected attribute to check bias for (e.g., 'Gender', 'Age')"
    )


class CorrectionRequest(BaseModel):
    """Request body for POST /api/apply-correction"""
    domain: DomainEnum
    records: list[dict] = Field(..., description="List of decision records")
    protected_attribute: str
    algorithm: str = Field(
        default="reweighing",
        description="AIF360 algorithm to use: 'reweighing' or 'calibrated_eq_odds'"
    )


class MemorySyncRequest(BaseModel):
    """Request body for POST /api/memory/sync — accepts frontend localStorage dump"""
    memory: dict = Field(
        ...,
        description="Full memory object from frontend localStorage"
    )


# ─── Response Models ─────────────────────────────────────────────

class FairnessTermBreakdown(BaseModel):
    """A single term in the 10-term fairness breakdown"""
    term: str
    sign: str = Field(description="'+' or '-'")
    value: float
    cap: float
    description: str


class FairnessScoreResponse(BaseModel):
    """Response from the fairness scorer"""
    total_score: float = Field(ge=0, le=1)
    letter_grade: LetterGrade
    breakdown: list[FairnessTermBreakdown]


class DecisionResponse(BaseModel):
    """Response body for POST /api/decide"""
    attribute_classifications: dict[str, str]
    fairness_score: FairnessScoreResponse
    memory_context_used: str


class BiasMetricsResponse(BaseModel):
    """Response from AIF360 bias analysis"""
    disparate_impact: float
    statistical_parity_diff: float
    bias_detected: bool
    protected_attribute: str
    sample_size: int


class SessionGradeResponse(BaseModel):
    """Response body for GET /api/session-grade/{domain}"""
    grade: float = Field(ge=0, le=1)
    letter_grade: LetterGrade
    criteria: dict[str, float]
    decision_count: int


class MemoryResponse(BaseModel):
    """Response body for GET /api/memory/{domain}"""
    domain: str
    rewarded: list[str]
    penalized: list[str]
    ambiguous: list[str]
