# backend/schemas/domains.py
# ============================================================
# Pydantic models for request/response validation
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional


class DecisionRequest(BaseModel):
    """Request to analyze a decision."""
    domain: str = Field(..., description="Domain: 'job', 'loan', or 'college'")
    profile: dict = Field(..., description="Profile attributes dict")
    decision: str = Field(..., description="AI decision: 'Hire', 'Reject', etc.")
    weighted_attributes: list = Field(
        default=[],
        description="List of weighted attributes from the AI decision"
    )


class FeedbackRequest(BaseModel):
    """Request to reward/penalize an attribute."""
    domain: str = Field(..., description="Domain: 'job', 'loan', or 'college'")
    attribute: str = Field(..., description="The attribute name to reward/penalize")


class BiasAnalysisRequest(BaseModel):
    """Request to run AIF360 bias analysis on session data."""
    domain: str = Field(..., description="Domain: 'job', 'loan', or 'college'")
    records: list = Field(
        ...,
        description="List of decision records, each with 'attributes' dict and 'decision' string"
    )
    protected_attribute: str = Field(
        default="Gender",
        description="Protected attribute to check for bias"
    )


class PostProcessingRequest(BaseModel):
    """Request to apply AIF360 post-processing correction."""
    domain: str = Field(..., description="Domain: 'job', 'loan', or 'college'")
    records: list = Field(
        ...,
        description="List of decision records"
    )
    protected_attribute: str = Field(
        default="Gender",
        description="Protected attribute to correct for"
    )
    algorithm: str = Field(
        default="calibrated_eq_odds",
        description="Algorithm: 'calibrated_eq_odds', 'eq_odds', or 'reject_option'"
    )


class MemorySyncRequest(BaseModel):
    """Request to sync frontend memory with backend."""
    memory: dict = Field(
        ...,
        description="Frontend localStorage memory object"
    )
