# backend/routers/bias.py
# ============================================================
# Bias Analysis & Post-Processing Routes
# ============================================================
# These endpoints handle the AIF360 integration:
#   - /analyze-bias: Run bias metrics on session data
#   - /apply-correction: Apply post-processing to fix bias
#   - /recommend: Get algorithm recommendation
#   - /memory: Get current RL memory state
# ============================================================

from fastapi import APIRouter
from schemas.domains import BiasAnalysisRequest, PostProcessingRequest
from services.aif360_service import (
    compute_bias_metrics,
    apply_post_processing,
    get_recommendation,
    PROTECTED_ATTRIBUTES,
)
from services.rl_memory import get_full_memory, get_stats
from services.attribute_classifier import classify_attributes, get_classification_summary

router = APIRouter()


@router.post("/analyze-bias")
def analyze_bias(req: BiasAnalysisRequest):
    """
    Run AIF360 bias metrics on a batch of session decisions.
    
    Send your session history (array of {attributes, decision} records)
    and get back disparate impact + statistical parity metrics.
    """
    metrics = compute_bias_metrics(
        records=req.records,
        protected_attr=req.protected_attribute,
    )
    return metrics


@router.post("/apply-correction")
def apply_correction(req: PostProcessingRequest):
    """
    Apply AIF360 post-processing to correct biased decisions.
    
    Supported algorithms:
      - calibrated_eq_odds (default, balanced)
      - eq_odds (strictest)
      - reject_option (borderline flip)
    """
    result = apply_post_processing(
        records=req.records,
        protected_attr=req.protected_attribute,
        algorithm=req.algorithm,
    )
    return result


@router.post("/recommend")
def recommend_algorithm(req: BiasAnalysisRequest):
    """
    Analyze bias and recommend which post-processing algorithm to use.
    """
    recommendation = get_recommendation(
        records=req.records,
        protected_attr=req.protected_attribute,
    )
    return recommendation


@router.get("/memory")
def get_memory():
    """Get the full RL memory bank (all domains)."""
    return {
        "memory": get_full_memory(),
        "stats": get_stats(),
    }


@router.get("/protected-attributes/{domain}")
def get_protected_attrs(domain: str):
    """Get the list of protected attributes to check for a domain."""
    attrs = PROTECTED_ATTRIBUTES.get(domain.lower(), ["Gender"])
    return {"domain": domain, "protected_attributes": attrs}


@router.post("/classify-attributes")
def classify_attrs(domain: str, attributes: dict):
    """Classify attributes as NORMAL/AMBIGUOUS/REDUNDANT/PROTECTED."""
    summary = get_classification_summary(domain, attributes)
    return summary
