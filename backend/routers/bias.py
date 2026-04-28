"""
Bias Analysis Router — AIF360 bias metrics and post-processing.

Endpoints:
  POST /api/analyze-bias       — Run AIF360 analysis on session data
  POST /api/apply-correction   — Apply AIF360 post-processing algorithm
  GET  /api/bias-report/{domain} — Full bias report for a domain
"""

from fastapi import APIRouter
from schemas.domains import BiasAnalysisRequest, CorrectionRequest, BiasMetricsResponse
from services.aif360_service import (
    compute_bias_metrics,
    compute_bias_metrics_aif360,
    apply_reweighing,
    get_domain_protected_attrs
)
from services.rl_memory import get_session_decisions

router = APIRouter()


@router.post("/analyze-bias")
def analyze_bias(request: BiasAnalysisRequest):
    """
    Run AIF360 bias analysis on a batch of decision records.
    Returns disparate impact, statistical parity difference, and whether bias is detected.

    The 80% Rule:
      - DI < 0.8 → Negative bias (unprivileged group disadvantaged)
      - DI > 1.25 → Positive bias (over-favoring unprivileged)
      - 0.8 to 1.25 → Acceptable range
    """
    result = compute_bias_metrics(
        records=request.records,
        protected_attr=request.protected_attribute
    )
    return result


@router.post("/analyze-bias-aif360")
def analyze_bias_full(request: BiasAnalysisRequest):
    """
    Full AIF360 analysis using BinaryLabelDataset.
    Falls back to manual computation if AIF360 is not installed.
    """
    result = compute_bias_metrics_aif360(
        records=request.records,
        protected_attr=request.protected_attribute
    )
    return result


@router.post("/apply-correction")
def apply_correction(request: CorrectionRequest):
    """
    Apply an AIF360 post-processing algorithm to de-bias decisions.

    Supported algorithms:
      - "reweighing": Reweighing pre-processing (adjusts instance weights)
      - "calibrated_eq_odds": Calibrated Equalized Odds post-processing

    Returns the corrected weights and new bias metrics.
    """
    if request.algorithm == "reweighing":
        result = apply_reweighing(
            records=request.records,
            protected_attr=request.protected_attribute
        )
    else:
        result = {
            "algorithm": request.algorithm,
            "success": False,
            "message": f"Algorithm '{request.algorithm}' not yet implemented. Use 'reweighing'."
        }

    return result


@router.get("/bias-report/{domain}")
def get_bias_report(domain: str):
    """
    Generate a full bias report for a domain using session data.
    Checks all protected attributes defined for the domain.
    """
    session = get_session_decisions(domain)

    if len(session) < 3:
        return {
            "domain": domain,
            "decision_count": len(session),
            "reports": [],
            "message": f"Need at least 3 decisions to generate a bias report. Currently: {len(session)}"
        }

    # Build records from session decisions
    records = []
    for sd in session:
        profile = sd.get("profile", {})
        attrs = profile.get("attributes", profile)
        record = dict(attrs)
        record["decision"] = sd.get("decision", "Reject")
        records.append(record)

    # Check each protected attribute
    protected_attrs = get_domain_protected_attrs(domain)
    reports = []

    for pa in protected_attrs:
        # Check if this attribute exists in records
        if any(pa in record for record in records):
            metrics = compute_bias_metrics(records, pa)
            reports.append(metrics)
        else:
            reports.append({
                "protected_attribute": pa,
                "message": f"Attribute '{pa}' not found in session profiles.",
                "disparate_impact": 1.0,
                "statistical_parity_diff": 0.0,
                "bias_detected": False,
                "sample_size": len(records)
            })

    # Overall assessment
    any_bias = any(r.get("bias_detected", False) for r in reports)

    return {
        "domain": domain,
        "decision_count": len(session),
        "overall_bias_detected": any_bias,
        "reports": reports
    }
