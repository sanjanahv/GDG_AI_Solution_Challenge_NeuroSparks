# backend/routers/decision.py
# ============================================================
# Decision Routes
# ============================================================
# These endpoints handle the decision-making flow:
#   - /classify: Classify attributes as NORMAL/AMBIGUOUS/REDUNDANT
#   - /store-decision: Store a decision for later bias analysis
# ============================================================

from fastapi import APIRouter
from services.attribute_classifier import classify_attributes, get_classification_summary
from services.rl_memory import get_memory_context

router = APIRouter()

# In-memory session storage for bias analysis
session_decisions = []


@router.post("/classify")
def classify(domain: str, attributes: dict):
    """
    Classify profile attributes for a given domain.
    Returns NORMAL / AMBIGUOUS / REDUNDANT / PROTECTED for each attribute.
    """
    summary = get_classification_summary(domain, attributes)
    return summary


@router.post("/store-decision")
def store_decision(decision_record: dict):
    """
    Store a decision record for later AIF360 bias analysis.
    The frontend sends each decision here after it's made.
    """
    session_decisions.append(decision_record)
    return {
        "status": "stored",
        "total_decisions": len(session_decisions),
        "ready_for_analysis": len(session_decisions) >= 5,
        "message": (
            f"Decision stored. {len(session_decisions)} total. "
            f"{'Ready for bias analysis!' if len(session_decisions) >= 5 else f'Need {5 - len(session_decisions)} more for bias analysis.'}"
        ),
    }


@router.get("/session-decisions")
def get_session_decisions():
    """Get all stored session decisions."""
    return {
        "decisions": session_decisions,
        "count": len(session_decisions),
    }


@router.post("/clear-session")
def clear_session():
    """Clear all stored session decisions."""
    session_decisions.clear()
    return {"status": "cleared"}


@router.get("/memory-context/{domain}")
def memory_context(domain: str):
    """
    Get the RL memory context string for a domain.
    This is what gets injected into Gemini prompts.
    """
    context = get_memory_context(domain)
    return {
        "domain": domain,
        "context": context,
    }


@router.post("/score-decision")
def score_decision_endpoint(decision_record: dict):
    """
    Score a single decision using the 10-term fairness breakdown.
    Inspired by Financial Triage's 14-term additive reward.
    """
    from services.fairness_scorer import score_single_decision
    from services.rl_memory import get_full_memory

    result = score_single_decision(
        domain=decision_record.get("domain", "job"),
        weighted_attributes=decision_record.get("weighted_attributes", []),
        rl_memory=get_full_memory(),
        session_history=session_decisions,
    )
    return result


@router.get("/session-grade/{domain}")
def session_grade(domain: str):
    """
    Compute an overall session fairness grade (A-F).
    Inspired by Financial Triage's grade_episode().
    """
    from services.fairness_scorer import grade_session
    from services.rl_memory import get_full_memory

    result = grade_session(
        session_history=session_decisions,
        domain=domain,
        rl_memory=get_full_memory(),
    )
    return result
