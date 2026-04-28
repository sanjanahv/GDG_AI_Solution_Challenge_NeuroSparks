"""
Decision Router — Handles profile classification, decision scoring, and session grading.

Endpoints:
  POST /api/decide         — Classify attributes + score a decision
  POST /api/score-decision — Score a single decision against fairness criteria
  GET  /api/session-grade/{domain} — Grade entire session
"""

from fastapi import APIRouter
from schemas.domains import DecisionRequest, DecisionResponse, FairnessScoreResponse, SessionGradeResponse
from services.attribute_classifier import classify_attributes, get_proxy_explanation
from services.fairness_scorer import score_single_decision, grade_session
from services.rl_memory import (
    load_memory, get_memory_context, add_session_decision, get_session_decisions
)

router = APIRouter()


@router.post("/decide")
def decide(request: DecisionRequest):
    """
    Process an AI decision: classify attributes, compute fairness score,
    store in session, and return classifications + score.

    The frontend sends the profile + decision + weighted attributes
    (generated client-side via Gemini), and the backend analyzes it.
    """
    domain = request.domain.value
    profile = request.profile
    decision = request.decision
    weighted_attrs = request.weighted_attributes

    # 1. Classify all profile attributes
    classifications = classify_attributes(domain, profile)

    # Add proxy explanations for ambiguous attributes
    explanations = {}
    for attr, cls in classifications.items():
        if cls in ("AMBIGUOUS", "PROTECTED"):
            explanation = get_proxy_explanation(attr)
            if explanation:
                explanations[attr] = explanation

    # 2. Load RL memory
    memory = load_memory()
    memory_context = get_memory_context(domain)

    # 3. Get session decisions for context
    session = get_session_decisions(domain)

    # 4. Score the decision
    fairness = score_single_decision(
        domain=domain,
        profile=profile,
        decision=decision,
        weighted_attributes=weighted_attrs,
        memory=memory,
        session_decisions=session
    )

    # 5. Store this decision in session
    add_session_decision(domain, {
        "profile": {"attributes": profile},
        "decision": decision,
        "weighted_attributes": weighted_attrs,
        "feedback_given": False,
        "fairness_score": fairness["total_score"]
    })

    return {
        "attribute_classifications": classifications,
        "proxy_explanations": explanations,
        "fairness_score": fairness,
        "memory_context_used": memory_context,
        "session_decision_count": len(session) + 1
    }


@router.post("/score-decision")
def score_decision(request: DecisionRequest):
    """
    Score a single decision without storing it in the session.
    Useful for what-if analysis.
    """
    domain = request.domain.value
    memory = load_memory()
    session = get_session_decisions(domain)

    fairness = score_single_decision(
        domain=domain,
        profile=request.profile,
        decision=request.decision,
        weighted_attributes=request.weighted_attributes,
        memory=memory,
        session_decisions=session
    )

    return fairness


@router.get("/session-grade/{domain}")
def get_session_grade(domain: str):
    """
    Grade the entire session for a domain.
    Uses 6 weighted criteria (30% DI, 20% balance, 15% hygiene, etc.)
    """
    memory = load_memory()
    session = get_session_decisions(domain)

    result = grade_session(
        domain=domain,
        session_decisions=session,
        memory=memory
    )

    return result


@router.get("/session-decisions/{domain}")
def list_session_decisions(domain: str):
    """Get all decisions made in the current session for a domain."""
    decisions = get_session_decisions(domain)
    return {
        "domain": domain,
        "count": len(decisions),
        "decisions": decisions
    }
