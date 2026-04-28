# backend/routers/decision.py
# ============================================================
# Decision Routes
# ============================================================
# These endpoints handle the decision-making flow:
#   - /classify: Classify attributes as NORMAL/AMBIGUOUS/REDUNDANT/UNKNOWN
#   - /store-decision: Store a decision for later bias analysis
#   - /decide: Full decision pipeline (classify + score + store)
#   - /generate-profile: Generate a profile without Gemini
# ============================================================

import random
from fastapi import APIRouter
from services.attribute_classifier import classify_attributes, get_classification_summary
from services.rl_memory import get_memory_context, get_full_memory
from services.profile_generator import generate_profile as gen_profile
from services.fairness_scorer import score_single_decision

router = APIRouter()

# In-memory session storage for bias analysis
session_decisions = []


@router.get("/generate-profile")
def generate_profile_endpoint(domain: str = "job", seed: int = None, difficulty: str = "medium"):
    """
    Generate a profile using the backend seeded generator.
    No Gemini API call needed — deterministic and free.
    
    If no seed is provided, a random one is generated.
    """
    if seed is None:
        seed = random.randint(0, 999999)
    
    profile = gen_profile(domain.lower(), seed, difficulty)
    return profile


@router.post("/classify")
def classify(domain: str, attributes: dict):
    """
    Classify profile attributes for a given domain.
    Returns NORMAL / AMBIGUOUS / REDUNDANT / PROTECTED / UNKNOWN for each attribute.
    """
    summary = get_classification_summary(domain, attributes)
    return summary


@router.post("/decide")
def decide(decision_record: dict):
    """
    Full decision pipeline:
      1. Classify the profile attributes
      2. Score the decision using the 11-term fairness breakdown
      3. Store the decision for later AIF360 batch analysis
    
    Expected input:
    {
      "domain": "job",
      "profile": { "Age": 28, "Gender": "Female", ... },
      "decision": "Hire",
      "weighted_attributes": [
        {"attribute": "Experience", "weight": "40%", "reasoning": "..."}
      ]
    }
    """
    domain = decision_record.get("domain", "job").lower()
    profile = decision_record.get("profile", {})
    decision_str = decision_record.get("decision", "Unknown")
    weighted_attrs = decision_record.get("weighted_attributes", [])

    # 1. Classify attributes
    classifications = classify_attributes(domain, profile, use_gemini=False)

    # 2. Score the decision with the 11-term fairness breakdown
    rl_memory = get_full_memory()
    fairness = score_single_decision(
        domain=domain,
        weighted_attributes=weighted_attrs,
        rl_memory=rl_memory,
        session_history=session_decisions,
        profile=profile,
    )

    # 3. Store in session for AIF360 batch analysis
    record = {
        "attributes": profile,
        "decision": decision_str,
        "weighted_attributes": weighted_attrs,
        "feedback_given": False,
    }
    session_decisions.append(record)

    # Build fairness score response matching frontend expectations
    breakdown_list = []
    for term, value in fairness.get("breakdown", {}).items():
        sign = "+" if value > 0 else "-"
        # Determine cap based on term
        caps = {
            "fair_attribute_weight": 0.25, "protected_attr_ignored": 0.15,
            "rl_alignment": 0.20, "transparency_score": 0.12,
            "diverse_outcome": 0.10, "protected_attr_weighted": 0.30,
            "proxy_leak": 0.20, "redundant_attr_used": 0.16,
            "outcome_skew": 0.12, "inaction_streak": 0.08,
            "unknown_attr_risk": 0.18,
        }
        breakdown_list.append({
            "term": term,
            "value": abs(value),
            "sign": sign,
            "cap": caps.get(term, 0.20),
        })

    return {
        "classifications": classifications,
        "fairness_score": {
            "total_score": fairness.get("score", 0.5),
            "letter_grade": fairness.get("grade", "C"),
            "interpretation": fairness.get("interpretation", ""),
            "breakdown": breakdown_list,
            "unknown_attributes": fairness.get("unknown_attributes", []),
        },
        "decision_stored": True,
        "total_session_decisions": len(session_decisions),
        "ready_for_aif360": len(session_decisions) >= 5,
    }


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
    Score a single decision using the 11-term fairness breakdown.
    Inspired by Financial Triage's 14-term additive reward.
    """
    result = score_single_decision(
        domain=decision_record.get("domain", "job"),
        weighted_attributes=decision_record.get("weighted_attributes", []),
        rl_memory=get_full_memory(),
        session_history=session_decisions,
        profile=decision_record.get("profile", None),
    )
    return result


@router.get("/session-grade/{domain}")
def session_grade(domain: str):
    """
    Compute an overall session fairness grade (A-F).
    Inspired by Financial Triage's grade_episode().
    """
    from services.fairness_scorer import grade_session

    result = grade_session(
        session_history=session_decisions,
        domain=domain,
        rl_memory=get_full_memory(),
    )
    return result
