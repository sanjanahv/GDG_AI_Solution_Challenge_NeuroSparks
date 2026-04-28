# backend/routers/feedback.py
# ============================================================
# RL Feedback Routes (Reward / Penalize)
# ============================================================
# These endpoints let the frontend send reward/penalty feedback
# for specific attributes, which updates the RL memory bank.
#
# NEW: When a user penalizes an UNKNOWN attribute, it is auto-
# learned as a proxy so future classifications don't need Gemini.
# ============================================================

from fastapi import APIRouter
from schemas.domains import FeedbackRequest, MemorySyncRequest
from services.rl_memory import (
    reward_attribute,
    penalize_attribute,
    get_stats,
    get_full_memory,
    reset_memory,
    sync_memory,
)
from services.attribute_classifier import learn_proxy_from_feedback

router = APIRouter()


@router.post("/reward")
def reward(req: FeedbackRequest):
    """
    Reward an attribute — mark it as trusted/positive for a domain.
    This feeds into the RL memory that is injected into future Gemini prompts.

    If the attribute was previously UNKNOWN, this confirms it as safe (NORMAL)
    for future classifications.
    """
    memory = reward_attribute(req.domain, req.attribute)

    # Auto-learn: if this was an unknown attribute, mark it as safe
    learn_proxy_from_feedback(req.domain, req.attribute, action="reward")

    return {
        "status": "rewarded",
        "domain": req.domain,
        "attribute": req.attribute,
        "memory": memory,
        "stats": get_stats(),
    }


@router.post("/penalize")
def penalize(req: FeedbackRequest):
    """
    Penalize an attribute — mark it as biased/negative for a domain.
    Future Gemini decisions will be instructed to avoid weighting this attribute.

    If the attribute was previously UNKNOWN, this auto-learns it as a proxy
    attribute so future classifications flag it as AMBIGUOUS without needing
    a Gemini call.
    """
    memory = penalize_attribute(req.domain, req.attribute)

    # Auto-learn: if this was an unknown attribute, add it to learned proxies
    learn_proxy_from_feedback(req.domain, req.attribute, action="penalize")

    return {
        "status": "penalized",
        "domain": req.domain,
        "attribute": req.attribute,
        "memory": memory,
        "stats": get_stats(),
    }


@router.get("/stats")
def stats():
    """Get aggregate stats across all domains."""
    return get_stats()


@router.post("/sync-memory")
def sync(req: MemorySyncRequest):
    """
    Sync frontend localStorage memory with backend JSON file.
    Merges both, resolving conflicts by preferring the frontend version.
    """
    merged = sync_memory(req.memory)
    return {
        "status": "synced",
        "memory": merged,
        "stats": get_stats(),
    }


@router.post("/reset-memory")
def reset():
    """Reset all RL memory to empty state."""
    memory = reset_memory()
    return {
        "status": "reset",
        "memory": memory,
    }
