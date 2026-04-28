# backend/routers/feedback.py
# ============================================================
# RL Feedback Routes (Reward / Penalize)
# ============================================================
# These endpoints let the frontend send reward/penalty feedback
# for specific attributes, which updates the RL memory bank.
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

router = APIRouter()


@router.post("/reward")
def reward(req: FeedbackRequest):
    """
    Reward an attribute — mark it as trusted/positive for a domain.
    This feeds into the RL memory that is injected into future Gemini prompts.
    """
    memory = reward_attribute(req.domain, req.attribute)
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
    """
    memory = penalize_attribute(req.domain, req.attribute)
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
