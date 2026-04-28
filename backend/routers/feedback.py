"""
Feedback Router — Handles reward, penalize, flag-proxy, and memory operations.

Endpoints:
  POST /api/reward       — Reward an attribute (mark as fair)
  POST /api/penalize     — Penalize an attribute (mark as biased)
  POST /api/flag-proxy   — Flag an attribute as proxy bias risk
  GET  /api/memory/{domain} — Get current RL memory for a domain
  POST /api/memory/sync  — Sync frontend localStorage memory to backend
"""

from fastapi import APIRouter
from schemas.domains import FeedbackRequest, MemorySyncRequest, MemoryResponse
from services.rl_memory import (
    reward_attribute, penalize_attribute, flag_ambiguous,
    load_memory, sync_from_frontend, get_session_decisions
)

router = APIRouter()


@router.post("/reward")
def reward(request: FeedbackRequest):
    """
    Reward an attribute — marks it as trusted/fair for the domain.
    Removes it from the penalized list if present (conflict resolution).
    """
    domain = request.domain.value
    attribute = request.attribute

    memory = reward_attribute(domain, attribute)

    # Mark the last session decision as having feedback
    _mark_feedback(domain, attribute)

    return {
        "status": "rewarded",
        "domain": domain,
        "attribute": attribute,
        "memory": memory[domain]
    }


@router.post("/penalize")
def penalize(request: FeedbackRequest):
    """
    Penalize an attribute — marks it as biased for the domain.
    Removes it from the rewarded list if present (conflict resolution).
    """
    domain = request.domain.value
    attribute = request.attribute

    memory = penalize_attribute(domain, attribute)

    # Mark the last session decision as having feedback
    _mark_feedback(domain, attribute)

    return {
        "status": "penalized",
        "domain": domain,
        "attribute": attribute,
        "memory": memory[domain]
    }


@router.post("/flag-proxy")
def flag_proxy(request: FeedbackRequest):
    """
    Flag an attribute as a proxy bias risk (ambiguous).
    This doesn't remove it from rewarded/penalized lists.
    """
    domain = request.domain.value
    attribute = request.attribute

    memory = flag_ambiguous(domain, attribute)

    return {
        "status": "flagged_as_proxy",
        "domain": domain,
        "attribute": attribute,
        "memory": memory[domain]
    }


@router.get("/memory/{domain}")
def get_memory(domain: str):
    """Get the current RL memory for a specific domain."""
    memory = load_memory()
    d = domain.lower()
    domain_mem = memory.get(d, {"rewarded": [], "penalized": [], "ambiguous": []})

    return {
        "domain": d,
        "rewarded": domain_mem.get("rewarded", []),
        "penalized": domain_mem.get("penalized", []),
        "ambiguous": domain_mem.get("ambiguous", [])
    }


@router.get("/memory")
def get_all_memory():
    """Get the full RL memory across all domains."""
    return load_memory()


@router.post("/memory/sync")
def sync_memory(request: MemorySyncRequest):
    """
    Sync the frontend's localStorage memory to the backend.
    Frontend format uses 'positive'/'negative', backend uses 'rewarded'/'penalized'.
    This endpoint handles the mapping automatically.
    """
    merged = sync_from_frontend(request.memory)
    return {
        "status": "synced",
        "memory": merged
    }


def _mark_feedback(domain: str, attribute: str):
    """Mark the most recent session decision as having received feedback."""
    decisions = get_session_decisions(domain)
    if decisions:
        # Find the most recent decision that includes this attribute
        for decision in reversed(decisions):
            wa_attrs = [wa.get("attribute", "") for wa in decision.get("weighted_attributes", [])]
            if attribute in wa_attrs:
                decision["feedback_given"] = True
                break
        else:
            # If attribute not found in weights, just mark the last decision
            decisions[-1]["feedback_given"] = True
