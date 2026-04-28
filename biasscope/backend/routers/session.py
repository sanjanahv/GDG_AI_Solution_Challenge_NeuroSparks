"""
Session Router — OpenEnv-style API for structured bias testing sessions.

Endpoints:
  POST /api/session/reset     — Start new session with domain + difficulty + seed
  POST /api/session/step      — Submit one decision/action
  GET  /api/session/score     — Get current session grade
  GET  /api/session/state     — Full session state
  POST /api/session/ablation  — Run ablation study
  POST /api/session/baselines — Compare against baseline policies
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from services.profile_generator import generate_profile, get_difficulty_config
from services.fairness_scorer import grade_session, score_single_decision
from services.ablation import run_ablation
from services.baselines import compare_policies
from services.rl_memory import load_memory

# In-memory session decision storage
_session_decisions = {"job": [], "loan": [], "college": []}

router = APIRouter()

# ─── In-memory session state ─────────────────────────────────────

_active_sessions = {}


class SessionResetRequest(BaseModel):
    domain: str = Field(default="job")
    difficulty: str = Field(default="medium")
    seed: int = Field(default=42)


class SessionStepRequest(BaseModel):
    domain: str
    decision: str
    profile: dict
    weighted_attributes: list[dict] = []


class AblationRequest(BaseModel):
    domain: str = Field(default="job")
    n_seeds: int = Field(default=20, ge=5, le=100)
    base_seed: int = Field(default=42)


class BaselineRequest(BaseModel):
    domain: str = Field(default="job")
    n_seeds: int = Field(default=20, ge=5, le=100)


@router.post("/session/reset")
def reset_session(request: SessionResetRequest):
    """Start a new structured session with specified domain, difficulty, and seed."""
    domain = request.domain.lower()
    config = get_difficulty_config(request.difficulty)

    # Clear existing session decisions
    _session_decisions[domain] = []

    # Store session config
    _active_sessions[domain] = {
        "domain": domain,
        "difficulty": request.difficulty,
        "seed": request.seed,
        "config": config,
        "current_step": 0,
        "max_steps": config["decisions_required"],
        "profiles_generated": [],
    }

    # Pre-generate the first profile
    first_profile = generate_profile(domain, request.seed, request.difficulty)

    return {
        "status": "session_started",
        "domain": domain,
        "difficulty": request.difficulty,
        "seed": request.seed,
        "decisions_required": config["decisions_required"],
        "description": config["description"],
        "first_profile": first_profile,
    }


@router.post("/session/step")
def session_step(request: SessionStepRequest):
    """Submit one decision in the current session."""
    domain = request.domain.lower()
    session = _active_sessions.get(domain)

    if not session:
        return {"error": "No active session. Call /session/reset first."}

    if session["current_step"] >= session["max_steps"]:
        return {"error": "Session complete. Call /session/score for final grade."}

    # Score this decision
    memory = load_memory()
    session_decisions = _session_decisions.get(domain, [])

    fairness = score_single_decision(
        domain=domain,
        profile=request.profile,
        decision=request.decision,
        weighted_attributes=request.weighted_attributes,
        memory=memory,
        session_decisions=session_decisions,
    )

    # Store decision
    _session_decisions.setdefault(domain, []).append({
        "profile": {"attributes": request.profile},
        "decision": request.decision,
        "weighted_attributes": request.weighted_attributes,
        "feedback_given": False,
        "fairness_score": fairness["total_score"],
    })

    session["current_step"] += 1

    # Generate next profile if session not complete
    next_profile = None
    if session["current_step"] < session["max_steps"]:
        next_seed = session["seed"] + session["current_step"]
        next_profile = generate_profile(domain, next_seed, session["difficulty"])

    return {
        "step": session["current_step"],
        "max_steps": session["max_steps"],
        "fairness_score": fairness,
        "session_complete": session["current_step"] >= session["max_steps"],
        "next_profile": next_profile,
    }


@router.get("/session/score")
def session_score(domain: str = "job"):
    """Get the current session grade."""
    memory = load_memory()
    decisions = _session_decisions.get(domain.lower(), [])
    result = grade_session(decisions, domain, rl_memory=memory)
    return result


@router.get("/session/state")
def session_state(domain: str = "job"):
    """Get full session state."""
    session = _active_sessions.get(domain.lower())
    decisions = _session_decisions.get(domain.lower(), [])

    return {
        "active_session": session is not None,
        "session_config": session,
        "decision_count": len(decisions),
        "decisions": decisions[-5:],  # Last 5 decisions
    }


@router.post("/session/ablation")
def run_ablation_study(request: AblationRequest):
    """Run an ablation study to identify which fairness mechanic matters most."""
    result = run_ablation(
        domain=request.domain.lower(),
        n_seeds=request.n_seeds,
        base_seed=request.base_seed,
    )
    return result


@router.post("/session/baselines")
def run_baseline_comparison(request: BaselineRequest):
    """Compare user's session against 4 baseline policies."""
    memory = load_memory()
    result = compare_policies(
        domain=request.domain.lower(),
        n_seeds=request.n_seeds,
        memory=memory,
    )
    return result
