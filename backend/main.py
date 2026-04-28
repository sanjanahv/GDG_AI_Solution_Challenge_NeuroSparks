"""
BiasScope Backend — FastAPI Entry Point

De-biasing AI Decision Systems using AIF360 + RL
GDG Solution Challenge 2026 · Neuro Sparks · RVCE

Routes:
  /api/decide, /api/score-decision, /api/session-grade/{domain}
  /api/reward, /api/penalize, /api/flag-proxy, /api/memory/{domain}, /api/memory/sync
  /api/analyze-bias, /api/apply-correction, /api/bias-report/{domain}

Run with:
  cd backend
  python -m uvicorn main:app --reload --port 8000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import decision, feedback, bias

app = FastAPI(
    title="BiasScope API",
    description="AI Bias Detection & De-biasing System using AIF360 + Reinforcement Learning",
    version="1.0.0"
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(decision.router, prefix="/api", tags=["Decisions"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(bias.router, prefix="/api", tags=["Bias Analysis"])


@app.on_event("startup")
async def startup():
    """Ensure data directory exists on startup."""
    os.makedirs("data", exist_ok=True)


@app.get("/")
def root():
    return {
        "name": "BiasScope API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "decisions": "/api/decide",
            "feedback": "/api/reward, /api/penalize, /api/flag-proxy",
            "memory": "/api/memory/{domain}",
            "bias_analysis": "/api/analyze-bias",
            "bias_report": "/api/bias-report/{domain}",
            "session_grade": "/api/session-grade/{domain}"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint for frontend to detect if backend is running."""
    return {"status": "healthy", "service": "biasscope-backend"}
