# backend/main.py
# FastAPI entry point for BiasScope backend
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import decision, feedback, bias

app = FastAPI(
    title="BiasScope API",
    description="Post-hoc AI Bias Detection & De-biasing using AIF360 + RL",
    version="1.0.0"
)

# Allow frontend (Vite dev server) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(decision.router, prefix="/api", tags=["Decision"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(bias.router, prefix="/api", tags=["Bias Analysis"])


@app.get("/")
def root():
    return {"status": "ok", "message": "BiasScope API is running"}
