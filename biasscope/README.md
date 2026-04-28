# BiasScope — AI Bias Detection & De-biasing System

> **GDG Solution Challenge 2026 · Neuro Sparks · RVCE**

BiasScope is an interactive tool that detects and mitigates AI biases across **hiring, lending, and college admissions** using IBM AIF360, Reinforcement Learning, and real-time proxy-bias classification.

## Architecture

```
biasscope/
├── backend/                    Python FastAPI
│   ├── main.py                 Entry point (CORS, routers)
│   ├── services/
│   │   ├── aif360_service.py   Disparate Impact, Reweighing
│   │   ├── fairness_scorer.py  10-term per-decision scoring
│   │   ├── rl_memory.py        JSON-persisted RL memory bank
│   │   ├── attribute_classifier.py  NORMAL/AMBIGUOUS/REDUNDANT/PROTECTED
│   │   ├── profile_generator.py     Seeded, difficulty-scaled profiles
│   │   ├── ablation.py         Ablation studies
│   │   └── baselines.py        4 comparison policies
│   ├── routers/
│   │   ├── decision.py         /api/decide, /api/session-grade
│   │   ├── feedback.py         /api/reward, /api/penalize, /api/flag-proxy
│   │   ├── bias.py             /api/analyze-bias, /api/apply-correction
│   │   └── session.py          /api/session/reset, /step, /ablation
│   ├── schemas/domains.py      Pydantic models
│   └── data/
│       ├── memory_bank.json    RL memory persistence
│       └── calibration.json    Real-world bias statistics
│
├── frontend/                   React + Vite
│   ├── src/
│   │   ├── App.jsx             Main UI with 3 tabs
│   │   ├── index.css           Dark glassmorphism theme
│   │   └── services/
│   │       ├── llm.js          Gemini API (profile gen + decisions)
│   │       ├── rl.js           Client-side RL memory
│   │       ├── attributeClassifier.js  Proxy-bias classification
│   │       └── api.js          Backend API client
│   └── index.html              SEO + Open Graph tags
│
├── implementation.md           Full technical specification
├── plan.md                     7-phase implementation plan
└── README.md                   ← You are here
```

## Quick Start

### 1. Frontend
```bash
cd biasscope/frontend
npm install
npm run dev
```
Open **http://localhost:5173** — enter your Gemini API key.

### 2. Backend (optional — enables AIF360 + advanced features)
```bash
cd biasscope/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
The frontend auto-detects the backend. Swagger docs at **http://localhost:8000/docs**.

## Key Features

| Feature | Description |
|---|---|
| **3 Domains** | Job hiring, loan approval, college admissions |
| **Proxy-Bias Detection** | Flags attributes like Zip Code (→ race), University (→ wealth) |
| **10-Term Fairness Score** | Per-decision score with signed breakdown (A-F grade) |
| **AIF360 Integration** | Disparate Impact, Statistical Parity, Reweighing post-processing |
| **RL Feedback Loop** | Reward/Penalize/Flag Proxy → shapes AI behavior via prompt injection |
| **Session Grading** | 6 weighted criteria (30% DI, 20% balance, 15% hygiene, etc.) |
| **Ablation Studies** | Disable one mechanic at a time, measure Δ in bias |
| **Baseline Comparison** | Compare your session against 4 naive policies |
| **Seeded Profiles** | Reproducible profile generation with 3 difficulty levels |

## How It Works

1. **Generate** a profile (Gemini creates realistic applicant data)
2. **Observe** the AI's decision + weighted attributes
3. **Classify** each attribute as Normal 🟢, Ambiguous 🟡, Redundant 🔴, or Protected 🩷
4. **Give feedback** — Reward, Penalize, or Flag Proxy
5. **Track** fairness score, session grade, and bias metrics in real-time
6. **Correct** detected biases using AIF360 post-processing algorithms

## Tech Stack

- **Frontend:** React 18, Vite, Lucide Icons, Google Fonts (Inter + Outfit)
- **Backend:** FastAPI, Pydantic, AIF360, Pandas, NumPy
- **AI:** Google Gemini Flash (profile generation + decision-making)
- **Design:** Dark glassmorphism, gradient accents, micro-animations

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/decide` | Classify + score a decision |
| `POST` | `/api/reward` | Reward an attribute |
| `POST` | `/api/penalize` | Penalize an attribute |
| `POST` | `/api/flag-proxy` | Flag as proxy bias |
| `GET` | `/api/memory/{domain}` | Get RL memory |
| `POST` | `/api/analyze-bias` | Run AIF360 analysis |
| `POST` | `/api/apply-correction` | Apply Reweighing correction |
| `GET` | `/api/bias-report/{domain}` | Full bias report |
| `GET` | `/api/session-grade/{domain}` | Session grade (A-F) |
| `POST` | `/api/session/ablation` | Run ablation study |
| `POST` | `/api/session/baselines` | Compare baseline policies |

## Team

**Neuro Sparks** — RV College of Engineering, Bengaluru

## License

MIT
