# 🧭 BIASSCOPE — Implementation Guide
**De-biasing AI Decision Systems using AIF360 + RL**  
*GDG Solution Challenge 2026*

---

## 🗂️ Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack Decision](#3-tech-stack-decision)
4. [Folder Structure](#4-folder-structure)
5. [Phase-by-Phase Implementation](#5-phase-by-phase-implementation)
   - Phase 1: Backend — AIF360 Bias Detection Core
   - Phase 2: RL Reward/Penalty Memory System
   - Phase 3: Ambiguity Classification (Normal / Redundant / Ambiguous)
   - Phase 4: Frontend (React)
   - Phase 5: Gemini API Integration
   - Phase 6: Connecting Everything
6. [Domain Schemas](#6-domain-schemas)
7. [AIF360 Integration Notes](#7-aif360-integration-notes)
8. [RL Logic Explained Simply](#8-rl-logic-explained-simply)
9. [Ambiguity Engine](#9-ambiguity-engine)
10. [What to Build First (Priority Order)](#10-what-to-build-first-priority-order)
11. [Submission Checklist](#11-submission-checklist)

---

## 1. Project Overview

**BiasScope** is a post-hoc AI bias detection and de-biasing system.

Instead of retraining models, BiasScope:
- Takes AI decisions as input (e.g., "Hire / Reject", "Approve / Deny")
- Identifies which **attributes** the AI gave weight to
- Classifies each attribute as **Normal**, **Redundant**, or **Ambiguous**
- Uses **AIF360** to measure statistical bias
- Uses an **RL reward/penalty memory** to teach the system which attributes are acceptable over time
- Outputs a bias-corrected decision with transparency

**Domains:** Job Hiring · Loan Approval · College Admission

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (React)                   │
│  Domain Selector → Profile Generator → Decision UI  │
│  Reward/Penalize buttons → Bias Tracker Dashboard    │
└────────────────┬────────────────────────────────────┘
                 │ REST API calls
┌────────────────▼────────────────────────────────────┐
│               BACKEND (FastAPI - Python)             │
│                                                      │
│  ┌──────────────────┐    ┌────────────────────────┐  │
│  │  Gemini Flash    │    │  AIF360 Bias Engine    │  │
│  │  (Decision +     │    │  - Disparate Impact    │  │
│  │   Weight Estim.) │    │  - Statistical Parity  │  │
│  └──────────────────┘    │  - Reweighing algo     │  │
│                          └────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐    │
│  │  RL Memory Bank (per domain)                 │    │
│  │  - Rewarded attributes (positive bias list)  │    │
│  │  - Penalized attributes (negative bias list) │    │
│  │  - Ambiguous attributes (caution list)       │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  Attribute Classifier                        │    │
│  │  NORMAL | REDUNDANT | AMBIGUOUS              │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack Decision

| Layer | Technology | Why |
|---|---|---|
| Frontend | **React + Vite** | Fast, component-based, easy to connect APIs |
| Styling | **Tailwind CSS** | Dark theme UI like your existing screenshot |
| Backend | **FastAPI (Python)** | Native Python = native AIF360 + easy REST APIs |
| Bias Library | **AIF360 (IBM)** | Industry-standard, covers all 3 domains |
| AI Decisions | **Gemini Flash API** | Already in your prototype, keep it |
| RL Memory | **In-memory dict + JSON file** | Simple, no DB needed for prototype |
| Hosting (optional) | **Firebase (frontend) + Railway/Render (backend)** | Free tier friendly |

---

## 4. Folder Structure

```
biasscope/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DomainSelector.jsx
│   │   │   ├── ProfileCard.jsx
│   │   │   ├── DecisionPanel.jsx
│   │   │   ├── BiasTracker.jsx
│   │   │   ├── MemoryBank.jsx
│   │   │   └── AttributeTag.jsx
│   │   ├── pages/
│   │   │   └── App.jsx
│   │   ├── api/
│   │   │   └── client.js          ← axios calls to FastAPI
│   │   └── main.jsx
│   └── package.json
│
├── backend/
│   ├── main.py                    ← FastAPI app entry
│   ├── routers/
│   │   ├── decision.py            ← /generate-profile, /decide
│   │   ├── feedback.py            ← /reward, /penalize
│   │   └── bias.py                ← /analyze-bias, /memory
│   ├── services/
│   │   ├── gemini_service.py      ← Gemini Flash calls
│   │   ├── aif360_service.py      ← AIF360 bias measurement
│   │   ├── rl_memory.py           ← RL reward/penalty engine
│   │   └── attribute_classifier.py ← Normal/Redundant/Ambiguous
│   ├── schemas/
│   │   ├── job_schema.py
│   │   ├── loan_schema.py
│   │   └── college_schema.py
│   ├── data/
│   │   └── memory_bank.json       ← Persisted RL memory
│   └── requirements.txt
│
└── IMPLEMENTATION.md
```

---

## 5. Phase-by-Phase Implementation

---

### ✅ PHASE 1 — Backend: AIF360 Bias Detection Core

**Start here. This is your foundation.**

#### Step 1.1 — Install dependencies

```bash
pip install fastapi uvicorn aif360 pandas numpy scikit-learn google-generativeai
```

#### Step 1.2 — Create `aif360_service.py`

This service takes a batch of decisions + profiles and computes bias metrics.

```python
# backend/services/aif360_service.py
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd

def compute_bias_metrics(records: list, protected_attr: str, label_col: str = "decision"):
    """
    records: list of dicts with profile attributes + decision (0/1)
    protected_attr: e.g., "gender", "age_group", "race"
    Returns: disparate_impact, statistical_parity_diff
    """
    df = pd.DataFrame(records)
    df[label_col] = df[label_col].map({"Hire": 1, "Reject": 0,
                                        "Approve": 1, "Deny": 0,
                                        "Admit": 1, "Waitlist": 0}).fillna(df[label_col])
    
    dataset = BinaryLabelDataset(
        df=df,
        label_names=[label_col],
        protected_attribute_names=[protected_attr],
        favorable_label=1,
        unfavorable_label=0
    )
    
    # Privileged = majority group (e.g., male=1)
    privileged = [{protected_attr: df[protected_attr].mode()[0]}]
    unprivileged = [{protected_attr: 1 - df[protected_attr].mode()[0]}]
    
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=privileged,
        unprivileged_groups=unprivileged
    )
    
    return {
        "disparate_impact": round(metric.disparate_impact(), 4),
        "statistical_parity_diff": round(metric.statistical_parity_difference(), 4),
        "bias_detected": metric.disparate_impact() < 0.8 or metric.disparate_impact() > 1.25
    }
```

> **AIF360 Rule of Thumb:**
> - `disparate_impact < 0.8` → Negative bias against unprivileged group
> - `disparate_impact > 1.25` → Positive bias (over-favoring)
> - `0.8 to 1.25` → Acceptable range (80% rule from US employment law)

---

### ✅ PHASE 2 — RL Reward/Penalty Memory System

This is the **core innovation** of BiasScope — a reinforcement learning memory that learns from human feedback.

#### How it works:
- User clicks **Reward** → attribute gets added to "trusted" list for that domain
- User clicks **Penalize** → attribute gets added to "bias risk" list
- Each new decision is **weighted** using this memory before being shown to user
- Memory gets **injected into the Gemini prompt** so AI is aware of past feedback

#### Step 2.1 — Create `rl_memory.py`

```python
# backend/services/rl_memory.py
import json
import os
from collections import defaultdict

MEMORY_FILE = "data/memory_bank.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {
        "job": {"rewarded": [], "penalized": [], "ambiguous": []},
        "loan": {"rewarded": [], "penalized": [], "ambiguous": []},
        "college": {"rewarded": [], "penalized": [], "ambiguous": []}
    }

def save_memory(memory):
    os.makedirs("data", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def reward_attribute(domain: str, attribute: str):
    memory = load_memory()
    if attribute not in memory[domain]["rewarded"]:
        memory[domain]["rewarded"].append(attribute)
    # Remove from penalized if exists
    memory[domain]["penalized"] = [a for a in memory[domain]["penalized"] if a != attribute]
    save_memory(memory)

def penalize_attribute(domain: str, attribute: str):
    memory = load_memory()
    if attribute not in memory[domain]["penalized"]:
        memory[domain]["penalized"].append(attribute)
    # Remove from rewarded if exists
    memory[domain]["rewarded"] = [a for a in memory[domain]["rewarded"] if a != attribute]
    save_memory(memory)

def get_memory_context(domain: str) -> str:
    """Returns a string to inject into Gemini prompt"""
    memory = load_memory()
    d = memory.get(domain, {})
    lines = []
    if d.get("rewarded"):
        lines.append(f"TRUSTED attributes (use these heavily): {', '.join(d['rewarded'])}")
    if d.get("penalized"):
        lines.append(f"BIASED attributes (avoid weighting these): {', '.join(d['penalized'])}")
    if d.get("ambiguous"):
        lines.append(f"AMBIGUOUS attributes (flag these): {', '.join(d['ambiguous'])}")
    return "\n".join(lines) if lines else "No prior feedback yet."
```

---

### ✅ PHASE 3 — Attribute Classifier (Normal / Redundant / Ambiguous)

This is your **original academic contribution** — classifying which attributes are safe vs risky.

#### Classification Rules:

| Class | Definition | Examples |
|---|---|---|
| **Normal** | Directly relevant to the decision, no proxy for protected attributes | Skills, GPA, Experience |
| **Redundant** | Duplicates another attribute, adds no new info | Both "city" and "zip code" |
| **Ambiguous** | Correlates with protected attributes (proxy bias) | Zip code (→ race), university name (→ wealth) |

#### Step 3.1 — Create `attribute_classifier.py`

```python
# backend/services/attribute_classifier.py

# Known ambiguous attributes per domain (proxy bias risks)
AMBIGUOUS_MAP = {
    "job": ["zip_code", "university_name", "graduation_year", "hobbies", "photo"],
    "loan": ["zip_code", "neighborhood", "employer_name", "spending_category"],
    "college": ["high_school_name", "extracurriculars", "zip_code", "parent_occupation"]
}

# Known redundant pairs (if both present, one is redundant)
REDUNDANT_PAIRS = {
    "job": [("years_experience", "number_of_jobs")],
    "loan": [("monthly_income", "annual_income")],
    "college": [("gpa", "class_rank")]
}

def classify_attributes(domain: str, attributes: dict) -> dict:
    """
    Returns dict with classification for each attribute.
    """
    ambiguous = set(AMBIGUOUS_MAP.get(domain, []))
    result = {}
    
    for attr in attributes:
        if attr in ambiguous:
            result[attr] = "AMBIGUOUS"
        else:
            result[attr] = "NORMAL"
    
    # Check redundancy
    for (a, b) in REDUNDANT_PAIRS.get(domain, []):
        if a in result and b in result:
            result[b] = "REDUNDANT"  # second one is redundant
    
    return result
```

> **For the hackathon:** Start with hardcoded ambiguous lists. Later you can use Gemini to dynamically detect proxy bias.

---

### ✅ PHASE 4 — Frontend (React + Vite)

Your UI from the screenshots is already solid. Here's what to build/enhance:

#### Step 4.1 — Setup

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install axios tailwindcss
```

#### Step 4.2 — Key Components to Build

**`ProfileCard.jsx`** — Displays generated profile with attribute classifications color-coded:
- 🟢 Green badge = NORMAL
- 🟡 Yellow badge = AMBIGUOUS  
- 🔴 Red badge = REDUNDANT

**`DecisionPanel.jsx`** — Shows:
- AI Decision (Hire/Reject etc.)
- Weighted attributes (your existing UI from screenshot)
- Reward / Penalize buttons

**`BiasTracker.jsx`** — Right panel showing:
- Total patterns detected
- Neg bias count / Pos bias count / Caution count
- Detected attributes list

**`MemoryBank.jsx`** — The "AI Learning Memory" tab:
- Shows rewarded/penalized lists per domain
- Live updates when user gives feedback

---

### ✅ PHASE 5 — Gemini API Integration

#### Step 5.1 — Create `gemini_service.py`

```python
# backend/services/gemini_service.py
import google.generativeai as genai
import json
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def get_decision_with_weights(profile: dict, domain: str, memory_context: str) -> dict:
    prompt = f"""
You are an AI decision system for {domain} evaluation.

PAST FEEDBACK FROM HUMAN REVIEWERS:
{memory_context}

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

Your task:
1. Make a decision: {get_decision_labels(domain)}
2. List the TOP 3 attributes you gave most weight to, with percentage weights that sum to 100%
3. Give a brief reason for each attribute's weight

Respond ONLY in this JSON format:
{{
  "decision": "...",
  "weighted_attributes": [
    {{"attribute": "...", "weight": 40, "reason": "..."}},
    {{"attribute": "...", "weight": 35, "reason": "..."}},
    {{"attribute": "...", "weight": 25, "reason": "..."}}
  ]
}}
"""
    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(text)

def get_decision_labels(domain: str) -> str:
    labels = {
        "job": "Hire or Reject",
        "loan": "Approve or Deny",
        "college": "Admit or Waitlist"
    }
    return labels.get(domain, "Approve or Reject")
```

---

### ✅ PHASE 6 — FastAPI Routes (Connecting Everything)

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import decision, feedback, bias

app = FastAPI(title="BiasScope API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(decision.router, prefix="/api")
app.include_router(feedback.router, prefix="/api")
app.include_router(bias.router, prefix="/api")
```

```python
# backend/routers/decision.py
from fastapi import APIRouter
from services.gemini_service import get_decision_with_weights
from services.rl_memory import get_memory_context
from services.attribute_classifier import classify_attributes
import random

router = APIRouter()

@router.post("/decide")
def decide(domain: str, profile: dict):
    memory_context = get_memory_context(domain)
    result = get_decision_with_weights(profile, domain, memory_context)
    classifications = classify_attributes(domain, profile)
    return {
        "decision": result["decision"],
        "weighted_attributes": result["weighted_attributes"],
        "attribute_classifications": classifications,
        "memory_context_used": memory_context
    }

@router.get("/generate-profile")
def generate_profile(domain: str):
    # Return a random profile for the domain
    profiles = get_domain_profiles(domain)
    return random.choice(profiles)
```

---

## 6. Domain Schemas

### Job Domain Profile

```json
{
  "name": "Alex Johnson",
  "age": 28,
  "gender": "Female",
  "years_experience": 6,
  "previous_industry": "Finance",
  "highest_education": "Bachelor's",
  "university_name": "State University",
  "skills": ["Python", "SQL", "Excel"],
  "zip_code": "94102",
  "applied_role": "Data Analyst"
}
```

### Loan Domain Profile

```json
{
  "age": 35,
  "gender": "Male",
  "annual_income": 65000,
  "credit_score": 720,
  "employment_status": "Full-time",
  "loan_amount_requested": 20000,
  "loan_purpose": "Home improvement",
  "neighborhood": "Eastside",
  "existing_debt": 5000
}
```

### College Admission Profile

```json
{
  "age": 18,
  "gender": "Female",
  "gpa": 3.8,
  "sat_score": 1380,
  "high_school_name": "Lincoln Public School",
  "extracurriculars": ["Debate", "Volunteer"],
  "parent_occupation": "Teacher",
  "zip_code": "10023",
  "applied_major": "Computer Science"
}
```

---

## 7. AIF360 Integration Notes

AIF360 works on **datasets** (batches), not single decisions. For the prototype:

- **Store every decision made in session** in a local list
- After 10+ decisions, run AIF360 to compute `disparate_impact`
- Display this in the **Bias Tracker** panel on the right
- The protected attributes to check per domain:

| Domain | Protected Attributes to Check |
|---|---|
| Job | gender, age_group |
| Loan | gender, race (if available) |
| College | gender, parent_income_bracket |

> ⚠️ **Important:** For the hackathon prototype, you can simulate the AIF360 dataset using the session's decisions. You don't need a real external dataset.

---

## 8. RL Logic Explained Simply

Think of it like **training a teacher**:

```
Round 1: AI sees profile → decides based on general knowledge
         → User says "that's biased" (penalize gender weight)
         → Memory: "gender is a bias risk in Job domain"

Round 2: AI sees new profile → memory injected into prompt
         → AI now ignores gender, weights experience more
         → User says "good decision" (reward experience weight)
         → Memory: "experience is trusted in Job domain"

Round N: AI has a refined bias-aware decision profile
         → Bias Tracker shows patterns detected over time
```

This is the **reward shaping** concept from RL — you're not training weights directly, you're shaping the AI's behavior through prompt engineering guided by human feedback.

---

## 9. Ambiguity Engine

The ambiguity detection is your **academic differentiator**. Here's how to explain it:

**Why does ambiguity matter?**
Some attributes seem neutral but are **proxies** for protected attributes:
- `zip_code` → correlates with race (due to historical housing segregation)
- `university_name` → correlates with wealth/privilege
- `graduation_year` → can proxy for age discrimination

**Your system flags these automatically** and shows them in yellow in the UI, letting humans decide whether to penalize them.

For the submission, this can be shown as:
1. Profile is generated
2. Ambiguous attributes are highlighted in yellow
3. User can choose to penalize them — memory learns
4. Future decisions automatically down-weight those attributes

---

## 10. What to Build First (Priority Order)

```
Week 1 - Core Logic:
  [ ] 1. FastAPI skeleton + CORS setup
  [ ] 2. Domain profile generators (hardcoded random profiles)
  [ ] 3. Gemini decision endpoint (already have this)
  [ ] 4. RL memory bank (reward/penalize + JSON persistence)
  [ ] 5. Attribute classifier (hardcoded ambiguous lists)

Week 2 - Integration:
  [ ] 6. AIF360 bias metrics on session data
  [ ] 7. Connect React frontend to FastAPI
  [ ] 8. Color-coded attribute badges (Normal/Ambiguous/Redundant)
  [ ] 9. Memory Bank UI (Insights tab - already designed)
  [ ] 10. Bias Tracker live update

Week 3 - Polish for Submission:
  [ ] 11. Cross-domain bias comparison
  [ ] 12. Demo video walkthrough
  [ ] 13. Prototype deck (PPT)
  [ ] 14. GitHub README with architecture diagram
```

---

## 11. Submission Checklist

Per the evaluation criteria from your screenshot:

- [ ] **Prototype Deck (PPT)** — Problem → Solution → Architecture → Demo screenshots → Impact
- [ ] **Video Demo** — Walk through all 3 domains, show reward/penalize memory learning, show AIF360 bias score changing
- [ ] **Working Prototype** — Must run live, all 3 domains functional
- [ ] **GitHub Repo** — Clean README, folder structure, setup instructions

---

## 🚀 Quick Start Commands

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
# → Opens at http://localhost:5173
```

---

*Built for GDG Solution Challenge 2026 ·Neuro Sparks· RVCE*
