# BiasScope

BiasScope is a full-stack auditing tool for detecting and mitigating algorithmic bias in AI decision-making systems. It was built for the Google Developer Groups (GDG) Solution Challenge 2026 by Team NeuroSparks at RVCE.

The system simulates automated decisions across three high-stakes domains — job hiring, loan approval, and college admissions — using Google Gemini to generate applicant profiles and make decisions. A human auditor interacts with each decision, classifying input attributes and providing feedback. That feedback is stored in a persistent reinforcement learning memory bank and injected into future prompts to steer the model toward fairer outcomes.

---

## How It Works

1. Gemini generates a realistic applicant profile for the selected domain (job, loan, or college).
2. Gemini makes a decision (hire / reject, approve / decline, admit / deny) and lists the attributes it weighted.
3. The auditor classifies each attribute as Normal, Ambiguous, Redundant, or a Protected/Proxy attribute.
4. The auditor rewards or penalizes the decision. Proxy attributes can be explicitly flagged.
5. Feedback is written to `backend/data/memory_bank.json`. On the next decision, this memory is retrieved and injected into the Gemini prompt as explicit constraints.
6. After a session, IBM AIF360 computes Disparate Impact and Statistical Parity Difference across all decisions. The session is graded A–F.

---

## Architecture

```
biasscope/
├── backend/                         FastAPI (Python)
│   ├── main.py                      Entry point, CORS, router registration
│   ├── routers/
│   │   ├── decision.py              POST /api/decide, POST /api/session-grade
│   │   ├── feedback.py              POST /api/reward, /api/penalize, /api/flag-proxy
│   │   ├── bias.py                  POST /api/analyze-bias, /api/apply-correction
│   │   └── session.py              POST /api/session/reset, /step, /ablation
│   ├── services/
│   │   ├── aif360_service.py        Disparate Impact and Reweighing via IBM AIF360
│   │   ├── fairness_scorer.py       11-term per-decision fairness scoring
│   │   ├── rl_memory.py             JSON-persisted reward/penalty memory bank
│   │   ├── attribute_classifier.py  4-category attribute classification
│   │   ├── profile_generator.py     Seeded, difficulty-scaled profile generation
│   │   ├── ablation.py              Ablation study runner
│   │   └── baselines.py             Four naive comparison policies
│   ├── schemas/domains.py           Pydantic request/response models
│   └── data/
│       ├── memory_bank.json         RL memory (persisted across sessions)
│       └── calibration.json         Real-world bias base rates
│
└── frontend/                        React 19 + Vite
    └── src/
        ├── App.jsx                  Main dashboard — 3 tabs
        ├── index.css                Dark glassmorphism theme
        └── services/
            ├── llm.js               Gemini API calls
            ├── rl.js                Client-side RL memory (mirrors backend)
            ├── attributeClassifier.js  Front-end proxy classification
            └── api.js               FastAPI client
```

---

## Fairness Scoring

Each decision receives an **11-term decomposed fairness score**. Terms include:

- Protected attribute avoidance
- Proxy attribute penalty (e.g. Zip Code, University Name, Surname)
- Redundant attribute penalty
- Merit attribute coverage
- Decision-outcome balance (approve/reject ratio)
- Unknown attribute risk (attributes the system has not seen and cannot verify)

Sessions are graded A–F based on weighted aggregates across all decisions, with Disparate Impact carrying the largest weight (30%).

---

## RL Memory Bank

The memory bank is a JSON file keyed by domain:

```json
{
  "job":     { "positive": ["Years of Experience"], "negative": ["Zip Code", "Surname"] },
  "loan":    { "positive": ["Credit Score"],        "negative": ["Neighborhood"] },
  "college": { "positive": ["GPA"],                 "negative": ["High School Name"] }
}
```

`get_memory_context(domain)` serializes this into a string that is appended to the Gemini system prompt before each new decision. The model is explicitly told which attributes have been flagged as biased proxies and instructed to ignore them.

The frontend `rl.js` maintains a client-side mirror. On session start, the frontend syncs with the backend via `POST /api/session/reset`, merging both copies by taking the union of positive and negative sets and resolving conflicts in favor of the most recent frontend signal.

---

## Domains and Attribute Classification

### Attribute Categories

| Category  | Meaning                                                                 |
| :-------- | :---------------------------------------------------------------------- |
| Normal    | Legitimate merit criterion for the domain                               |
| Ambiguous | Borderline — may be legitimate or a soft proxy depending on context     |
| Redundant | Duplicate or noise — adds no information beyond another present attribute|
| Protected/Proxy | Directly protected class or a known statistical proxy for one    |

### Domain-specific Proxy Examples

| Domain  | Proxy Attribute      | Protected Class It Proxies |
| :------ | :------------------- | :------------------------- |
| Job     | Zip Code             | Race / Socioeconomic status |
| Job     | University Name      | Wealth / Social class      |
| Loan    | Neighborhood         | Race                       |
| College | High School Name     | Socioeconomic status       |
| College | Parent Occupation    | Class / Wealth             |

---

## Running the Project

### Frontend

```bash
cd biasscope/frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. Enter a Gemini API key when prompted. The frontend works standalone without the backend — AIF360 features and persistent RL memory will be disabled.

### Backend

```bash
cd biasscope/backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`.

---

## Tech Stack

| Component        | Technology                                      |
| :--------------- | :---------------------------------------------- |
| Frontend         | React 19, Vite, Lucide Icons                    |
| Backend          | Python, FastAPI, Uvicorn, Pydantic              |
| Fairness Engine  | IBM AI Fairness 360 (AIF360), Pandas, NumPy     |
| AI               | Google Gemini Flash (`@google/generative-ai`)   |
| Persistence      | JSON flat files (memory bank, calibration data) |

---

## Team

Sanjana H V, Saish Ambar, Vikas Prakash Ambore — RV College of Engineering, Bengaluru.
GDG Solution Challenge 2026.
