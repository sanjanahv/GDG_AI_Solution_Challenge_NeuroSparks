<div align="center">

# ⚖️ BiasScope — AI Bias Auditing & De-Biasing Platform

> **Google Developer Groups (GDG) Solution Challenge · Team NeuroSparks · RVCE**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![Gemini](https://img.shields.io/badge/Google_Gemini_AI-8E75C2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![AIF360](https://img.shields.io/badge/IBM_AIF360-Fairness-blue?style=for-the-badge)](https://aif360.mybluemix.net/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)

<p align="center">
  <b>An interactive full-stack auditing engine that diagnoses, quantifies, and mitigates algorithmic bias in AI decision-making systems across hiring, lending, and college admissions.</b>
</p>

</div>

---

## 📌 Executive Summary

Algorithmic bias is one of the most critical vulnerabilities in automated decision-making. When machine learning models screen job candidates, evaluate loan applications, or assess university admissions, they often perpetuate historical inequities. Even when explicit protected attributes (e.g., race, gender) are omitted, models frequently exploit **proxy attributes** (e.g., ZIP codes as proxies for race, or educational institutions as proxies for socioeconomic status).

**BiasScope** provides an end-to-end framework to:
1. **Simulate high-stakes decisions** using Google Gemini across diverse applicant profiles.
2. **Detect latent proxy bias** using automated attribute classification.
3. **Quantify statistical fairness** using **IBM AI Fairness 360 (AIF360)** and a comprehensive 10-term fairness scoring algorithm.
4. **Dynamically de-bias future inferences** through a persistent **Reinforcement Learning (RL) memory feedback loop**.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────┐
                               │     Google Gemini Flash API      │
                               │  (Generates applicant profiles   │
                               │   & context-aware decisions)     │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
┌───────────────────────────────┐         HTTP / JSON         ┌─────────────────────────────────┐
│     Interactive UI Dashboard  │ ◄─────────────────────────► │       FastAPI Backend Engine    │
│  (React 19 + Vite + Tailwind) │                             │   (Python + AIF360 + Pydantic)  │
│                               │                             │                                 │
│  • Real-time Auditing Panel   │                             │  • Disparate Impact Analysis    │
│  • Proxy-Bias Classification  │                             │  • 10-Term Fairness Scorer      │
│  • Session Grading (A–F)      │                             │  • Persistent RL Memory Bank    │
│  • Dark Glassmorphic Design   │                             │  • Ablation & Baseline Studies  │
└───────────────────────────────┘                             └─────────────────────────────────┘
```

---

## 🌟 Core Features & Modules

### 1. 🎯 Three Real-World High-Stakes Domains
- 💼 **Recruitment & Hiring:** Audits automated resume screening algorithms for gender, pedigree, and age biases.
- 🏦 **Credit & Loan Underwriting:** Identifies discriminatory lending criteria and socioeconomic proxy markers.
- 🎓 **Higher Education Admissions:** Analyzes holistic admissions algorithms for systemic demographic skews.

### 2. 🔍 Proxy-Bias Classification Engine
Categorizes every input attribute into four distinct analytical classifications:
* 🟢 **Normal:** Legitimate, performance-relevant merit criteria.
* 🟡 **Ambiguous:** Borderline criteria that warrant contextual auditing.
* 🔴 **Redundant:** Non-informative attributes introducing decision noise.
* 🩷 **Protected / Proxy:** Latent indicators acting as statistical proxies for protected demographic classes.

### 3. 📊 IBM AIF360 & 10-Term Fairness Scoring
* Computes mathematically sound fairness metrics including **Disparate Impact (DI)**, **Demographic Parity**, and **Equal Opportunity Difference**.
* Grades each session from **A to F** based on:
  * Disparate Impact compliance (30%)
  * Attribute selection balance (20%)
  * Decision hygiene (15%)
  * Demographic distribution consistency (35%)

### 4. 🔄 Reinforcement Learning (RL) De-Biasing Loop
* Human auditors can reward or penalize individual decisions and flag proxy attributes.
* Feedback is written directly to a persistent **RL Memory Bank** (`backend/data/memory_bank.json`).
* Learned constraints are dynamically injected into future inference contexts, guiding the LLM toward demonstrably fairer outcomes without requiring full model retraining.

### 5. 🔬 Ablation Studies & Baselines
* Compare audit sessions against **4 naive baseline policies** (Random, Merit-Only, Demographic-Blind, Majority-Favored).
* Run ablation tests to isolate the individual contribution of each de-biasing mechanism.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, Vite, Lucide Icons, Modern CSS Glassmorphism |
| **Backend Engine** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **Fairness & ML** | IBM AI Fairness 360 (AIF360), Scikit-Learn, Pandas, NumPy |
| **Generative AI** | Google Gemini Flash API (`@google/generative-ai`) |
| **Data & Persistence**| JSON-backed RL Memory Bank, Calibration Dataset |

---

## 📂 Repository Structure

```text
GDG_AI_Solution_Challenge_NeuroSparks/
├── biasscope/
│   ├── backend/                     # Python FastAPI microservice
│   │   ├── main.py                  # API routes, middleware, CORS
│   │   ├── services/
│   │   │   ├── aif360_service.py    # IBM AIF360 disparity computation
│   │   │   ├── fairness_scorer.py   # 10-term fairness metric scoring
│   │   │   ├── rl_memory.py         # Reinforcement learning memory bank
│   │   │   ├── attribute_classifier.py # 4-tier proxy-bias classifier
│   │   │   ├── profile_generator.py # Deterministic seeded profile generator
│   │   │   ├── ablation.py          # Ablation test suite
│   │   │   └── baselines.py         # Naive baseline comparison models
│   │   ├── routers/                 # Modular API route controllers
│   │   └── data/                    # Memory bank & calibration datasets
│   │
│   └── frontend/                    # Modern React 19 client application
│       ├── src/
│       │   ├── App.jsx              # Main auditing dashboard UI
│       │   ├── services/
│       │   │   ├── llm.js           # Gemini API integration
│       │   │   ├── rl.js            # Client-side RL memory handler
│       │   │   ├── attributeClassifier.js # Front-end proxy tagger
│       │   │   └── api.js           # FastAPI client integration
│       │   └── index.css            # Custom design tokens & dark styling
│       └── package.json
│
├── implementation.md                # Full engineering specification
├── plan.md                          # Phased development roadmap
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started

### 1. Frontend Setup

```bash
# Navigate to the frontend directory
cd biasscope/frontend

# Install dependencies
npm install

# Launch the development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser and enter your **Google Gemini API Key** when prompted.

---

### 2. Backend Setup (Enables IBM AIF360 Engine)

```bash
# Navigate to the backend directory
cd biasscope/backend

# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

The backend documentation will be live at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 👥 Team NeuroSparks

Developed for the **Google Developer Groups (GDG) AI Solution Challenge** by students of **RV College of Engineering (RVCE)**:

* 👩‍💻 **Sanjana H V** 
* 👨‍💻 **Saish Ambar**
* 👨‍💻 **Vikas Prakash Ambore** 

---

<div align="center">
  <sub>Built with ❤️ by Team NeuroSparks · RV College of Engineering, Bengaluru</sub>
</div>
