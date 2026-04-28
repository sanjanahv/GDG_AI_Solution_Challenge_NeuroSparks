# BiasScope — Phased Implementation Plan

> **Rule:** Each phase touches a **completely separate set of files**. No two phases modify the same file. You can execute phases in order, or work on non-adjacent phases in parallel.

---

## Phase Overview

| Phase | Name | Files Touched | Estimated Time | Dependencies |
|---|---|---|---|---|
| **1** | RL Memory Fixes | `src/services/rl.js` | 30 min | None |
| **2** | Attribute Classifier Service | `src/services/attributeClassifier.js` (**NEW**) | 30 min | None |
| **3** | Backend Core (FastAPI + AIF360) | ALL files in `backend/` (**NEW folder**) | 3-4 hours | None |
| **4** | Frontend UI Enhancements | `src/App.jsx` + `src/index.css` | 2-3 hours | Phase 1 & 2 done |
| **5** | Frontend ↔ Backend Integration | `src/services/llm.js` + `src/api/client.js` (**NEW**) | 1-2 hours | Phase 3 & 4 done |
| **6** | Advanced Backend Features | NEW files in `backend/services/` & `backend/routers/` | 3-4 hours | Phase 3 done |
| **7** | Polish & Submission Assets | `index.html` + `README.md` (**NEW**) | 1 hour | All above done |

---

## Phase 1 — RL Memory Fixes

### Files Modified
| Action | File |
|---|---|
| MODIFY | `src/services/rl.js` |

### What Changes

#### 1.1 Add `ambiguous` list to memory schema
The default memory structure currently only has `positive` and `negative` lists. Add a third `ambiguous` list per domain to track proxy-bias attributes.

```diff
 return {
-  job:     { positive: [], negative: [] },
-  college: { positive: [], negative: [] },
-  loan:    { positive: [], negative: [] }
+  job:     { positive: [], negative: [], ambiguous: [] },
+  college: { positive: [], negative: [], ambiguous: [] },
+  loan:    { positive: [], negative: [], ambiguous: [] }
 };
```

#### 1.2 Add conflict resolution to `addFeedback()`
When rewarding an attribute, remove it from the `negative` list (and vice versa). Currently both lists can contain the same attribute — this is a bug.

```diff
 export const addFeedback = (domain, attribute, type) => {
   const memory = getMemory();
   const d = domain.toLowerCase();
   if (!memory[d]) memory[d] = { positive: [], negative: [], ambiguous: [] };
+
+  // Conflict resolution: remove from opposite list
+  if (type === 'positive') {
+    memory[d].negative = memory[d].negative.filter(a => a !== attribute);
+  } else if (type === 'negative') {
+    memory[d].positive = memory[d].positive.filter(a => a !== attribute);
+  }
+
   if (!memory[d][type].includes(attribute)) {
     memory[d][type].push(attribute);
   }
   saveMemory(memory);
   return memory;
 };
```

#### 1.3 Fix `cautionCount` to use real data
Replace the hardcoded `Math.floor(negBiasCount * 1.5)` with an actual count of ambiguous attributes from the memory bank.

```diff
-cautionCount: Math.floor(negBiasCount * 1.5),
+cautionCount: memory.job.ambiguous.length + memory.college.ambiguous.length + memory.loan.ambiguous.length,
```

Also add ambiguous attributes to the `allAttributes` array:
```diff
+memory[d].ambiguous.forEach(attr => allAttributes.push({ attr, type: 'caution' }));
```

#### 1.4 Add export/import memory functions
```js
export const exportMemory = () => {
  const memory = getMemory();
  const blob = new Blob([JSON.stringify(memory, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'biasscope_memory.json';
  a.click();
  URL.revokeObjectURL(url);
};

export const importMemory = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const memory = JSON.parse(e.target.result);
        saveMemory(memory);
        resolve(memory);
      } catch (err) { reject(err); }
    };
    reader.readAsText(file);
  });
};
```

---

## Phase 2 — Attribute Classifier Service

### Files Modified
| Action | File |
|---|---|
| NEW | `src/services/attributeClassifier.js` |

### What This File Does

This is the **academic differentiator** of the project. It classifies every profile attribute into one of three categories:

| Class | Color | Meaning | Examples |
|---|---|---|---|
| **NORMAL** | 🟢 Green | Safe, directly relevant to decision | Skills, GPA, Experience, Credit Score |
| **AMBIGUOUS** | 🟡 Yellow | Proxy for protected attributes | zip_code → race, university_name → wealth |
| **REDUNDANT** | 🔴 Red | Duplicates another attribute | annual_income + monthly_income |

### Implementation Details

1. **Hardcoded ambiguous maps** per domain (job, loan, college) — lists of attributes known to be proxy-bias risks
2. **Redundant pair detection** — if both attributes in a known pair exist, the second is marked redundant
3. **Fuzzy matching** — attribute names from Gemini output (e.g., "Zip Code", "zip_code", "ZIP") are normalized before lookup
4. **Exportable function** `classifyAttributes(domain, attributesObject)` → returns `{ attrName: "NORMAL"|"AMBIGUOUS"|"REDUNDANT" }`

### Proxy Bias Explanations
The file also exports an explanations map, so the UI can show *why* an attribute is flagged:

```js
export const PROXY_EXPLANATIONS = {
  zip_code: "Correlates with race due to historical housing segregation patterns.",
  university_name: "Correlates with socioeconomic status and wealth/privilege.",
  graduation_year: "Can serve as a proxy for age discrimination.",
  high_school_name: "Correlates with neighborhood wealth and racial demographics.",
  parent_occupation: "Proxy for family socioeconomic status.",
  neighborhood: "Correlates with race and income level.",
  // ...
};
```

---

## Phase 3 — Backend Core (FastAPI + AIF360)

### Files Modified
| Action | File |
|---|---|
| NEW | `backend/main.py` |
| NEW | `backend/requirements.txt` |
| NEW | `backend/services/aif360_service.py` |
| NEW | `backend/services/rl_memory.py` |
| NEW | `backend/services/attribute_classifier.py` |
| NEW | `backend/services/fairness_scorer.py` |
| NEW | `backend/routers/decision.py` |
| NEW | `backend/routers/feedback.py` |
| NEW | `backend/routers/bias.py` |
| NEW | `backend/schemas/domains.py` |
| NEW | `backend/data/memory_bank.json` |

### What Each File Does

#### `main.py`
- FastAPI app entry point
- CORS middleware (allow all origins for dev)
- Include 3 routers: decision, feedback, bias
- Startup event to ensure `data/` directory exists

#### `requirements.txt`
```
fastapi
uvicorn
aif360
pandas
numpy
scikit-learn
google-generativeai
pydantic
```

#### `services/aif360_service.py`
- `compute_bias_metrics(records, protected_attr)` → returns `{ disparate_impact, statistical_parity_diff, bias_detected }`
- `apply_reweighing(records, protected_attr)` → applies AIF360 Reweighing algorithm to de-bias dataset
- `apply_calibrated_equalized_odds(records, protected_attr)` → applies CalibratedEqOddsPostprocessing
- Works on **batched session data** (needs 5+ decisions per domain)

#### `services/rl_memory.py`
- `load_memory()` / `save_memory(memory)` — JSON file persistence
- `reward_attribute(domain, attr)` — add to rewarded, remove from penalized
- `penalize_attribute(domain, attr)` — add to penalized, remove from rewarded
- `flag_ambiguous(domain, attr)` — add to ambiguous list
- `get_memory_context(domain)` → string injected into Gemini prompt
- `sync_from_frontend(frontend_memory)` — accepts frontend localStorage dump, merges into backend

#### `services/attribute_classifier.py`
- Python version of Phase 2's JS classifier
- Same hardcoded ambiguous maps + redundant pairs
- `classify_attributes(domain, attributes)` → dict of classifications

#### `services/fairness_scorer.py` (10-term score)
- `score_single_decision(domain, profile, decision, weighted_attrs, memory, session_decisions)` → returns `{ total_score, letter_grade, breakdown: [...10 terms] }`
- `grade_session(domain, session_decisions, memory)` → returns `{ grade, criteria: { disparate_impact_closeness, outcome_balance, attribute_hygiene, feedback_engagement, rl_alignment, proxy_resistance } }`
- Each of the 10 terms is independently computed and signed (see implementation.md §12.1)

#### `routers/decision.py`
- `POST /api/decide` — takes domain + profile, returns decision + weighted attrs + classifications + fairness score
- `POST /api/score-decision` — score a single decision against fairness criteria
- `GET /api/session-grade/{domain}` — grade entire session

#### `routers/feedback.py`
- `POST /api/reward` — reward an attribute
- `POST /api/penalize` — penalize an attribute
- `POST /api/flag-proxy` — flag as ambiguous
- `GET /api/memory/{domain}` — get current memory for domain
- `POST /api/memory/sync` — sync frontend memory to backend

#### `routers/bias.py`
- `POST /api/analyze-bias` — run AIF360 on session data, return disparate_impact
- `POST /api/apply-correction` — apply AIF360 post-processing (Reweighing or CalibratedEqOdds)
- `GET /api/bias-report/{domain}` — full bias report for domain

#### `schemas/domains.py`
- Pydantic models for: `Profile`, `Decision`, `FeedbackRequest`, `BiasReport`, `FairnessScore`
- Domain-specific profile schemas (Job, Loan, College) with field validation

---

## Phase 4 — Frontend UI Enhancements

### Files Modified
| Action | File |
|---|---|
| MODIFY | `src/App.jsx` |
| MODIFY | `src/index.css` |

### What Changes

#### 4.1 Import & use attribute classifier (from Phase 2)
- Import `classifyAttributes` and `PROXY_EXPLANATIONS` from `./services/attributeClassifier`
- After a profile is generated, compute classifications for all attributes
- Pass classifications to the profile card rendering

#### 4.2 Color-coded badges on profile attributes
Each attribute in the profile card gets a colored badge:
- 🟢 **NORMAL** — green badge
- 🟡 **AMBIGUOUS** — yellow badge with tooltip showing proxy explanation
- 🔴 **REDUNDANT** — red badge

#### 4.3 Auto-flag ambiguous attributes
When a profile is generated, any AMBIGUOUS-classified attribute is automatically added to the `ambiguous` memory list via `addFeedback(domain, attr, 'ambiguous')`.

#### 4.4 Wire up the Save button
The `Save` icon is already imported but unused. Add an "Export Memory" button in the header that downloads the RL memory as a JSON file.

#### 4.5 Add "Ambiguous" column to Memory Bank (Insights tab)
The Insights tab currently shows only "Rewarded" and "Penalized" lists. Add a third section showing "Ambiguous (Proxy Risks)" per domain in yellow.

#### 4.6 Fairness Score gauge (new section)
After each decision, if the backend is available, show a fairness score gauge:
- Score 0-1 displayed as a colored arc
- Letter grade (A/B/C/D/F)
- Expandable 10-term breakdown

#### 4.7 CSS additions to `index.css`
- `.badge`, `.badge-normal`, `.badge-ambiguous`, `.badge-redundant` styles
- `.tooltip` styles for proxy explanations
- `.fairness-gauge` styles for the score display
- `.fairness-breakdown` styles for the 10-term list

---

## Phase 5 — Frontend ↔ Backend Integration

### Files Modified
| Action | File |
|---|---|
| MODIFY | `src/services/llm.js` |
| NEW | `src/api/client.js` |

### What Changes

#### `src/api/client.js` (NEW)
- Axios/fetch wrapper for all backend API calls
- Base URL: `http://localhost:8000/api`
- Functions:
  - `postDecision(domain, profile)` → calls `POST /api/decide`
  - `postReward(domain, attribute)` → calls `POST /api/reward`
  - `postPenalize(domain, attribute)` → calls `POST /api/penalize`
  - `postFlagProxy(domain, attribute)` → calls `POST /api/flag-proxy`
  - `getSessionGrade(domain)` → calls `GET /api/session-grade/{domain}`
  - `analyzeBias(domain, records)` → calls `POST /api/analyze-bias`
  - `syncMemory(memory)` → calls `POST /api/memory/sync`
- Includes fallback: if backend is unreachable, return `null` so frontend can degrade gracefully to client-only mode

#### `src/services/llm.js` (MODIFY)
- Add a `useBackend` flag (auto-detected by pinging `http://localhost:8000/docs`)
- If backend is available:
  - Profile generation still happens via Gemini (client-side) — keeps it fast
  - After decision, call `client.postDecision()` to get classifications + fairness score from backend
  - Feedback actions call backend sync endpoint
- If backend is NOT available:
  - Everything works exactly as it does now (pure client-side)
  - This ensures the app never breaks if backend isn't running

---

## Phase 6 — Advanced Backend Features

### Files Modified
| Action | File |
|---|---|
| NEW | `backend/services/profile_generator.py` |
| NEW | `backend/services/ablation.py` |
| NEW | `backend/services/baselines.py` |
| NEW | `backend/routers/session.py` |
| NEW | `backend/data/calibration.json` |

### What Each File Does

#### `services/profile_generator.py`
- Seeded profile generation: `generate_profile(domain, seed, difficulty)`
- 3 difficulty levels:
  - **Easy** (5 decisions): Obvious merit gaps, no proxies, AI obviously uses protected attributes
  - **Medium** (10 decisions): Similar merit, 1 proxy injected, RL memory active
  - **Hard** (15 decisions): Proxy-heavy, hidden classifications, partial observability
- Uses isolated `random.Random(seed)` for reproducibility

#### `services/ablation.py`
- Runs paired-seed ablation studies
- Disables one fairness mechanic at a time (gender check, age check, proxy detection, redundancy filter, RL memory)
- Computes Δ in disparate impact vs. full-protection baseline
- Outputs ranked table: which mechanic matters most

#### `services/baselines.py`
- 4 comparison policies:
  - `do_nothing`: accept all AI decisions, no feedback
  - `random_feedback`: random reward/penalize
  - `current_session`: user's actual behavior
  - `attribute_blind`: ignore all protected + proxy attrs
- Runs each policy over n seeds, returns mean DI for comparison

#### `routers/session.py` (OpenEnv API)
- `POST /api/session/reset` — start new session with domain + difficulty + seed
- `POST /api/session/step` — submit one action
- `GET /api/session/score` — get current session grade
- `GET /api/session/state` — full session state

#### `data/calibration.json`
- Real-world bias statistics for narrative context:
  - EEOC: Gender discrimination = 27.1% of charges
  - HMDA: Black applicant denial rate 2.5× white
  - NACAC: Legacy admit rate 3× non-legacy

---

## Phase 7 — Polish & Submission Assets

### Files Modified
| Action | File |
|---|---|
| MODIFY | `index.html` |
| NEW | `README.md` |

### What Changes

#### `index.html`
- Add proper `<title>`, `<meta description>`, Open Graph tags
- Add favicon reference
- Add Google Fonts link (Inter or Outfit)

#### `README.md`
- Project overview + architecture diagram (Mermaid)
- Setup instructions (frontend + backend)
- Screenshots of the UI
- Link to demo video
- License + credits

---

## File Ownership Matrix (Conflict-Free Guarantee)

| File | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 |
|---|---|---|---|---|---|---|---|
| `src/services/rl.js` | ✏️ | | | | | | |
| `src/services/attributeClassifier.js` | | ✏️ | | | | | |
| `backend/*` (all core files) | | | ✏️ | | | | |
| `src/App.jsx` | | | | ✏️ | | | |
| `src/index.css` | | | | ✏️ | | | |
| `src/services/llm.js` | | | | | ✏️ | | |
| `src/api/client.js` | | | | | ✏️ | | |
| `backend/services/profile_generator.py` | | | | | | ✏️ | |
| `backend/services/ablation.py` | | | | | | ✏️ | |
| `backend/services/baselines.py` | | | | | | ✏️ | |
| `backend/routers/session.py` | | | | | | ✏️ | |
| `backend/data/calibration.json` | | | | | | ✏️ | |
| `index.html` | | | | | | | ✏️ |
| `README.md` | | | | | | | ✏️ |

> ✅ **No file appears in more than one phase.** Phases are fully independent.

---

## Execution Order

```
Phase 1 ──┐
Phase 2 ──┤──→ Phase 4 ──→ Phase 5 ──→ Phase 7
Phase 3 ──┘                    ↑
                         Phase 6 ──┘
```

- **Phases 1, 2, 3** can run in parallel (no file conflicts)
- **Phase 4** needs Phase 1 + 2 done (imports from those files)
- **Phase 5** needs Phase 3 + 4 done (connects frontend to backend)
- **Phase 6** needs Phase 3 done (extends the backend)
- **Phase 7** is last (polish)
