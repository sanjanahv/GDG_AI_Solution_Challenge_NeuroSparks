// services/api.js
// ============================================================
// Backend API Client for AIF360 Bias Analysis
// ============================================================
// This module handles all calls to the FastAPI backend.
// The existing llm.js (Gemini calls) and rl.js (localStorage)
// remain unchanged — this ADDS the AIF360 layer on top.
// ============================================================

const API_BASE = 'http://localhost:8000/api';

/**
 * Send session decisions to AIF360 for bias analysis.
 * @param {string} domain - 'job', 'college', or 'loan'
 * @param {Array} records - Array of {attributes: {...}, decision: "Hire"/"Reject"/etc.}
 * @param {string} protectedAttribute - e.g., "Gender", "Age", "Race"
 * @returns {Object} Bias metrics (disparate_impact, statistical_parity_diff, etc.)
 */
export const analyzeBias = async (domain, records, protectedAttribute = 'Gender') => {
  try {
    const res = await fetch(`${API_BASE}/analyze-bias`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        domain,
        records,
        protected_attribute: protectedAttribute,
      }),
    });
    return await res.json();
  } catch (err) {
    console.error('AIF360 analysis failed:', err);
    return { error: err.message, bias_detected: false };
  }
};

/**
 * Apply AIF360 post-processing to correct biased decisions.
 * @param {string} domain
 * @param {Array} records
 * @param {string} protectedAttribute
 * @param {string} algorithm - 'calibrated_eq_odds', 'eq_odds', or 'reject_option'
 * @returns {Object} Original metrics, corrected metrics, corrected decisions
 */
export const applyCorrection = async (
  domain,
  records,
  protectedAttribute = 'Gender',
  algorithm = 'calibrated_eq_odds'
) => {
  try {
    const res = await fetch(`${API_BASE}/apply-correction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        domain,
        records,
        protected_attribute: protectedAttribute,
        algorithm,
      }),
    });
    return await res.json();
  } catch (err) {
    console.error('AIF360 correction failed:', err);
    return { success: false, error: err.message };
  }
};

/**
 * Get AIF360's recommendation for which algorithm to use.
 * @param {string} domain
 * @param {Array} records
 * @param {string} protectedAttribute
 */
export const getRecommendation = async (domain, records, protectedAttribute = 'Gender') => {
  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        domain,
        records,
        protected_attribute: protectedAttribute,
      }),
    });
    return await res.json();
  } catch (err) {
    console.error('Recommendation failed:', err);
    return { recommendation: 'none', reason: err.message };
  }
};

/**
 * Classify profile attributes as NORMAL/AMBIGUOUS/REDUNDANT/PROTECTED.
 */
export const classifyAttributes = async (domain, attributes) => {
  try {
    const res = await fetch(`${API_BASE}/classify-attributes?domain=${domain}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(attributes),
    });
    return await res.json();
  } catch (err) {
    console.error('Classification failed:', err);
    return { classifications: {}, counts: {} };
  }
};

/**
 * Store a decision in the backend session for later AIF360 analysis.
 */
export const storeDecision = async (decisionRecord) => {
  try {
    const res = await fetch(`${API_BASE}/store-decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(decisionRecord),
    });
    return await res.json();
  } catch (err) {
    console.error('Store decision failed:', err);
    return { status: 'error', error: err.message };
  }
};

/**
 * Sync frontend RL memory with backend.
 */
export const syncMemory = async (memory) => {
  try {
    const res = await fetch(`${API_BASE}/sync-memory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memory }),
    });
    return await res.json();
  } catch (err) {
    console.error('Memory sync failed:', err);
    return null;
  }
};

/**
 * Reward an attribute via the backend.
 */
export const rewardAttribute = async (domain, attribute) => {
  try {
    const res = await fetch(`${API_BASE}/reward`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain, attribute }),
    });
    return await res.json();
  } catch (err) {
    console.error('Reward failed:', err);
    return null;
  }
};

/**
 * Penalize an attribute via the backend.
 */
export const penalizeAttribute = async (domain, attribute) => {
  try {
    const res = await fetch(`${API_BASE}/penalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain, attribute }),
    });
    return await res.json();
  } catch (err) {
    console.error('Penalize failed:', err);
    return null;
  }
};

/**
 * Check if the backend is running.
 */
export const healthCheck = async () => {
  try {
    const res = await fetch('http://localhost:8000/', { method: 'GET' });
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
};
