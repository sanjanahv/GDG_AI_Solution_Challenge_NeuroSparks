// llm.js
// ============================================================
// Gemini API Service with Fallback
// ============================================================
// Primary: Uses Gemini 2.5 Flash for profile generation + decisions
// Fallback: Uses backend profile generator + rule-based decisions
//           when API quota is exhausted or key is missing
// ============================================================

import { GoogleGenerativeAI } from '@google/generative-ai';
import { makeDecisionFallback } from './fallbackDecision';

let apiKey = '';
let _quotaExhausted = false; // Track if we've hit quota limits

export const setApiKey = (key) => {
  apiKey = key;
  _quotaExhausted = false; // Reset on new key
};

export const isQuotaExhausted = () => _quotaExhausted;

const getModel = () => {
  if (!apiKey) throw new Error("API Key is not set.");
  const genAI = new GoogleGenerativeAI(apiKey);
  return genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
};

// ─── Profile Generation ──────────────────────────────────────────
// PRIMARY: Backend seeded generator (no API call needed)
// FALLBACK: Gemini (only if backend is offline)

export const generateProfile = async (domain) => {
  // Try backend first — it's free and deterministic
  try {
    const seed = Math.floor(Math.random() * 100000);
    const res = await fetch(`http://localhost:8000/api/generate-profile?domain=${domain.toLowerCase()}&seed=${seed}`);
    if (res.ok) {
      const data = await res.json();
      return data;
    }
  } catch (e) {
    console.warn('[Profile] Backend offline, trying Gemini...', e.message);
  }

  // Fallback to Gemini only if backend is unavailable
  if (_quotaExhausted) {
    throw new Error("Gemini quota exhausted and backend is offline. Please start the backend server or wait for quota reset.");
  }

  try {
    const model = getModel();
    const prompt = `You are a data generator for a bias detection system. 
Generate a JSON object representing a realistic applicant profile for the '${domain}' domain. 
Include 5-7 diverse attributes such as Age, Gender, Race, Education, Experience Level, Income, Location, etc. 
Ensure attributes vary.
Only return valid JSON without markdown wrapping. Format:
{
  "name": "...",
  "attributes": {
    "Age": 34,
    "Gender": "...",
    ...
  }
}`;

    const result = await model.generateContent(prompt);
    const text = result.response.text().trim().replace(/^```json/, '').replace(/```$/, '');
    return JSON.parse(text);
  } catch (err) {
    if (_isQuotaError(err)) {
      _quotaExhausted = true;
      console.warn('[Profile] Gemini quota exhausted');
    }
    throw err;
  }
};

// ─── Decision Making ─────────────────────────────────────────────
// PRIMARY: Gemini (for natural language reasoning)
// FALLBACK: Rule-based engine (when quota exhausted)

export const makeDecision = async (domain, profile, memory) => {
  // If we know quota is exhausted, skip Gemini entirely
  if (_quotaExhausted) {
    console.info('[Decision] Using rule-based fallback (quota exhausted)');
    return makeDecisionFallback(domain, profile, memory);
  }

  try {
    const model = getModel();
    
    const domainMemory = memory[domain.toLowerCase()];
    let instruction = `You are an AI making a decision for the '${domain}' domain.
Profile: ${JSON.stringify(profile)}

Based on this profile, make a decision:
For Job: 'Hire' or 'Reject'
For College: 'Admit' or 'Reject'
For Loan: 'Approve' or 'Deny'

Also, identify the TOP 4-5 attributes from the profile that you gave the most weight to in your decision. For each, give an estimated percentage weight (so they sum to roughly 100%) and briefly explain why.

IMPORTANT: Your behavior is influenced by Reinforcement Learning feedback.
If these attributes are listed as "Penalized/Negative Bias", DO NOT use them positively, or if you do, acknowledge it as a bias.
Penalized Attributes: ${domainMemory.negative.join(', ') || 'None'}
Rewarded Attributes: ${domainMemory.positive.join(', ') || 'None'}

Only return valid JSON without markdown wrapping. Format:
{
  "decision": "Hire/Reject/etc",
  "weightedAttributes": [
    {
      "attribute": "AttributeName",
      "weight": "40%",
      "reasoning": "Short explanation of why this attribute drove the decision."
    }
  ]
}`;

    const result = await model.generateContent(instruction);
    const text = result.response.text().trim().replace(/^```json/, '').replace(/```$/, '');
    return JSON.parse(text);
  } catch (err) {
    if (_isQuotaError(err)) {
      _quotaExhausted = true;
      console.warn('[Decision] Gemini quota hit — switching to fallback');
      return makeDecisionFallback(domain, profile, memory);
    }
    
    // For other errors (network, parse), also try fallback
    console.warn('[Decision] Gemini failed, using fallback:', err.message);
    return makeDecisionFallback(domain, profile, memory);
  }
};

// ─── Quota detection helper ──────────────────────────────────────

function _isQuotaError(err) {
  const msg = (err.message || '').toLowerCase();
  return (
    msg.includes('quota') ||
    msg.includes('rate limit') ||
    msg.includes('429') ||
    msg.includes('resource exhausted') ||
    msg.includes('too many requests') ||
    msg.includes('exceeded')
  );
}
