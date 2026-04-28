// services/fallbackDecision.js
// ============================================================
// Rule-Based Fallback Decision Engine
// ============================================================
// When Gemini API quota is exhausted or unavailable, this engine
// provides deterministic decisions based on domain-specific rules.
// It uses the attribute classifier to weight merit vs. protected
// attributes and produces decisions with transparent reasoning.
// ============================================================

import { classifyWithDetails, CLASS } from './attributeClassifier';

// ─── Domain-specific merit scoring rules ─────────────────────────

const MERIT_RULES = {
  job: {
    positiveDecision: 'Hire',
    negativeDecision: 'Reject',
    scoring: {
      // Attribute name (normalized) → scoring function(value) → score 0-1
      experience: (v) => {
        const years = typeof v === 'string' ? parseInt(v) : Number(v);
        if (isNaN(years)) return 0.5;
        return Math.min(years / 15, 1.0);
      },
      years_of_experience: (v) => {
        const years = typeof v === 'string' ? parseInt(v) : Number(v);
        if (isNaN(years)) return 0.5;
        return Math.min(years / 15, 1.0);
      },
      education_level: (v) => {
        const map = { "PhD": 1.0, "Master's": 0.85, "Bachelor's": 0.7, "Associate": 0.5, "High School": 0.3 };
        const str = String(v);
        for (const [key, score] of Object.entries(map)) {
          if (str.toLowerCase().includes(key.toLowerCase())) return score;
        }
        return 0.5;
      },
      skills: (v) => {
        if (Array.isArray(v)) return Math.min(v.length / 6, 1.0);
        if (typeof v === 'string') return Math.min(v.split(',').length / 6, 1.0);
        return 0.5;
      },
      technical_skills: (v) => {
        if (Array.isArray(v)) return Math.min(v.length / 6, 1.0);
        return 0.5;
      },
      performance_rating: (v) => {
        const n = Number(v);
        return isNaN(n) ? 0.5 : Math.min(n / 10, 1.0);
      },
      certifications: (v) => {
        if (Array.isArray(v)) return Math.min(v.length / 4, 1.0);
        return v ? 0.7 : 0.3;
      },
    },
    threshold: 0.55,
  },

  loan: {
    positiveDecision: 'Approve',
    negativeDecision: 'Deny',
    scoring: {
      credit_score: (v) => {
        const n = typeof v === 'string' ? parseInt(v) : Number(v);
        if (isNaN(n)) return 0.5;
        return Math.max(0, Math.min((n - 300) / 550, 1.0));
      },
      annual_income: (v) => {
        const str = String(v).replace(/[$,K]/gi, '');
        const n = parseFloat(str);
        if (isNaN(n)) return 0.5;
        const income = n < 1000 ? n * 1000 : n;
        return Math.min(income / 200000, 1.0);
      },
      employment_status: (v) => {
        const str = String(v).toLowerCase();
        if (str.includes('full')) return 0.9;
        if (str.includes('part')) return 0.6;
        if (str.includes('self')) return 0.7;
        if (str.includes('unemploy')) return 0.2;
        return 0.5;
      },
      debt_to_income: (v) => {
        const str = String(v).replace('%', '');
        const n = parseFloat(str);
        if (isNaN(n)) return 0.5;
        return Math.max(0, 1.0 - (n / 60));
      },
      payment_history: (v) => {
        const str = String(v).toLowerCase();
        if (str.includes('excellent') || str.includes('strong')) return 0.95;
        if (str.includes('good')) return 0.75;
        if (str.includes('fair') || str.includes('average')) return 0.5;
        if (str.includes('poor')) return 0.2;
        return 0.5;
      },
      employment_length: (v) => {
        const years = parseInt(String(v));
        if (isNaN(years)) return 0.5;
        return Math.min(years / 20, 1.0);
      },
    },
    threshold: 0.55,
  },

  college: {
    positiveDecision: 'Admit',
    negativeDecision: 'Reject',
    scoring: {
      gpa: (v) => {
        const n = parseFloat(String(v));
        if (isNaN(n)) return 0.5;
        return Math.min(n / 4.0, 1.0);
      },
      sat_score: (v) => {
        const n = parseInt(String(v));
        if (isNaN(n)) return 0.5;
        return Math.max(0, (n - 600) / 1000);
      },
      act_score: (v) => {
        const n = parseInt(String(v));
        if (isNaN(n)) return 0.5;
        return Math.max(0, (n - 12) / 24);
      },
      ap_classes: (v) => {
        const n = parseInt(String(v));
        if (isNaN(n)) return 0.5;
        return Math.min(n / 10, 1.0);
      },
      volunteer_hours: (v) => {
        const n = parseInt(String(v));
        if (isNaN(n)) return 0.5;
        return Math.min(n / 500, 1.0);
      },
      awards: (v) => {
        if (Array.isArray(v)) return Math.min(v.length / 5, 1.0);
        return v ? 0.6 : 0.3;
      },
      essay_score: (v) => {
        const n = parseFloat(String(v));
        if (isNaN(n)) return 0.5;
        return Math.min(n / 10, 1.0);
      },
      extracurriculars: (v) => {
        if (Array.isArray(v)) return Math.min(v.length / 6, 1.0);
        if (typeof v === 'string') return Math.min(v.split(',').length / 6, 1.0);
        return 0.5;
      },
    },
    threshold: 0.55,
  },
};

// ─── Normalize attribute name for lookup ─────────────────────────

const normalize = (name) =>
  name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');

// ─── Core fallback decision function ─────────────────────────────

/**
 * Make a decision without Gemini. Uses domain-specific rules and
 * the attribute classifier to produce transparent weighted decisions.
 *
 * @param {string} domain - 'Job', 'Loan', or 'College'
 * @param {Object} profile - { name, attributes: {...} }
 * @param {Object} memory - RL memory from localStorage
 * @returns {Object} - { decision, weightedAttributes: [...] }
 */
export const makeDecisionFallback = (domain, profile, memory) => {
  const d = domain.toLowerCase();
  const rules = MERIT_RULES[d] || MERIT_RULES.job;
  const attrs = profile.attributes || {};
  const classifications = classifyWithDetails(d, attrs);
  const domainMemory = memory[d] || { positive: [], negative: [] };

  // Score each attribute
  const scored = [];
  for (const [attrName, attrValue] of Object.entries(attrs)) {
    const norm = normalize(attrName);
    const cls = classifications[attrName]?.class || 'NORMAL';

    // Skip protected attributes for scoring
    if (['gender', 'race', 'ethnicity', 'religion', 'age', 'sex', 'disability',
         'nationality', 'sexual_orientation'].includes(norm)) {
      continue;
    }

    // Find a matching scoring rule
    let score = null;
    let reasoning = '';

    if (rules.scoring[norm]) {
      score = rules.scoring[norm](attrValue);
      reasoning = `Rule-based score for ${attrName}: ${(score * 100).toFixed(0)}%`;
    } else if (cls === CLASS.AMBIGUOUS) {
      // Ambiguous attributes get 0 weight — don't use them
      score = null;
      reasoning = `Proxy attribute — excluded from scoring`;
    } else if (cls === CLASS.REDUNDANT) {
      score = null;
      reasoning = `Redundant attribute — excluded`;
    } else {
      // Unknown/unscored attribute — give it a neutral 0.5
      score = 0.5;
      reasoning = `No specific rule — neutral weight applied`;
    }

    // Apply RL memory adjustments
    if (domainMemory.negative.includes(attrName)) {
      score = null; // Penalized — exclude from decision
      reasoning = `Penalized by RL memory — excluded`;
    }
    if (domainMemory.positive.includes(attrName) && score !== null) {
      score = Math.min(score * 1.3, 1.0); // Rewarded — boost weight
      reasoning += ` (boosted by RL reward)`;
    }

    if (score !== null) {
      scored.push({ attribute: attrName, score, reasoning, cls });
    }
  }

  // Sort by score descending, pick top 4-5
  scored.sort((a, b) => Math.abs(b.score - 0.5) - Math.abs(a.score - 0.5));
  const topAttrs = scored.slice(0, Math.min(5, scored.length));

  // Compute overall decision
  const avgScore = topAttrs.length > 0
    ? topAttrs.reduce((sum, a) => sum + a.score, 0) / topAttrs.length
    : 0.5;

  const decision = avgScore >= rules.threshold
    ? rules.positiveDecision
    : rules.negativeDecision;

  // Compute weights that sum to ~100%
  const totalWeight = topAttrs.reduce((sum, a) => sum + Math.abs(a.score - 0.5) + 0.1, 0);
  const weightedAttributes = topAttrs.map(a => {
    const rawWeight = (Math.abs(a.score - 0.5) + 0.1) / totalWeight;
    return {
      attribute: a.attribute,
      weight: `${Math.round(rawWeight * 100)}%`,
      reasoning: a.reasoning,
    };
  });

  return {
    decision,
    weightedAttributes,
    _meta: {
      method: 'fallback_rule_based',
      avgScore: Math.round(avgScore * 100) / 100,
      threshold: rules.threshold,
      attributesScored: scored.length,
    },
  };
};
